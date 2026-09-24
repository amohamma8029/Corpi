use std::io::{BufRead, BufReader};
use std::os::windows::io::AsRawHandle;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use serde::Serialize;
use tauri::{Manager, RunEvent, State};
use win32job::{ExtendedLimitInfo, Job};

const READY_MARKER: &str = "CORPI_CORE_READY";
// Cold starts (fresh binary, Defender first-scan, uv resolving the venv) can be
// slow; the UI stays on "Connecting…" until the handshake lands. On timeout,
// early exit, or any other failure the child is stopped by the spawn thread.
const READY_TIMEOUT_MILLIS: u64 = 30_000;
const READ_POLL_MILLIS: u64 = 250;

/// Connection handed to the webview. The token is a per-launch credential:
/// whoever holds it can spend the user's model-API budget under BYOK, so it is
/// never persisted and never leaves the shell/webview pair.
#[derive(Clone, Serialize)]
struct CoreConnection {
    base_url: String,
    token: String,
}

struct CoreState {
    connection: Mutex<Option<CoreConnection>>,
    failure: Mutex<Option<String>>,
    child: Mutex<Option<Child>>,
    /// Job object with kill-on-job-close holding the core's process tree. The
    /// kernel terminates every process in the tree (uv, the venv trampoline,
    /// and the Python server) when this handle is closed — on explicit stop,
    /// on app exit, and even if the app crashes.
    job: Mutex<Option<Job>>,
}

/// Puts the spawned core into a job object that kills its whole process tree
/// when the job handle closes. `uv` spawns a venv trampoline plus the real
/// interpreter, so killing the direct child alone leaves the server running.
fn attach_core_job(child: &Child) -> Result<Job, String> {
    let mut info = ExtendedLimitInfo::new();
    info.limit_kill_on_job_close();
    let job = Job::create_with_limit_info(&info)
        .map_err(|error| format!("core job creation failed: {error}"))?;
    job.assign_process(child.as_raw_handle() as isize)
        .map_err(|error| format!("core job assignment failed: {error}"))?;
    Ok(job)
}

/// Stops the core tree deterministically: close the job handle (terminates the
/// tree), then reap the direct child.
fn stop_core(state: &State<'_, CoreState>) {
    if let Some(job) = state.job.lock().unwrap().take() {
        drop(job);
    }
    let mut guard = state.child.lock().unwrap();
    if let Some(ref mut child) = *guard {
        let _ = child.kill();
        let _ = child.wait();
    }
    *guard = None;
}

fn log_spawn_event(message: &str) {
    // A GUI app has no console to rely on under every launcher; keep the spawn
    // path observable on disk.
    let path = std::env::temp_dir().join("corpi-core-spawn.log");
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        use std::io::Write;
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let _ = writeln!(file, "[{stamp}] {message}");
    }
}

fn spawn_core() -> Result<(Child, String), String> {
    let core_dir = match std::env::var("CORPI_CORE_DIR") {
        Ok(dir) => dir,
        Err(_) => {
            let message =
                "CORPI_CORE_DIR is not set; launch the app through the root justfile".to_string();
            log_spawn_event(&message);
            return Err(message);
        }
    };
    let token = uuid::Uuid::new_v4().simple().to_string();

    let child = match Command::new("uv")
        .args(["run", "python", "-m", "core.api.server"])
        .current_dir(&core_dir)
        .env("CORPI_CORE_TOKEN", &token)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            let message = format!("failed to spawn core: {error}");
            log_spawn_event(&message);
            return Err(message);
        }
    };
    Ok((child, token))
}

/// Waits until the core reports READY (which, since the server prints it only
/// after binding its socket, means it accepts connections). Detects early
/// exits and enforces the timeout; the CALLER owns the child and must stop it
/// when this returns Err.
fn wait_for_core_ready(child: &mut Child) -> Result<u32, String> {
    let stdout = child.stdout.take().ok_or_else(|| {
        let message = "core stdout unavailable".to_string();
        log_spawn_event(&message);
        message
    })?;

    let (sender, receiver) = mpsc::channel::<String>();
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if sender.send(line).is_err() {
                break;
            }
        }
    });

    let started = Instant::now();
    loop {
        if let Ok(Some(status)) = child.try_wait() {
            let message = format!("core exited before reporting ready: {status}");
            log_spawn_event(&message);
            return Err(message);
        }
        if started.elapsed() > Duration::from_millis(READY_TIMEOUT_MILLIS) {
            let message = format!("core did not become ready within {READY_TIMEOUT_MILLIS} ms");
            log_spawn_event(&message);
            return Err(message);
        }
        match receiver.recv_timeout(Duration::from_millis(READ_POLL_MILLIS)) {
            Ok(line) => {
                if let Some(rest) = line.strip_prefix(READY_MARKER) {
                    if let Some(port) = rest
                        .split_whitespace()
                        .find_map(|part| part.strip_prefix("port="))
                        .and_then(|value| value.parse().ok())
                    {
                        return Ok(port);
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                let message = "core stdout closed before reporting ready".to_string();
                log_spawn_event(&message);
                return Err(message);
            }
        }
    }
}

/// Waits (bounded) until the background spawn finishes, so the first UI render
/// already knows whether the core is available.
#[tauri::command]
fn core_connection(state: State<'_, CoreState>) -> Result<CoreConnection, String> {
    let deadline = Instant::now() + Duration::from_millis(READY_TIMEOUT_MILLIS);
    loop {
        if let Some(connection) = state.connection.lock().unwrap().clone() {
            return Ok(connection);
        }
        if let Some(failure) = state.failure.lock().unwrap().clone() {
            return Err(failure);
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "core did not become ready within {READY_TIMEOUT_MILLIS} ms"
            ));
        }
        thread::sleep(Duration::from_millis(READ_POLL_MILLIS));
    }
}

fn main() {
    let state = CoreState {
        connection: Mutex::new(None),
        failure: Mutex::new(None),
        child: Mutex::new(None),
        job: Mutex::new(None),
    };

    tauri::Builder::default()
        .manage(state)
        .setup(|app| {
            let handle = app.handle().clone();
            // Spawning on a thread keeps the blocking handshake off the window
            // setup path; core_connection() gates the UI.
            thread::spawn(move || {
                let state: State<CoreState> = handle.state();
                let spawn_result = spawn_core();
                let (child, token) = match spawn_result {
                    Ok(result) => result,
                    Err(error) => {
                        log_spawn_event(&format!("failed to start core: {error}"));
                        *state.failure.lock().unwrap() = Some(error);
                        return;
                    }
                };

                // Register the child BEFORE the handshake so that app exit,
                // handshake timeout, or an early core exit can all stop it.
                match attach_core_job(&child) {
                    Ok(job) => {
                        *state.job.lock().unwrap() = Some(job);
                    }
                    Err(error) => {
                        // Without the job the plain child kill still runs; the
                        // tree-kill guarantee is weakened only in this
                        // unlikely Windows-API failure case.
                        log_spawn_event(&format!("{error} (falling back to child kill)"));
                    }
                }
                let mut guard = state.child.lock().unwrap();
                *guard = Some(child);
                let ready = wait_for_core_ready(guard.as_mut().unwrap());
                match ready {
                    Ok(port) => {
                        *state.connection.lock().unwrap() = Some(CoreConnection {
                            base_url: format!("http://127.0.0.1:{port}"),
                            token,
                        });
                    }
                    Err(error) => {
                        drop(guard);
                        stop_core(&state);
                        log_spawn_event(&format!("core startup failed: {error}"));
                        *state.failure.lock().unwrap() = Some(error);
                    }
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![core_connection])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                let state: State<CoreState> = app.state();
                stop_core(&state);
            }
        });
}
