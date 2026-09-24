use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use serde::Serialize;
use tauri::{Manager, RunEvent, State};

#[cfg(windows)]
use win32job::{ExtendedLimitInfo, Job};

const READY_MARKER: &str = "CORPI_CORE_READY";
// Cold starts (fresh binary, Defender first-scan, uv resolving the venv) can be
// slow; the UI stays on "Connecting…" until the handshake lands. On timeout,
// early exit, or any other failure the core tree is stopped.
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
    /// Windows: kill-on-close job holding the app and (by inheritance) every
    /// core process. Created empty and joined by the app BEFORE the core
    /// spawns, so uv and the Python interpreter are members from creation —
    /// there is no assignment race — and the kernel terminates the tree when
    /// the app process ends, including crash paths.
    #[cfg(windows)]
    job: Mutex<Option<Job>>,
}

fn log_spawn_event(message: &str) {
    // A GUI app has no console to rely on under every launcher; keep the spawn
    // path observable on disk.
    let path = std::env::temp_dir().join("corpi-core-spawn.log");
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        use std::io::Write;
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let _ = writeln!(file, "[{stamp}] {message}");
    }
}

/// Platform hook run BEFORE the core spawns. On Windows this must succeed for
/// startup to continue: without the job the tree cannot be stopped reliably.
#[cfg(windows)]
fn prepare_for_spawn(state: &State<'_, CoreState>) -> Result<(), String> {
    let mut info = ExtendedLimitInfo::new();
    info.limit_kill_on_job_close();
    let job = Job::create_with_limit_info(&info)
        .map_err(|error| format!("core job creation failed: {error}"))?;
    job.assign_current_process()
        .map_err(|error| format!("core job self-assignment failed: {error}"))?;
    *state.job.lock().unwrap() = Some(job);
    Ok(())
}

#[cfg(unix)]
fn prepare_for_spawn(_state: &State<'_, CoreState>) -> Result<(), String> {
    Ok(())
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

    let mut command = Command::new("uv");
    command
        .args(["run", "python", "-m", "core.api.server"])
        .current_dir(&core_dir)
        .env("CORPI_CORE_TOKEN", &token)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    #[cfg(unix)]
    {
        // Own process group: the group kill stops uv, the venv trampoline, and
        // the Python server together.
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }

    match command.spawn() {
        Ok(child) => Ok((child, token)),
        Err(error) => {
            let message = format!("failed to spawn core: {error}");
            log_spawn_event(&message);
            Err(message)
        }
    }
}

/// Stops the core tree deterministically. Safe to call at any point, including
/// while the handshake is still in flight and repeatedly.
#[cfg(windows)]
fn stop_core(state: &State<'_, CoreState>) {
    // Terminate every job member other than this app: uv, the venv trampoline,
    // and the Python server die together, however far startup got.
    if let Some(job) = state.job.lock().unwrap().as_ref() {
        terminate_other_job_members(job);
    }
    let mut guard = state.child.lock().unwrap();
    if let Some(ref mut child) = *guard {
        let _ = child.kill();
        let _ = child.wait();
    }
    *guard = None;
}

#[cfg(unix)]
fn stop_core(state: &State<'_, CoreState>) {
    let mut guard = state.child.lock().unwrap();
    if let Some(ref mut child) = *guard {
        // Kill the whole process group (uv + interpreter), then reap uv.
        unsafe {
            libc::kill(-(child.id() as i32), libc::SIGKILL);
        }
        let _ = child.kill();
        let _ = child.wait();
    }
    *guard = None;
}

#[cfg(windows)]
fn terminate_other_job_members(job: &Job) {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{OpenProcess, TerminateProcess, PROCESS_TERMINATE};

    let Ok(pids) = job.query_process_id_list() else {
        return;
    };
    let own_pid = std::process::id() as usize;
    for pid in pids {
        if pid == own_pid {
            continue;
        }
        unsafe {
            let handle = OpenProcess(PROCESS_TERMINATE, 0, pid as u32);
            if !handle.is_null() {
                let _ = TerminateProcess(handle, 1);
                let _ = CloseHandle(handle);
            }
        }
    }
}

/// Waits until the core reports READY (which, since the server prints it only
/// after binding its socket, means it accepts connections). The child stays
/// registered in the state the whole time — locks are never held across the
/// wait — so exit or failure paths can always stop it.
fn wait_for_core_ready(state: &State<'_, CoreState>) -> Result<u32, String> {
    let (sender, receiver) = mpsc::channel::<String>();
    {
        let mut guard = state.child.lock().unwrap();
        let child = guard.as_mut().ok_or("core process is not registered")?;
        let stdout = child.stdout.take().ok_or_else(|| {
            let message = "core stdout unavailable".to_string();
            log_spawn_event(&message);
            message
        })?;
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if sender.send(line).is_err() {
                    break;
                }
            }
        });
    }

    let started = Instant::now();
    loop {
        {
            let mut guard = state.child.lock().unwrap();
            match guard.as_mut() {
                Some(child) => {
                    if let Ok(Some(status)) = child.try_wait() {
                        let message = format!("core exited before reporting ready: {status}");
                        log_spawn_event(&message);
                        return Err(message);
                    }
                }
                None => return Err("core process was stopped before it was ready".to_string()),
            }
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
        #[cfg(windows)]
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
                eprintln!("[core] setup thread started");

                if let Err(error) = prepare_for_spawn(&state) {
                    // No usable stop mechanism: do not start the core at all.
                    log_spawn_event(&format!("core startup aborted: {error}"));
                    *state.failure.lock().unwrap() = Some(error);
                    return;
                }
                eprintln!("[core] prepare_for_spawn done");

                let spawn_result = spawn_core();
                let (child, token) = match spawn_result {
                    Ok(result) => result,
                    Err(error) => {
                        log_spawn_event(&format!("failed to start core: {error}"));
                        *state.failure.lock().unwrap() = Some(error);
                        return;
                    }
                };
                eprintln!("[core] spawned pid={}", child.id());

                // Register the child before the handshake: app exit, handshake
                // timeout, or an early core exit can all stop it from here on.
                *state.child.lock().unwrap() = Some(child);
                eprintln!("[core] child registered; waiting for ready");

                match wait_for_core_ready(&state) {
                    Ok(port) => {
                        eprintln!("[core] ready on port {port}");
                        log_spawn_event(&format!("core ready: port={port}"));
                        *state.connection.lock().unwrap() = Some(CoreConnection {
                            base_url: format!("http://127.0.0.1:{port}"),
                            token,
                        });
                    }
                    Err(error) => {
                        eprintln!("[core] startup failed: {error}");
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
