use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use serde::Serialize;
use tauri::{Manager, RunEvent, State};

const READY_MARKER: &str = "CORPI_CORE_READY";
// Cold starts (fresh binary, Defender first-scan, uv resolving the venv) can be
// slow; the UI stays on "Connecting…" until the handshake lands.
const READY_TIMEOUT_POLLS: u32 = 300;
const READY_POLL_MILLIS: u64 = 100;

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

fn spawn_core() -> Result<(Child, CoreConnection), String> {
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

    let mut child = match Command::new("uv")
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

    let stdout = child.stdout.take().ok_or_else(|| {
        let message = "core stdout unavailable".to_string();
        log_spawn_event(&message);
        message
    })?;
    let reader = BufReader::new(stdout);
    let mut port: Option<u32> = None;
    for line in reader.lines() {
        let line = match line {
            Ok(line) => line,
            Err(error) => {
                let message = format!("core stdout read failed: {error}");
                log_spawn_event(&message);
                return Err(message);
            }
        };
        if let Some(rest) = line.strip_prefix(READY_MARKER) {
            port = rest
                .split_whitespace()
                .find_map(|part| part.strip_prefix("port="))
                .and_then(|value| value.parse().ok());
            break;
        }
    }

    let port = match port {
        Some(port) => port,
        None => {
            let message = "core did not report a port before its stdout closed".to_string();
            log_spawn_event(&message);
            return Err(message);
        }
    };
    let connection = CoreConnection {
        base_url: format!("http://127.0.0.1:{port}"),
        token,
    };
    log_spawn_event(&format!("core ready: {port}"));
    Ok((child, connection))
}

/// Waits (bounded) until the background spawn finishes, so the first UI render
/// already knows whether the core is available.
#[tauri::command]
fn core_connection(state: State<'_, CoreState>) -> Result<CoreConnection, String> {
    for _ in 0..READY_TIMEOUT_POLLS {
        if let Some(connection) = state.connection.lock().unwrap().clone() {
            return Ok(connection);
        }
        if let Some(failure) = state.failure.lock().unwrap().clone() {
            return Err(failure);
        }
        thread::sleep(Duration::from_millis(READY_POLL_MILLIS));
    }
    Err("core did not become ready within 10 seconds".to_string())
}

fn main() {
    let state = CoreState {
        connection: Mutex::new(None),
        failure: Mutex::new(None),
        child: Mutex::new(None),
    };

    tauri::Builder::default()
        .manage(state)
        .setup(|app| {
            let handle = app.handle().clone();
            // Spawning on a thread keeps the blocking READY-line read off the
            // window setup path; core_connection() gates the UI.
            thread::spawn(move || match spawn_core() {
                Ok((child, connection)) => {
                    let state: State<CoreState> = handle.state();
                    *state.connection.lock().unwrap() = Some(connection);
                    *state.child.lock().unwrap() = Some(child);
                }
                Err(error) => {
                    eprintln!("failed to start core: {error}");
                    let state: State<CoreState> = handle.state();
                    *state.failure.lock().unwrap() = Some(error);
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
                let mut guard = state.child.lock().unwrap();
                if let Some(ref mut child) = *guard {
                    let _ = child.kill();
                    let _ = child.wait();
                }
                *guard = None;
            }
        });
}
