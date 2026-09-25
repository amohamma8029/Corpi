pub mod core_process;

use std::thread;
use std::time::{Duration, Instant};

use tauri::{Manager, RunEvent, State};

use core_process::{log_spawn_event, CoreConnection, CoreState, READ_TIMEOUT_POLL_MILLIS};

/// Waits (bounded) until the background spawn finishes, so the first UI render
/// already knows whether the core is available.
#[tauri::command]
fn core_connection(state: State<'_, CoreState>) -> Result<CoreConnection, String> {
    let deadline = Instant::now() + Duration::from_millis(core_process::READY_TIMEOUT_MILLIS);
    loop {
        if let Some(connection) = state.connection() {
            return Ok(connection);
        }
        if let Some(failure) = state.failure() {
            return Err(failure);
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "core did not become ready within {} ms",
                core_process::READY_TIMEOUT_MILLIS
            ));
        }
        thread::sleep(Duration::from_millis(READ_TIMEOUT_POLL_MILLIS));
    }
}

pub fn run() {
    tauri::Builder::default()
        .manage(CoreState::new())
        .setup(|app| {
            let state: State<CoreState> = app.state();
            let data_dir = match app.path().app_data_dir() {
                Ok(path) => path,
                Err(_) => {
                    let message = "core user app-data directory is unavailable".to_string();
                    log_spawn_event(&message);
                    state.set_failure(message);
                    return Ok(());
                }
            };
            state.set_data_dir(data_dir);

            let handle = app.handle().clone();
            // Spawning on a thread keeps the blocking handshake off the window
            // setup path; core_connection() gates the UI.
            thread::spawn(move || {
                let state: State<CoreState> = handle.state();
                eprintln!("[core] setup thread started");

                if let Err(error) = state.prepare_for_spawn() {
                    // No usable stop mechanism: do not start the core at all.
                    eprintln!("[core] startup aborted: {error}");
                    log_spawn_event(&format!("core startup aborted: {error}"));
                    state.set_failure(error);
                    return;
                }
                eprintln!("[core] prepare_for_spawn done");

                match state.launch_blocking() {
                    Ok(connection) => {
                        eprintln!("[core] ready at {}", connection.base_url);
                    }
                    Err(error) => {
                        eprintln!("[core] startup failed: {error}");
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
                state.stop();
            }
        });
}
