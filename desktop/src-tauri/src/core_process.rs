//! Core process management: spawn `uv` + the Python server, wait for the READY
//! handshake, and stop the whole process tree reliably on any platform.
//!
//! Shutdown design:
//! - The child lock is held across spawn + registration, so `CoreState::stop`
//!   called concurrently waits for an in-flight spawn to finish registering
//!   and then stops the whole tree.
//! - If `stop` ran first, the `stopping` flag makes a pending launch return
//!   without spawning anything.
//! - Windows additionally joins a kill-on-close job before spawning, so uv and
//!   the Python interpreter are members from creation and the kernel stops the
//!   tree even on crash paths; on Unix the core runs in its own process group
//!   and is stopped with a group kill.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use serde::Serialize;

#[cfg(windows)]
use win32job::{ExtendedLimitInfo, Job};

pub const READY_MARKER: &str = "CORPI_CORE_READY";
pub const ERROR_MARKER: &str = "CORPI_CORE_ERROR";
/// Cold starts (fresh binary, Defender first-scan, uv resolving the venv) can
/// be slow; the UI stays on "Connecting…" until the handshake lands. On
/// timeout, early exit, or any other failure the core tree is stopped.
pub const READY_TIMEOUT_MILLIS: u64 = 30_000;
pub const READ_TIMEOUT_POLL_MILLIS: u64 = 250;

/// Connection handed to the webview. The token is a per-launch credential:
/// whoever holds it can spend the user's model-API budget under BYOK, so it is
/// never persisted and never leaves the shell/webview pair.
#[derive(Clone, Serialize)]
pub struct CoreConnection {
    pub base_url: String,
    pub token: String,
}

pub struct CoreState {
    connection: Mutex<Option<CoreConnection>>,
    failure: Mutex<Option<String>>,
    child: Mutex<Option<Child>>,
    data_dir: Mutex<Option<PathBuf>>,
    /// Set when the app starts shutting down; a launch that has not spawned
    /// yet is cancelled instead of starting a process nobody would stop.
    stopping: Mutex<bool>,
    /// Windows: kill-on-close job holding the app and (by inheritance) every
    /// core process. Created and joined by the app BEFORE the core spawns, so
    /// there is no assignment race, and the kernel terminates the tree when
    /// the app process ends, including crash paths.
    #[cfg(windows)]
    job: Mutex<Option<Job>>,
}

impl Default for CoreState {
    fn default() -> Self {
        Self::new()
    }
}

impl CoreState {
    pub fn new() -> Self {
        Self {
            connection: Mutex::new(None),
            failure: Mutex::new(None),
            child: Mutex::new(None),
            data_dir: Mutex::new(None),
            stopping: Mutex::new(false),
            #[cfg(windows)]
            job: Mutex::new(None),
        }
    }

    pub fn connection(&self) -> Option<CoreConnection> {
        self.connection.lock().unwrap().clone()
    }

    pub fn failure(&self) -> Option<String> {
        self.failure.lock().unwrap().clone()
    }

    pub fn set_failure(&self, message: String) {
        *self.failure.lock().unwrap() = Some(message);
    }

    pub fn set_data_dir(&self, path: PathBuf) {
        *self.data_dir.lock().unwrap() = Some(path);
    }

    /// The direct child (uv) pid while a core process is registered.
    pub fn child_pid(&self) -> Option<u32> {
        self.child.lock().unwrap().as_ref().map(Child::id)
    }

    /// Platform hook run BEFORE the core spawns. On Windows this must succeed
    /// or startup aborts: without the job the tree cannot be stopped reliably.
    #[cfg(windows)]
    pub fn prepare_for_spawn(&self) -> Result<(), String> {
        let mut info = ExtendedLimitInfo::new();
        info.limit_kill_on_job_close();
        let job = Job::create_with_limit_info(&info)
            .map_err(|error| format!("core job creation failed: {error}"))?;
        job.assign_current_process()
            .map_err(|error| format!("core job self-assignment failed: {error}"))?;
        *self.job.lock().unwrap() = Some(job);
        Ok(())
    }

    #[cfg(unix)]
    pub fn prepare_for_spawn(&self) -> Result<(), String> {
        Ok(())
    }

    /// Starts the core and blocks until it is ready or startup fails.
    ///
    /// Holds the child lock across spawn + registration so `stop` called
    /// concurrently waits for the launch to reach a killable state; a launch
    /// started after `stop` returns without spawning.
    pub fn launch_blocking(&self) -> Result<CoreConnection, String> {
        let mut guard = self.child.lock().unwrap();
        if *self.stopping.lock().unwrap() {
            return Err("app is shutting down; core launch cancelled".to_string());
        }
        let data_dir = self
            .data_dir
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| "core app-data directory is unavailable".to_string())?;
        let (child, token) = spawn_core(&data_dir)?;
        *guard = Some(child);
        drop(guard);

        match self.wait_for_core_ready() {
            Ok(port) => {
                let connection = CoreConnection {
                    base_url: format!("http://127.0.0.1:{port}"),
                    token,
                };
                *self.connection.lock().unwrap() = Some(connection.clone());
                log_spawn_event(&format!("core ready: port={port}"));
                Ok(connection)
            }
            Err(error) => {
                self.stop();
                log_spawn_event(&format!("core startup failed: {error}"));
                self.set_failure(error.clone());
                Err(error)
            }
        }
    }

    /// Stops the core tree and cancels any launch that has not spawned yet.
    /// Safe at any point and from any thread; blocks until an in-flight spawn
    /// has registered its child.
    pub fn stop(&self) {
        let mut guard = self.child.lock().unwrap();
        *self.stopping.lock().unwrap() = true;
        #[cfg(windows)]
        {
            // Terminate every job member other than this app: uv, the venv
            // trampoline, and the Python server die together, however far
            // startup got. The job stays open; the kernel closes it with the
            // process for crash paths.
            if let Some(job) = self.job.lock().unwrap().as_ref() {
                terminate_other_job_members(job);
            }
        }
        if let Some(ref mut child) = *guard {
            // Close the stdin write end first: the server watches it and shuts
            // down on EOF, which reaches it even when the interpreter was
            // regrouped away from the direct child (uv manages its own child
            // process group on Unix).
            drop(child.stdin.take());
            #[cfg(unix)]
            unsafe {
                // Best effort: kill the whole process group (uv + interpreter)
                // when the interpreter still shares it.
                libc::kill(-(child.id() as i32), libc::SIGKILL);
            }
            let _ = child.kill();
            let _ = child.wait();
        }
        *guard = None;
    }

    /// Waits until the core reports READY (which, since the server prints it
    /// only after binding its socket, means it accepts connections). The child
    /// stays visible to `stop` the whole time — locks are never held across
    /// the wait.
    fn wait_for_core_ready(&self) -> Result<u32, String> {
        let (sender, receiver) = mpsc::channel::<String>();
        {
            let mut guard = self.child.lock().unwrap();
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
            if started.elapsed() > Duration::from_millis(READY_TIMEOUT_MILLIS) {
                let message = format!("core did not become ready within {READY_TIMEOUT_MILLIS} ms");
                log_spawn_event(&message);
                return Err(message);
            }
            match receiver.recv_timeout(Duration::from_millis(READ_TIMEOUT_POLL_MILLIS)) {
                Ok(line) => {
                    if let Some(message) = parse_core_error(&line) {
                        return Err(message);
                    }
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
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    let mut guard = self.child.lock().unwrap();
                    match guard.as_mut() {
                        Some(child) => {
                            if let Ok(Some(status)) = child.try_wait() {
                                let message =
                                    format!("core exited before reporting ready: {status}");
                                log_spawn_event(&message);
                                return Err(message);
                            }
                        }
                        None => {
                            return Err("core process was stopped before it was ready".to_string());
                        }
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    let message = "core stdout closed before reporting ready".to_string();
                    log_spawn_event(&message);
                    return Err(message);
                }
            }
        }
    }
}

fn parse_core_error(line: &str) -> Option<String> {
    let code = line
        .strip_prefix(ERROR_MARKER)?
        .trim()
        .strip_prefix("code=")?
        .split_whitespace()
        .next()?;
    let reason = match code {
        "invalid_data_dir" => "the local data directory is unavailable",
        "unsupported_sqlite" => "the SQLite runtime is unsupported",
        "unsupported_schema" => "the local database schema is unsupported",
        "database_busy" => "the local database remained busy during startup",
        "migration_failed" => "the local database could not be upgraded",
        _ => return None,
    };
    Some(format!("core persistence startup failed: {reason}"))
}

#[cfg(test)]
mod tests {
    use super::{parse_core_error, ERROR_MARKER};

    #[test]
    fn maps_only_the_supported_core_error_codes() {
        for code in [
            "invalid_data_dir",
            "unsupported_sqlite",
            "unsupported_schema",
            "database_busy",
            "migration_failed",
        ] {
            assert!(parse_core_error(&format!("{ERROR_MARKER} code={code}")).is_some());
        }
        assert!(parse_core_error(&format!("{ERROR_MARKER} code=unknown")).is_none());
        assert!(parse_core_error("CORPI_CORE_READY port=1234").is_none());
    }
}

fn spawn_core(data_dir: &Path) -> Result<(Child, String), String> {
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
        .env("CORPI_DATA_DIR", data_dir)
        // The shell holds the stdin pipe; the server exits on EOF, which
        // reaches it even when process-group kills miss the interpreter.
        .env("CORPI_CORE_WATCH_STDIN", "1")
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

pub(crate) fn log_spawn_event(message: &str) {
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
