//! Focused launcher tests: exercise the real `core_process` code paths (spawn
//! uv, READY handshake, stop) rather than a re-implementation.
//!
//! Two scenarios, mirroring how the app shuts down:
//! 1. stop after the core is ready;
//! 2. stop while a launch is in progress (startup).
//!
//! Both assert that no `/health` listener and no core process remains. The
//! process-group probe is Unix-specific; Windows shutdown is additionally
//! covered by `just e2e`.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use corpi_desktop_lib::core_process::CoreState;

fn ensure_core_dir() {
    if std::env::var("CORPI_CORE_DIR").is_err() {
        let core_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("core");
        std::env::set_var("CORPI_CORE_DIR", core_dir);
    }
}

fn port_of(base_url: &str) -> u16 {
    base_url
        .rsplit(':')
        .next()
        .expect("port in base_url")
        .parse()
        .expect("numeric port")
}

/// One minimal HTTP GET so the test needs no extra dependency.
fn http_status(port: u16, token: Option<&str>) -> Option<u16> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
    let mut request =
        format!("GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n");
    if let Some(token) = token {
        request.push_str(&format!("Authorization: Bearer {token}\r\n"));
    }
    request.push_str("\r\n");
    stream.write_all(request.as_bytes()).ok()?;
    let mut response = String::new();
    stream.read_to_string(&mut response).ok()?;
    response.split_whitespace().nth(1)?.parse().ok()
}

/// Shutdown can be immediate (SIGKILL on the group) or graceful (the server
/// notices stdin EOF and exits); allow a bounded drain either way.
const SHUTDOWN_DRAIN_TIMEOUT: Duration = Duration::from_secs(10);

fn assert_port_dead(port: u16) {
    let deadline = Instant::now() + SHUTDOWN_DRAIN_TIMEOUT;
    loop {
        if TcpStream::connect(("127.0.0.1", port)).is_err() {
            return;
        }
        if Instant::now() >= deadline {
            panic!("port {port} still accepts connections after stop");
        }
        thread::sleep(Duration::from_millis(50));
    }
}

#[cfg(unix)]
fn assert_group_dead(pgid: u32) {
    let deadline = Instant::now() + SHUTDOWN_DRAIN_TIMEOUT;
    loop {
        // 0 means "does this group have any live members"; ESRCH means none.
        let alive = unsafe { libc::kill(-(pgid as i32), 0) == 0 };
        if !alive {
            return;
        }
        if Instant::now() >= deadline {
            panic!("process group {pgid} still has live members after stop");
        }
        thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn stops_after_the_core_is_ready() {
    ensure_core_dir();
    let state = CoreState::new();
    state.prepare_for_spawn().expect("prepare_for_spawn");

    let connection = state.launch_blocking().expect("core should start");
    let port = port_of(&connection.base_url);
    assert_eq!(http_status(port, None), Some(401));
    assert_eq!(http_status(port, Some(&connection.token)), Some(200));

    let _pid = state.child_pid().expect("child registered after launch");
    state.stop();

    assert_port_dead(port);
    assert!(state.child_pid().is_none());
    #[cfg(unix)]
    assert_group_dead(_pid);

    // Dropping the state would close the kill-on-close job handle, which the
    // kernel answers by terminating every job member — including this test
    // process. The app relies on that at exit; leak it here instead.
    std::mem::forget(state);
}

#[test]
fn stops_while_the_core_is_starting() {
    ensure_core_dir();
    let state = CoreState::new();
    state.prepare_for_spawn().expect("prepare_for_spawn");

    thread::scope(|scope| {
        let launcher = scope.spawn(|| state.launch_blocking());

        // Land inside the startup window: uv is running (or about to be), but
        // the READY handshake has not completed.
        thread::sleep(Duration::from_millis(250));
        let _pid_before_stop = state.child_pid();
        state.stop();

        let result = launcher.join().expect("launcher thread panicked");
        assert!(state.child_pid().is_none(), "child still registered");

        match result {
            Ok(connection) => {
                // The core won the race to readiness; stop must still have
                // killed it.
                assert_port_dead(port_of(&connection.base_url));
            }
            Err(_) => {
                // Launch cancelled or interrupted mid-startup — the expected
                // path. Cleanup is verified below where information exists.
            }
        }

        #[cfg(unix)]
        if let Some(pid) = _pid_before_stop {
            assert_group_dead(pid);
        }
    });

    // A second stop is harmless (idempotent) and a later launch is refused.
    state.stop();
    assert!(
        state.launch_blocking().is_err(),
        "launch after stop must fail"
    );

    std::mem::forget(state); // see note in stops_after_the_core_is_ready
}

#[test]
fn waits_for_an_in_flight_spawn_before_stopping() {
    ensure_core_dir();
    let state = CoreState::new();
    state.prepare_for_spawn().expect("prepare_for_spawn");

    thread::scope(|scope| {
        let launcher = scope.spawn(|| state.launch_blocking());

        // Stop as early as possible: this exercises the lock handoff where
        // `stop` either waits for the spawn to register or cancels a launch
        // that has not spawned yet.
        let started = Instant::now();
        state.stop();
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "stop should not block longer than a slow spawn"
        );

        let result = launcher.join().expect("launcher thread panicked");
        if let Ok(connection) = result {
            assert_port_dead(port_of(&connection.base_url));
        }
        assert!(state.child_pid().is_none());
    });

    std::mem::forget(state); // see note in stops_after_the_core_is_ready
}
