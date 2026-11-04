"""Polling must not take a writer lock or reset durable setup evidence."""

import json
from dataclasses import asdict

import pytest
from test_model_setup import SmallSetup

from core.persistence.errors import ConflictingWriteError
from core.persistence.store import PersistenceStore


@pytest.mark.parametrize("state", ["missing", "downloading", "installed", "failed", "cancelled"])
def test_existing_setup_state_is_readable_while_another_connection_writes(
    store: PersistenceStore, state: str
) -> None:
    service = SmallSetup(store)
    profile = service.profile("minilm-en")
    service.state(profile.id)
    with store._transaction() as connection:
        connection.execute(
            "UPDATE model_installations SET state=?,generation=7,received_bytes=3,"
            "spent_seconds=12.5,transfer_attempts=2,failure_code='retained',updated_at='saved'"
            " WHERE manifest_id=?",
            (state, profile.identity),
        )
    expected = service.state(profile.id)
    # Keep the real write transaction open throughout the poll. A polling writer
    # deterministically expires its existing lock wait instead of reading WAL state.
    with store._transaction() as writer:
        writer.execute(
            "UPDATE model_installations SET received_bytes=4 WHERE manifest_id=?",
            (profile.identity,),
        )
        assert service.state(profile.id) == expected
        assert writer.in_transaction
    assert service.state(profile.id) == {**expected, "received_bytes": 4}


@pytest.mark.parametrize("existing", ["neither", "manifest", "installation"])
def test_setup_state_initializes_only_missing_records(
    store: PersistenceStore, existing: str
) -> None:
    service = SmallSetup(store)
    profile = service.profile("minilm-en")
    manifest = json.dumps(asdict(profile), sort_keys=True)
    with store._transaction() as connection:
        if existing == "manifest":
            connection.execute(
                "INSERT INTO model_manifests VALUES(?,?,?)",
                (profile.identity, profile.id, manifest),
            )
        if existing == "installation":
            connection.execute(
                "INSERT INTO model_installations(manifest_id,profile_id,state,generation,"
                "received_bytes,spent_seconds,transfer_attempts,updated_at)"
                " VALUES(?,?,'failed',7,3,12.5,2,'saved')",
                (profile.identity, profile.id),
            )
    state = service.state(profile.id)
    assert state["state"] == ("failed" if existing == "installation" else "missing")
    assert state["generation"] == (7 if existing == "installation" else 0)
    with store.database.connect() as connection:
        assert connection.execute("SELECT count(*) FROM model_installations").fetchone()[0] == 1
        assert tuple(connection.execute("SELECT * FROM model_manifests").fetchone()) == (
            profile.identity,
            profile.id,
            manifest,
        )
    assert service.state(profile.id) == state


@pytest.mark.parametrize("has_installation", [False, True])
def test_setup_state_still_rejects_conflicting_manifest_without_rewriting_it(
    store: PersistenceStore, has_installation: bool
) -> None:
    service = SmallSetup(store)
    profile = service.profile("minilm-en")
    with store._transaction() as connection:
        connection.execute(
            "INSERT INTO model_manifests VALUES(?,?,'{}')", (profile.identity, profile.id)
        )
        if has_installation:
            connection.execute(
                "INSERT INTO model_installations(manifest_id,profile_id,state,updated_at)"
                " VALUES(?,?,'missing','saved')",
                (profile.identity, profile.id),
            )
    with pytest.raises(ConflictingWriteError, match="identity mismatch"):
        service.state(profile.id)
    with store.database.connect() as connection:
        assert connection.execute("SELECT manifest_json FROM model_manifests").fetchone()[0] == "{}"
        assert connection.execute("SELECT count(*) FROM model_installations").fetchone()[0] == int(
            has_installation
        )
