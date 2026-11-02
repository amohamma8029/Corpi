import sqlite3

import pytest
from test_storage import seed

from core.persistence.errors import ConflictingWriteError
from core.search.query import QueryService
from core.search.representation import ARABIC, CANONICAL, search_text, selected, settings_for
from core.search.storage import verify_generation


def test_search_only_arabic_folding_leaves_other_unicode_and_canonical_text_unchanged(store):
    canonical = "ﻣﻄـﺮ café ① ﬁ"
    assert search_text(canonical, CANONICAL) == canonical
    assert search_text(canonical, ARABIC) == "مطر café ① ﬁ"
    notebook, _, _, index, _, _ = seed(
        store,
        canonical,
        [1.0] + [0.0] * 383,
        representation=ARABIC,
    )
    result = QueryService(store, store.database.config.data_dir).search(notebook, "مطر", "words")
    assert result["results"][0]["text"] == canonical
    assert store.load_index(index).chunks[0].text == canonical


def test_derived_proof_or_input_identity_cannot_be_downgraded(store):
    _, _, _, index, _, _ = seed(
        store,
        "ﻣﻄـﺮ",
        [1.0] + [0.0] * 383,
        representation=ARABIC,
    )
    with store.database.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE search_generations SET settings_json='{}' WHERE index_version_id=?",
                (index,),
            )
        connection.execute("UPDATE search_text_inputs SET search_text='tampered'")
        with pytest.raises(ConflictingWriteError, match="derived input proof"):
            verify_generation(connection, index)


def test_legacy_input_has_no_derived_fields_and_changed_unicode_rule_is_rejected():
    assert settings_for(CANONICAL) == {} and selected({}) == CANONICAL
    with pytest.raises(ValueError, match="unsupported_search_representation"):
        selected({**settings_for(ARABIC), "search_unicode_version": "changed"})
