"""Versioned search inputs; canonical text and citation coordinates never change."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

CANONICAL = "canonical-v1"
ARABIC = "arabic-forms-tatweel-v1"
_FIELDS = ("search_representation", "search_unicode_version", "search_rule_sha256")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def settings_for(name: str) -> dict[str, str]:
    """Missing settings retain exact legacy identity; derived rules bind Unicode data."""
    if name == CANONICAL:
        return {}
    if name != ARABIC:
        raise ValueError("unsupported_search_representation")
    rules = [ARABIC, "NFKC:FB50-FDFF,FE70-FEFF", "delete:0640", unicodedata.unidata_version]
    digest = hashlib.sha256(json.dumps(rules, separators=(",", ":")).encode()).hexdigest()
    return dict(zip(_FIELDS, (ARABIC, unicodedata.unidata_version, digest), strict=True))


def selected(settings: Mapping) -> str:
    name = settings.get(_FIELDS[0], CANONICAL)
    expected = settings_for(name)
    if name == CANONICAL:
        if any(field in settings for field in _FIELDS[1:]):
            raise ValueError("unsupported_search_representation")
    elif any(settings.get(field) != value for field, value in expected.items()):
        raise ValueError("unsupported_search_representation")
    return name


def search_text(text: str, name: str) -> str:
    settings_for(name)
    if name == CANONICAL:
        return text
    folded = "".join(
        unicodedata.normalize("NFKC", char)
        if 0xFB50 <= ord(char) <= 0xFDFF or 0xFE70 <= ord(char) <= 0xFEFF
        else char
        for char in text
    )
    return folded.replace("\u0640", "")


def batch_input_hash(chunk_ids: Sequence[str], texts: Sequence[str], settings: Mapping) -> str:
    """Length-framed JSON binds order, identities, original bytes and derived bytes."""
    name = selected(settings)
    if not chunk_ids or len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("invalid_search_input_membership")
    rows = [
        [key, text_hash(text), text_hash(search_text(text, name))]
        for key, text in zip(chunk_ids, texts, strict=True)
    ]
    payload = [name, settings_for(name), rows]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class EncodedQuery:
    vector: bytes
    manifest_id: str
    representation: str
    rule_sha256: str | None


def encode_query(
    session, text: str, representation: str = CANONICAL, *, timeout: float = 120
) -> EncodedQuery:
    """Use the supervised session without changing pooling or cancellation behavior."""
    from core.search.storage import validate_vector

    rules = settings_for(representation)
    if not 0 < timeout <= 120:
        raise ValueError("invalid_embedding_timeout")
    vectors = session.embed([search_text(text, representation)], query=True, timeout=timeout)
    if len(vectors) != 1:
        raise ValueError("invalid_vector_count")
    validate_vector(vectors[0], session.profile.dimensions)
    return EncodedQuery(
        vectors[0], session.profile.identity, representation, rules.get("search_rule_sha256")
    )
