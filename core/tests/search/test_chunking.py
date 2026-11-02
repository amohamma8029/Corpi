from dataclasses import replace

import pytest

from core.persistence.models import Passage, new_id, utc_timestamp
from core.search.chunking import make_chunks
from core.search.profiles import profiles


def passage(text: str, ordinal: int = 0, page: int = 1) -> Passage:
    return Passage(
        new_id(), new_id(), new_id(), new_id(), ordinal, text, 1, {"page": page}, utc_timestamp()
    )


@pytest.mark.parametrize("preserve", (False, True))
def test_unicode_slices_reconstruct_and_cover_oversized_passages(preserve) -> None:
    profile = replace(profiles()["bge-small-en"], chunk_tokens=32, overlap_tokens=4)
    saved = passage("العربية café 🇫🇷\nrow one | 42\n" * 8)
    chunks = make_chunks(
        [saved], profile, len, lambda text: text, preserve_fitting_passages=preserve
    )
    covered = set()
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 32
        assert chunk.text == profile.separator.join(part.text for part in chunk.mappings)
        assert len({part.passage.id for part in chunk.mappings}) == len(chunk.mappings)
        for part in chunk.mappings:
            assert 0 <= part.start < part.end <= len(saved.text)
            covered.update(range(part.start, part.end))
    assert covered == set(range(len(saved.text)))
    assert saved.text == "العربية café 🇫🇷\nrow one | 42\n" * 8
    # Final overlap by itself must not become a redundant extra chunk.
    assert len(chunks[-1].text) > profile.overlap_tokens


@pytest.mark.parametrize("preserve", (False, True))
def test_adjacent_passages_pack_but_do_not_cross_page_boundaries(preserve) -> None:
    profile = replace(profiles()["bge-small-en"], chunk_tokens=50, overlap_tokens=4)
    a, b, c = (
        passage("First list item"),
        passage("Second list item", 1),
        passage("Later page", 2, 3),
    )
    chunks = make_chunks(
        [a, b, c], profile, len, lambda text: text, preserve_fitting_passages=preserve
    )
    assert [chunk.text for chunk in chunks] == ["First list item\n\nSecond list item", "Later page"]


def test_caps_fail_instead_of_silently_dropping_text() -> None:
    profile = replace(profiles()["bge-small-en"], chunk_tokens=8, overlap_tokens=2)
    with pytest.raises(ValueError, match="search_text_limit"):
        make_chunks([passage("é" * 20)], profile, len, lambda text: text, max_text_bytes=30)
    with pytest.raises(ValueError, match="search_chunk_limit"):
        make_chunks([passage("a" * 50)], profile, len, lambda text: text, max_chunks=2)


def test_model_input_validation_is_applied_before_storing() -> None:
    profile = replace(profiles()["bge-small-en"], chunk_tokens=40, overlap_tokens=2)

    def validate(text: str) -> str:
        if len(text) > 10:
            raise ValueError("special tokens and prefix exhaust the input budget")
        return text

    chunks = make_chunks([passage("0123456789" * 5)], profile, len, validate)
    assert all(len(chunk.text) <= 10 for chunk in chunks)


def test_opt_in_preserves_fitting_passage_and_default_chunk_identity() -> None:
    profile = replace(profiles()["bge-small-en"], chunk_tokens=10, overlap_tokens=2, separator="")
    a, b = passage("abcde"), passage("IJKLMNO", 1)
    default = make_chunks([a, b], profile, len, lambda text: text)
    alternate = make_chunks([a, b], profile, len, lambda text: text, preserve_fitting_passages=True)
    assert [chunk.text for chunk in default] == ["abcdeIJKLM", "LMNO"]
    assert [chunk.text for chunk in alternate] == ["abcde", "deIJKLMNO"]
    assert any(
        part.passage.id == b.id and part.start == 0 and part.end == len(b.text)
        for chunk in alternate
        for part in chunk.mappings
    )
    assert all(chunk.text == "".join(part.text for part in chunk.mappings) for chunk in alternate)


def test_passage_preservation_respects_chunk_cap_when_it_adds_chunks() -> None:
    profile = replace(profiles()["bge-small-en"], chunk_tokens=10, overlap_tokens=0, separator="")
    saved = [passage("a" * 6), passage("b" * 6, 1), passage("c" * 6, 2)]
    assert len(make_chunks(saved, profile, len, lambda text: text, max_chunks=2)) == 2
    with pytest.raises(ValueError, match="search_chunk_limit"):
        make_chunks(
            saved, profile, len, lambda text: text, max_chunks=2, preserve_fitting_passages=True
        )
