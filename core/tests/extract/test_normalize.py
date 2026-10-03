"""Faithfully normalized text v1: meaning preserved, never rewritten."""

from core.extract.normalize import normalize_cell, normalize_table_row, normalize_text


def test_whitespace_collapses_and_lines_join_with_spaces() -> None:
    assert normalize_text("Hello   world") == "Hello world"
    assert normalize_text("Hello\nworld") == "Hello world"
    assert normalize_text("  padded  \n  line  ") == "padded line"


def test_ambiguous_line_break_hyphens_stay_hyphenated() -> None:
    assert normalize_text("re-\ncreation") == "re-creation"
    assert normalize_text("un-\nfortunate") == "un-fortunate"
    assert normalize_text("mid-\nterm") == "mid-term"


def test_spaced_dash_continues_with_a_space() -> None:
    assert normalize_text("foo -\nbar") == "foo - bar"


def test_original_list_numbering_is_never_rewritten() -> None:
    assert normalize_text("2. Item text") == "2. Item text"
    assert normalize_text("3) Third item") == "3) Third item"
    assert normalize_text("1. First\n2. Second") == "1. First 2. Second"


def test_nfc_normalization_applies() -> None:
    assert normalize_text("e\u0301clair") == "\u00e9clair"


def test_empty_and_whitespace_blocks_normalize_to_empty() -> None:
    assert normalize_text("") == ""
    assert normalize_text("  \n \t \n") == ""


def test_blank_lines_inside_a_block_do_not_break_joins() -> None:
    assert normalize_text("First\n\nSecond") == "First Second"


def test_table_cells_normalize_before_tab_separators_survive() -> None:
    assert normalize_table_row(["a  b", " c ", ""]) == "a b\tc\t"
    assert "\t" in normalize_table_row(["left", "right"])
    assert normalize_cell(" x \t y ") == "x y"
    assert "\n" not in normalize_cell("multi\nline")
