from core.extract.reading_order import GlyphText, logical_line_text


def glyphs(texts, *, descending=False):
    return [GlyphText(text, -i if descending else i) for i, text in enumerate(texts)]


def test_visual_rtl_keeps_a_ligatures_internal_unicode_order():
    # The font's one glyph maps to logical "لم", not two independently
    # reversible characters. Reversing the whole string would corrupt it.
    units = glyphs(["ة", "ا", "و", "ا", "س", "لم", "ا"])
    assert logical_line_text(units) == "المساواة"


def test_already_logical_rtl_is_not_reversed_twice():
    units = glyphs(["ا", "لم", "س", "ا", "و", "ا", "ة"], descending=True)
    assert logical_line_text(units) == "المساواة"


def test_numbers_and_latin_punctuation_keep_their_internal_order():
    units = glyphs(list("(RFC 9110) مقر ريزولا لاق"))
    assert logical_line_text(units) == "قال الوزير رقم (RFC 9110)"


def test_separate_arabic_marks_follow_their_base_in_logical_output():
    assert logical_line_text(glyphs(list("ُعيمج"))) == "جميعُ"


def test_ltr_and_ambiguous_positions_remain_faithful():
    assert logical_line_text(glyphs(list("Rainfall 13.5%"))) == "Rainfall 13.5%"
    units = [GlyphText(c, 0) for c in "المساواة"]
    assert logical_line_text(units) == "المساواة"
