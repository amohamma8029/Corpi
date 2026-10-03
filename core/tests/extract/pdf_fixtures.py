"""Deterministic PDF fixtures for M1.5 extraction tests.

Raw PDF assembly (no generator dependency), byte-stable for identical inputs — matching the
1.4 fixture discipline. Object numbering is reserved up front: 1 catalog, 2 pages tree.
"""

from __future__ import annotations

PAGE_WIDTH = 612.0
PAGE_HEIGHT = 792.0

IMAGE_RGB_2X2 = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])


def native_text_stream(text: str, *, x: float = 72, y: float = 720, size: int = 24) -> bytes:
    return f"BT /F1 {size} Tf {x} {y} Td ({text}) Tj ET".encode("ascii")


def page_number_stream(number: int, *, x: float = 300, y: float = 40) -> bytes:
    return f"BT /F1 12 Tf {x} {y} Td ({number}) Tj ET".encode("ascii")


def undecodable_stream(*, x: float = 72, y: float = 720) -> bytes:
    """CIDs through Identity-H with no ToUnicode map: the parser emits (cid:N) tokens."""
    return f"BT /F2 24 Tf {x} {y} Td <0001000200030004> Tj ET".encode("ascii")


def image_draw_stream(
    *, x: float = 50, y: float = 100, width: float = 500, height: float = 600
) -> bytes:
    return f"q {width} 0 0 {height} {x} {y} cm /Im1 Do Q".encode("ascii")


def build_pdf(pages: list[tuple[bytes, bool, bool, bool]]) -> bytes:
    """Assemble pages of (content_stream, needs_helvetica, needs_cid_font, needs_image)."""
    objects: dict[int, bytes] = {}
    next_id = 3

    def add(body: bytes) -> int:
        nonlocal next_id
        objects[next_id] = body
        next_id += 1
        return next_id - 1

    helvetica_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    descriptor_id = add(
        b"<< /Type /FontDescriptor /FontName /TestCID /Flags 4 "
        b"/FontBBox [0 -200 1000 800] /ItalicAngle 0 /Ascent 800 /Descent -200 "
        b"/CapHeight 700 /StemV 80 >>"
    )
    cid_font_id = add(
        b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /TestCID "
        b"/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> "
        b"/FontDescriptor " + f"{descriptor_id} 0 R".encode() + b" /DW 500 >>"
    )
    type0_id = add(
        b"<< /Type /Font /Subtype /Type0 /BaseFont /TestCID-Identity "
        b"/Encoding /Identity-H /DescendantFonts [" + f"{cid_font_id} 0 R".encode() + b"] >>"
    )
    image_id = add(
        b"<< /Type /XObject /Subtype /Image /Width 2 /Height 2 "
        b"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Length 12 >>\nstream\n"
        + IMAGE_RGB_2X2
        + b"\nendstream"
    )

    page_ids: list[int] = []
    for content, needs_helvetica, needs_cid, needs_image in pages:
        fonts = b""
        if needs_helvetica:
            fonts += b" /F1 " + f"{helvetica_id} 0 R".encode()
        if needs_cid:
            fonts += b" /F2 " + f"{type0_id} 0 R".encode()
        resources = b"<< /Font <<" + fonts + b" >>"
        if needs_image:
            resources += b" /XObject << /Im1 " + f"{image_id} 0 R".encode() + b" >>"
        resources += b" >>"
        stream_body = (
            b"<< /Length "
            + str(len(content)).encode()
            + b" >>\nstream\n"
            + content
            + b"\nendstream"
        )
        content_id = add(stream_body)
        page_ids.append(
            add(
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 "
                + f"{PAGE_WIDTH:g} {PAGE_HEIGHT:g}".encode()
                + b"] /Contents "
                + f"{content_id} 0 R".encode()
                + b" /Resources "
                + resources
                + b" >>"
            )
        )

    kids = b"[" + b" ".join(f"{obj_id} 0 R".encode() for obj_id in page_ids) + b"]"
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = (
        b"<< /Type /Pages /Kids " + kids + b" /Count " + str(len(page_ids)).encode() + b" >>"
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for object_id in sorted(objects):
        offsets[object_id] = len(out)
        out += f"{object_id} 0 obj\n".encode() + objects[object_id] + b"\nendobj\n"
    size = max(objects) + 1
    xref_at = len(out)
    out += f"xref\n0 {size}\n".encode()
    out += b"0000000000 65535 f \n"
    for object_id in range(1, size):
        out += f"{offsets[object_id]:010d} 00000 n \n".encode()
    out += b"trailer\n<< /Root 1 0 R /Size " + str(size).encode() + b" >>\n"
    out += b"startxref\n" + str(xref_at).encode() + b"\n%%EOF\n"
    return bytes(out)


def native_text_pdf(*texts: str) -> bytes:
    return build_pdf([(native_text_stream(text), True, False, False) for text in texts])


def undecodable_text_pdf() -> bytes:
    return build_pdf([(undecodable_stream(), False, True, False)])


def image_only_pdf() -> bytes:
    return build_pdf([(image_draw_stream(), False, False, True)])


def scanned_page_number_pdf() -> bytes:
    content = image_draw_stream() + b"\n" + page_number_stream(3)
    return build_pdf([(content, True, False, True)])


def blank_pdf() -> bytes:
    return build_pdf([(b"", False, False, False)])
