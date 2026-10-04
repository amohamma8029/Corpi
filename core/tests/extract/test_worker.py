"""Worker entrypoint behaviour (M1.5): the page limit is checked inside the child, pre-parse."""

import json

from pdf_fixtures import PageSpec, build_pdf, native_text_stream

from core.extract import worker


def test_worker_rejects_over_limit_before_parsing(tmp_path, capsys) -> None:
    over_limit = build_pdf(
        [PageSpec(native_text_stream(f"page {index}"), helvetica=True) for index in range(201)]
    )
    path = tmp_path / "over.pdf"
    path.write_bytes(over_limit)

    code = worker.main([str(path)])

    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert code == 1
    assert lines[0]["type"] == "error"
    assert lines[0]["code"] == "page_limit_exceeded"
    assert all(line["type"] != "page" for line in lines), "pages must not parse past the limit"


def test_worker_parses_within_the_limit(tmp_path, capsys) -> None:
    small = build_pdf([PageSpec(native_text_stream("Hello there"), helvetica=True)])
    path = tmp_path / "small.pdf"
    path.write_bytes(small)

    code = worker.main([str(path)])

    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert code == 0
    assert [line["type"] for line in lines] == ["page", "done"]
    assert lines[0]["metrics"]["extracted_chars"] == len("Hellothere")
