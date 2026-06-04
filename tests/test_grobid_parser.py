from __future__ import annotations

from typing import Any

from vigyan.parsers import GrobidParser


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        pass


def test_parse_requests_repeated_tei_coordinates_so_grobid_returns_page_numbers(
    monkeypatch,
) -> None:
    """GROBID treats teiCoordinates as a repeated/list parameter.

    If we send one comma-separated value, paragraph coords are omitted and all
    parsed paragraphs fall back to page 1.
    """
    captured: dict[str, Any] = {}

    xml_with_paragraph_coords = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text><body>
        <p xml:id="p1" coords="2,10,10,50,50;3,10,10,50,50">Cross-page paragraph.</p>
      </body></text>
    </TEI>
    """
    xml_without_paragraph_coords = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text><body>
        <p xml:id="p1">Cross-page paragraph.</p>
      </body></text>
    </TEI>
    """

    def fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        captured.update(kwargs)
        requested_coordinates = kwargs["data"]["teiCoordinates"]
        if requested_coordinates == ["p", "s", "head", "ref"]:
            return FakeResponse(xml_with_paragraph_coords)
        return FakeResponse(xml_without_paragraph_coords)

    monkeypatch.setattr("vigyan.parsers.grobid.httpx.post", fake_post)

    parser = GrobidParser(server_url="http://grobid.test")
    paragraphs, _ = parser.parse(b"%PDF fixture")

    assert captured["data"]["teiCoordinates"] == ["p", "s", "head", "ref"]
    assert paragraphs[0].page_start == 2
    assert paragraphs[0].page_end == 3
    assert paragraphs[0].coords == "2,10,10,50,50;3,10,10,50,50"
