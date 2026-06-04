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
    parsed paragraphs fall back to page 1. Include figure coordinates so table
    page spans can be extracted too.
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
        if requested_coordinates == ["p", "s", "head", "ref", "figure"]:
            return FakeResponse(xml_with_paragraph_coords)
        return FakeResponse(xml_without_paragraph_coords)

    monkeypatch.setattr("vigyan.parsers.grobid.httpx.post", fake_post)

    parser = GrobidParser(server_url="http://grobid.test")
    paragraphs, _ = parser.parse(b"%PDF fixture")

    assert captured["data"]["teiCoordinates"] == ["p", "s", "head", "ref", "figure"]
    assert paragraphs[0].page_start == 2
    assert paragraphs[0].page_end == 3
    assert paragraphs[0].coords == "2,10,10,50,50;3,10,10,50,50"


def test_parse_derives_paragraph_pages_from_child_coordinates_and_section_path() -> None:
    tei_xml = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text><body>
        <div>
          <head>Results</head>
          <div>
            <head>Device performance</head>
            <p xml:id="p1">
              <s coords="4,10,10,50,50">The first sentence lacks parent coords.</s>
              <s coords="5,10,10,50,50">The second sentence continues.</s>
            </p>
          </div>
        </div>
      </body></text>
    </TEI>
    """

    paragraphs = GrobidParser._parse_tei_to_paragraphs(tei_xml)

    assert len(paragraphs) == 1
    paragraph = paragraphs[0]
    assert paragraph.text == (
        "The first sentence lacks parent coords. The second sentence continues."
    )
    assert paragraph.page_start == 4
    assert paragraph.page_end == 5
    assert paragraph.coords == "4,10,10,50,50;5,10,10,50,50"
    assert paragraph.section_path == ["Results", "Device performance"]


def test_parse_extracts_tables_as_searchable_blocks_with_caption_cells_and_section() -> None:
    tei_xml = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text><body>
        <div>
          <head>Performance metrics</head>
          <figure type="table" xml:id="tab_1" coords="6,20,20,200,100">
            <head>Table <label>1</label></head>
            <figDesc>Champion device metrics.</figDesc>
            <table>
              <row>
                <cell>Metric</cell>
                <cell>Value</cell>
              </row>
              <row>
                <cell>PCE</cell>
                <cell>25.1%</cell>
              </row>
              <row>
                <cell cols="2">Measured under AM1.5G</cell>
              </row>
            </table>
          </figure>
        </div>
      </body></text>
    </TEI>
    """

    blocks = GrobidParser._parse_tei_to_paragraphs(tei_xml)

    assert len(blocks) == 1
    table = blocks[0]
    assert table.block_type == "table"
    assert table.para_id == "tab_1"
    assert table.page_start == 6
    assert table.page_end == 6
    assert table.coords == "6,20,20,200,100"
    assert table.section_path == ["Performance metrics"]
    assert table.caption == "Table 1 Champion device metrics."
    assert table.cells == [
        ["Metric", "Value"],
        ["PCE", "25.1%"],
        ["Measured under AM1.5G", "Measured under AM1.5G"],
    ]
    assert table.text.startswith("[TABLE]\nCaption: Table 1 Champion device metrics.")
    assert "| Metric | Value |" in table.text
    assert "| PCE | 25.1% |" in table.text
