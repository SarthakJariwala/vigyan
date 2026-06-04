from __future__ import annotations

import re

import httpx
from lxml import etree  # type: ignore[import-untyped]

from ..interfaces import DocumentParser
from ..models import Document, Paragraph


class GrobidParser(DocumentParser):
    """GROBID-based PDF fulltext parser returning paragraphs.

    It requests paragraph and sentence coordinates for precise citations.
    """

    def __init__(self, server_url: str = "http://localhost:8070") -> None:
        self.server_url = server_url.rstrip("/")

    def parse(self, pdf_bytes: bytes) -> tuple[list[Paragraph], str | None]:
        tei_xml = self._grobid_fulltext_xml(pdf_bytes)
        paragraphs = self._parse_tei_to_paragraphs(tei_xml)
        return paragraphs, tei_xml

    def extract_metadata(self, pdf_bytes: bytes) -> Document:
        """Extract high-level metadata from PDF using GROBID header service.

        Falls back to parsing metadata out of the fulltext TEI if header is unavailable.
        """
        try:
            tei_header = self._grobid_header_xml(pdf_bytes)
            return self._meta_from_tei(tei_header)
        except Exception:
            # Fallback: use fulltext and parse header from it
            tei_full = self._grobid_fulltext_xml(pdf_bytes)
            return self._meta_from_tei(tei_full)

    def _grobid_fulltext_xml(self, pdf_bytes: bytes) -> str:
        # Use list-of-tuples for files to avoid any ambiguity
        files = [("input", ("doc.pdf", pdf_bytes, "application/pdf"))]
        # GROBID expects teiCoordinates as repeated/list form fields. A single
        # comma-separated value is treated as an unknown coordinate target and
        # paragraph coords are omitted, forcing page-number fallback to page 1.
        data = {
            "teiCoordinates": ["p", "s", "head", "ref", "figure"],
            "segmentSentences": "1",
        }
        r = httpx.post(
            f"{self.server_url}/api/processFulltextDocument",
            files=files,
            data=data,
            timeout=120,
        )
        r.raise_for_status()
        return r.text

    def _grobid_header_xml(self, pdf_bytes: bytes) -> str:
        files = [("input", ("doc.pdf", pdf_bytes, "application/pdf"))]
        r = httpx.post(
            f"{self.server_url}/api/processHeaderDocument",
            files=files,
            timeout=60,
        )
        r.raise_for_status()
        return r.text

    @staticmethod
    def _parse_tei_to_paragraphs(tei_xml: str) -> list[Paragraph]:
        """Parse GROBID TEI text blocks into searchable Paragraph records."""
        root = etree.fromstring(tei_xml.encode("utf-8"))
        ns = {"tei": "http://www.tei-c.org/ns/1.0"}

        out: list[Paragraph] = []
        block_nodes = root.xpath(
            "//tei:text//tei:p[not(ancestor::tei:figure)]"
            " | //tei:text//tei:figure[@type='table']",
            namespaces=ns,
        )
        for node in block_nodes:
            if etree.QName(node).localname == "figure":
                out.append(GrobidParser._parse_tei_table(node, ns))
                continue

            text = GrobidParser._node_text(node)
            if not text:
                continue
            xml_id = node.get("{http://www.w3.org/XML/1998/namespace}id")
            coords = GrobidParser._coords_for_node(node)
            page_start, page_end = GrobidParser._page_span_from_coords(coords)

            out.append(
                Paragraph(
                    text=text,
                    page_start=page_start,
                    page_end=page_end,
                    para_id=xml_id,
                    coords=coords,
                    section_path=GrobidParser._section_path_for_node(node, ns),
                )
            )
        return out

    @staticmethod
    def _parse_tei_table(figure: etree._Element, ns: dict[str, str]) -> Paragraph:
        xml_id = figure.get("{http://www.w3.org/XML/1998/namespace}id")
        head_nodes = figure.xpath("./tei:head[1]", namespaces=ns)
        desc_nodes = figure.xpath("./tei:figDesc[1]", namespaces=ns)
        head = GrobidParser._node_text(head_nodes[0]) if head_nodes else None
        desc = GrobidParser._node_text(desc_nodes[0]) if desc_nodes else None
        caption = GrobidParser._combine_caption(head, desc)

        cells: list[list[str]] = []
        row_nodes = figure.xpath(".//tei:table/tei:row", namespaces=ns)
        for row in row_nodes:
            row_cells: list[str] = []
            for cell in row.xpath("./tei:cell", namespaces=ns):
                cell_text = GrobidParser._node_text(cell)
                try:
                    colspan = max(1, int(cell.get("cols") or "1"))
                except ValueError:
                    colspan = 1
                row_cells.extend([cell_text] * colspan)
            if row_cells:
                cells.append(row_cells)

        coords = GrobidParser._coords_for_node(figure)
        page_start, page_end = GrobidParser._page_span_from_coords(coords)
        markdown_table = GrobidParser._cells_to_markdown(cells)
        text_parts = ["[TABLE]"]
        if caption:
            text_parts.append(f"Caption: {caption}")
        if markdown_table:
            text_parts.append(markdown_table)
        text = "\n".join(text_parts)

        return Paragraph(
            text=text,
            page_start=page_start,
            page_end=page_end,
            para_id=xml_id,
            coords=coords,
            section_path=GrobidParser._section_path_for_node(figure, ns),
            block_type="table",
            caption=caption,
            cells=cells or None,
        )

    @staticmethod
    def _node_text(node: etree._Element) -> str:
        raw = "".join(node.itertext())
        return re.sub(r"\s+", " ", raw).strip()

    @staticmethod
    def _coords_for_node(node: etree._Element) -> str | None:
        own_coords = node.get("coords")
        if own_coords:
            return own_coords
        descendant_coords = [
            str(coords).strip()
            for coords in node.xpath(".//*[@coords]/@coords")
            if str(coords).strip()
        ]
        return ";".join(descendant_coords) if descendant_coords else None

    @staticmethod
    def _page_span_from_coords(coords: str | None) -> tuple[int, int]:
        if not coords:
            return (1, 1)
        pages: set[int] = set()
        for box in coords.split(";"):
            if not box:
                continue
            page = box.split(",", 1)[0].strip()
            if page.isdigit():
                pages.add(int(page))
        if not pages:
            return (1, 1)
        pages_list = sorted(pages)
        return (pages_list[0], pages_list[-1])

    @staticmethod
    def _section_path_for_node(node: etree._Element, ns: dict[str, str]) -> list[str]:
        section_path: list[str] = []
        for div in node.xpath("ancestor::tei:div", namespaces=ns):
            head_nodes = div.xpath("./tei:head[1]", namespaces=ns)
            if not head_nodes:
                continue
            heading = GrobidParser._node_text(head_nodes[0])
            if heading:
                section_path.append(heading)
        return section_path

    @staticmethod
    def _combine_caption(head: str | None, desc: str | None) -> str | None:
        parts = [part for part in [head, desc] if part]
        return " ".join(parts) if parts else None

    @staticmethod
    def _cells_to_markdown(cells: list[list[str]]) -> str:
        if not cells:
            return ""

        max_rows = 20
        max_cols = 10
        max_chars = 4000
        truncated = len(cells) > max_rows or any(len(row) > max_cols for row in cells)
        display_rows = [row[:max_cols] for row in cells[:max_rows]]
        width = max((len(row) for row in display_rows), default=0)
        if width == 0:
            return ""

        def normalize_row(row: list[str]) -> list[str]:
            padded = row + [""] * (width - len(row))
            return [cell.replace("|", r"\|").replace("\n", "<br>") for cell in padded]

        normalized = [normalize_row(row) for row in display_rows]
        header = normalized[0]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * width) + " |",
        ]
        for row in normalized[1:]:
            lines.append("| " + " | ".join(row) + " |")

        markdown = "\n".join(lines)
        if len(markdown) > max_chars:
            markdown = markdown[:max_chars].rstrip()
            truncated = True
        if truncated:
            markdown += "\n... (truncated)"
        return markdown

    @staticmethod
    def _meta_from_tei(tei_xml: str) -> Document:
        root = etree.fromstring(tei_xml.encode("utf-8"))
        ns = {"tei": "http://www.tei-c.org/ns/1.0"}

        # Title
        title_nodes = root.xpath(
            "//tei:teiHeader//tei:titleStmt/tei:title[1]", namespaces=ns
        )
        title = (title_nodes[0].text or "").strip() if title_nodes else "Untitled"

        # Authors (join forename/surname if present)
        author_nodes = root.xpath(
            "//tei:teiHeader//tei:author/tei:persName", namespaces=ns
        )
        authors: list[str] = []
        for pn in author_nodes:
            forename = " ".join(
                [t.strip() for t in pn.xpath("tei:forename/text()", namespaces=ns)]
            )
            surname_nodes = pn.xpath("tei:surname/text()", namespaces=ns)
            surname = surname_nodes[0].strip() if surname_nodes else ""
            name = (forename + " " + surname).strip()
            if not name:
                # Fallback: any text under persName
                name = " ".join([t.strip() for t in pn.itertext()]).strip()
            if name:
                authors.append(name)

        # DOI / arXiv / URL
        def _first_text(xpath: str) -> str | None:
            nodes = root.xpath(xpath, namespaces=ns)
            return nodes[0].strip() if nodes else None

        doi = _first_text("//tei:teiHeader//tei:idno[@type='DOI']/text()")
        arxiv_id = _first_text("//tei:teiHeader//tei:idno[@type='arXiv']/text()")
        url = _first_text("//tei:teiHeader//tei:idno[@type='URL']/text()")

        # Venue and year
        venue = None
        venue_nodes = root.xpath(
            "//tei:teiHeader//tei:monogr/tei:title[@level='j' or @level='m']/text()",
            namespaces=ns,
        )
        if venue_nodes:
            venue = venue_nodes[0].strip() or None

        year = None
        date_when = _first_text(
            "//tei:teiHeader//tei:monogr/tei:imprint/tei:date/@when"
        )
        if date_when and len(date_when) >= 4 and date_when[:4].isdigit():
            year = int(date_when[:4])

        # Build doc_id deterministically
        def _slugify(s: str) -> str:
            import re

            s = s.lower()
            s = re.sub(r"[^a-z0-9\s-]", "", s)
            s = re.sub(r"\s+", "-", s)
            s = re.sub(r"-+", "-", s)
            return s.strip("-")

        if doi:
            base_id = "doi-" + _slugify(doi.replace("/", "-"))
        elif arxiv_id:
            base_id = "arxiv-" + _slugify(arxiv_id)
        else:
            first_author_last = _slugify(
                (authors[0].split(" ")[-1] if authors else "doc")
            )
            year_part = str(year) if year else "nd"
            title_words = _slugify(title).split("-")[:6]
            base_id = f"{first_author_last}{year_part}-" + (
                "-".join(title_words) or "untitled"
            )

        return Document(
            doc_id=base_id,
            title=title,
            authors=authors or [],
            venue=venue,
            year=year,
            doi=doi,
            arxiv_id=arxiv_id,
            url=url,
        )
