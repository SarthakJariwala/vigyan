from __future__ import annotations

import httpx
from lxml import etree  # type: ignore[import-untyped]

from ..core.interfaces import DocumentParser
from ..core.models import Document, Paragraph


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
        # Use a single, comma-separated teiCoordinates value
        data = {
            "teiCoordinates": "p,s,head,ref",
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
        root = etree.fromstring(tei_xml.encode("utf-8"))
        ns = {"tei": "http://www.tei-c.org/ns/1.0"}

        para_nodes = root.xpath("//tei:text//tei:p", namespaces=ns)
        out: list[Paragraph] = []
        for p in para_nodes:
            text = "".join(p.itertext()).strip()
            if not text:
                continue
            xml_id = p.get("{http://www.w3.org/XML/1998/namespace}id")
            coords = p.get("coords")
            if coords:
                pages_list = sorted(
                    {int(bb.split(",")[0]) for bb in coords.split(";") if bb}
                )
                page_start = pages_list[0]
                page_end = pages_list[-1]
            else:
                page_start = page_end = 1

            out.append(
                Paragraph(
                    text=text,
                    page_start=page_start,
                    page_end=page_end,
                    para_id=xml_id,
                    coords=coords,
                )
            )
        return out

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
