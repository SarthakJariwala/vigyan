# Table Extraction from GROBID — Implementation Plan

## Overview

Add support for extracting and indexing tables from GROBID's TEI XML output. Tables become first-class searchable content alongside paragraphs, stored in the same vector index.

**Design principle:** Extend existing models with optional fields rather than introducing new types. No interface changes needed — `DocumentParser.parse()` still returns `list[Paragraph]`.

---

## Step 1 — Extend `Paragraph` model

**File:** `src/vigyan/models.py`

Add optional table fields so `Paragraph` doubles as a generic "block":

```python
class Paragraph(BaseModel):
    text: str
    page_start: int
    page_end: int
    para_id: str | None = None
    coords: str | None = None
    # New fields for table support
    block_type: Literal["paragraph", "table"] = "paragraph"
    caption: str | None = None
    cells: list[list[str]] | None = None  # row-major grid of cell text
```

All new fields have defaults, so existing code is unaffected.

---

## Step 2 — Parse tables in `GrobidParser`

**File:** `src/vigyan/parsers/grobid.py`

### a) Request table coordinates from GROBID

In `_grobid_fulltext_xml`, add `"figure"` to the `teiCoordinates` value so table bounding boxes are returned.

### b) Add `_cells_to_markdown()` helper

Renders a `list[list[str]]` cell grid as a markdown pipe-table. Truncates to ~4k chars, capping at ~20 rows and ~10 columns, appending `... (truncated)` when needed.

### c) Add `_parse_tei_tables()` static method

XPath target: `//tei:text//tei:figure[@type='table']`

For each `<figure type="table">` element:

1. Extract `xml:id` from the `<figure>` → `para_id` (e.g. `tab_1`)
2. Extract caption from `<figDesc>` child → `caption`
3. Extract `<head>` text (e.g. "Table 1 …") — prepend to caption if present
4. Walk `<table>/<row>/<cell>` to build `cells: list[list[str]]`
   - Handle `cols` attribute for colspan by repeating cell text
   - Normalize `<lb/>` line breaks within cell text
5. Extract `coords` from `<figure>` or `<table>` → derive `page_start` / `page_end`
6. Build `text` field as: `[TABLE]\nCaption: {caption}\n{markdown_table}`
7. Return a `Paragraph(block_type="table", ...)`

### d) Update `_parse_tei_to_paragraphs`

Rename to `_parse_tei_to_blocks`. Call both paragraph extraction and `_parse_tei_tables()`, concatenate results (order doesn't affect retrieval).

### Reference: GROBID TEI table structure

```xml
<figure type="table" xml:id="tab_1" coords="...">
  <head>Table <label>1</label> Title text</head>
  <figDesc>Caption text describing the table.</figDesc>
  <table>
    <row>
      <cell cols="2">Merged header</cell>
      <cell>Header 3</cell>
    </row>
    <row>
      <cell>Value A</cell>
      <cell>Value B</cell>
      <cell>Value C</cell>
    </row>
  </table>
  <note>Footnote text</note>
</figure>
```

---

## Step 3 — Extend `Chunk` model

**File:** `src/vigyan/models.py`

```python
class Chunk(BaseModel):
    ...
    chunk_type: str = "paragraph"
    caption: str | None = None
```

---

## Step 4 — Update `ChunkRecord` schema

**File:** `src/vigyan/vectordb/lancedb_store.py`

Add `chunk_type: str = "paragraph"` and `caption: str | None = None` to the `ChunkRecord` class inside `make_chunk_record_model`.

> **Migration note:** Adding columns to an existing LanceDB table may require recreating it. For local-only usage, delete the DB and re-ingest. Add a comment noting this.

---

## Step 5 — Update Corpus ingestion

**File:** `src/vigyan/corpus/ingestion.py`

In `CorpusIngestor.ingest_pdf`, propagate the new fields when building chunks:

```python
for p in paragraphs:
    chunks.append(Chunk(
        ...
        chunk_type=p.block_type,
        caption=p.caption,
        ...
    ))
```

No structural changes — the loop already iterates over all paragraphs (which now includes tables).

---

## Step 6 — Extend `QueryHit`

**File:** `src/vigyan/models.py`

Add `chunk_type: str = "paragraph"` so callers can distinguish table hits.

**File:** `src/vigyan/vectordb/lancedb_store.py`

Update `search()` to select `chunk_type` and pass it through `_format_hit` into `QueryHit`.

---

## Step 7 — Update exports

**File:** `src/vigyan/__init__.py`

No changes needed — `Paragraph`, `Chunk`, `QueryHit` are already exported. New fields are additive.

---

## File change summary

| File | Change |
|---|---|
| `src/vigyan/models.py` | Add fields to `Paragraph`, `Chunk`, `QueryHit` |
| `src/vigyan/parsers/grobid.py` | Add `_parse_tei_tables()`, `_cells_to_markdown()`, update parse method |
| `src/vigyan/corpus/ingestion.py` | Pass `block_type` → `chunk_type` and `caption` when building chunks |
| `src/vigyan/vectordb/lancedb_store.py` | Add `chunk_type`, `caption` to `ChunkRecord`; select `chunk_type` in `search()` |

No interface changes. No new files. All changes are additive with backward-compatible defaults.
