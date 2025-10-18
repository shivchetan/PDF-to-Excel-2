
import io
import os
import re
import hashlib
import tempfile
from datetime import datetime
from typing import List, Tuple, Optional

import pandas as pd
import streamlit as st

# Optional dependencies (guarded imports)
_CAMEL0T_AVAILABLE = False
_TABULA_AVAILABLE = False
_PDFPLUMBER_AVAILABLE = False

try:
    import camelot  # requires ghostscript; best for vector PDFs with visible lines
    _CAMEL0T_AVAILABLE = True
except Exception:
    pass

try:
    import tabula  # requires Java; robust table extraction
    _TABULA_AVAILABLE = True
except Exception:
    pass

try:
    import pdfplumber  # pure-Python; can extract tables & text heuristically
    _PDFPLUMBER_AVAILABLE = True
except Exception:
    pass


# -------------------------------
# Helpers
# -------------------------------
def _file_sha1(data: bytes) -> str:
    sha1 = hashlib.sha1()
    sha1.update(data)
    return sha1.hexdigest()


def _normalise_sheet_name(name: str) -> str:
    """Excel sheet names must be <=31 chars and avoid: : \ / ? * [ ]"""
    name = re.sub(r'[:\\/\?\*\[\]]', '_', name)
    return name[:31] if len(name) > 31 else name


def _parse_pages_spec(spec: str) -> str:
    """
    Normalise a pages spec for Camelot/Tabula (e.g., 'all', '1', '1,3-5').
    We keep it permissive; fallback to 'all' on invalid input.
    """
    if not spec or spec.strip().lower() in ("all", "*"):
        return "all"
    cleaned = spec.replace(" ", "")
    if re.fullmatch(r"(\d+(-\d+)?)(,(\d+(-\d+)?))*", cleaned or ""):
        return cleaned
    return "all"


def _write_preview(df_list: List[pd.DataFrame], limit: int = 3) -> None:
    """Show a quick preview of the first few tables."""
    preview_count = min(limit, len(df_list))
    if preview_count == 0:
        return
    st.subheader("Quick preview")
    for i in range(preview_count):
        st.caption(f"Table {i+1} — first 10 rows")
        st.dataframe(df_list[i].head(10), use_container_width=True)


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning: drop fully-empty columns/rows, strip whitespace."""
    # Strip whitespace from string cells
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    # Drop fully empty columns and rows
    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='all')
    # Reset index for tidy export
    return df.reset_index(drop=True)


def _ensure_tables_exist(tables: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Ensure we return a list; drop empty frames."""
    cleaned = []
    for t in tables or []:
        if isinstance(t, pd.DataFrame) and not t.empty:
            cleaned.append(_clean_dataframe(t))
    return cleaned


# -------------------------------
# Extraction strategies
# -------------------------------
def extract_with_camelot(pdf_path: str, pages: str) -> List[pd.DataFrame]:
    """
    Try Camelot with both 'lattice' and 'stream' flavours.
    Use the flavour that yields more tables (or merge if both yield something).
    """
    if not _CAMEL0T_AVAILABLE:
        return []
    try:
        tables_all: List[pd.DataFrame] = []
        # Attempt lattice (works well when there are ruling lines)
        try:
            t_lattice = camelot.read_pdf(pdf_path, pages=pages, flavour="lattice")
            tables_all.extend([t.df for t in t_lattice] if t_lattice else [])
        except Exception:
            pass

        # Attempt stream (works better for whitespace-separated columns)
        try:
            t_stream = camelot.read_pdf(pdf_path, pages=pages, flavour="stream")
            tables_all.extend([t.df for t in t_stream] if t_stream else [])
        except Exception:
            pass

        return _ensure_tables_exist(tables_all)
    except Exception:
        return []


def extract_with_tabula(pdf_path: str, pages: str) -> List[pd.DataFrame]:
    """Use tabula to read multiple tables."""
    if not _TABULA_AVAILABLE:
        return []
    try:
        dfs = tabula.read_pdf(pdf_path, pages=pages, multiple_tables=True)
        return _ensure_tables_exist(dfs)
    except Exception:
        return []


def extract_with_pdfplumber(pdf_bytes: bytes, pages: str) -> Tuple[List[pd.DataFrame], List[Tuple[int, str]]]:
    """
    pdfplumber heuristic extraction for tables & raw text.
    Returns (tables, [(page_number, text), ...])
    """
    if not _PDFPLUMBER_AVAILABLE:
        return [], []

    table_dfs: List[pd.DataFrame] = []
    text_pages: List[Tuple[int, str]] = []

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            total_pages = len(pdf.pages)
            if pages == "all":
                page_indices = range(total_pages)
            else:
                # Convert "1,3-5" to 0-based indices
                idx_set = set()
                for part in pages.split(","):
                    if "-" in part:
                        a, b = part.split("-")
                        a, b = int(a), int(b)
                        for p in range(a, b + 1):
                            idx_set.add(p - 1)
                    else:
                        idx_set.add(int(part) - 1)
                page_indices = sorted([i for i in idx_set if 0 <= i < total_pages])

            for i in page_indices:
                page = pdf.pages[i]
                # Try line-based extraction first
                settings_line = {
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "intersection_y_tolerance": 5,
                    "intersection_x_tolerance": 5,
                }
                tab_candidates = page.extract_tables(table_settings=settings_line) or []

                # Fallback to text-based extraction
                if len(tab_candidates) == 0:
                    settings_text = {
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "snap_tolerance": 3,
                        "join_tolerance": 3,
                    }
                    tab_candidates = page.extract_tables(table_settings=settings_text) or []

                for tbl in tab_candidates:
                    try:
                        df = pd.DataFrame(tbl)
                        table_dfs.append(df)
                    except Exception:
                        continue

                # Always capture raw text per page (helpful fallback)
                try:
                    text_pages.append((i + 1, page.extract_text() or ""))
                except Exception:
                    text_pages.append((i + 1, ""))

    except Exception:
        return [], []

    return _ensure_tables_exist(table_dfs), text_pages


def extract_tables(pdf_bytes: bytes, pages: str, method: str) -> Tuple[List[pd.DataFrame], List[Tuple[int, str]], str]:
    """
    Main dispatcher. Returns (tables, raw_text_by_page, method_used).
    method ∈ {'auto','camelot','tabula','pdfplumber'}
    """
    # Write to temp file for tools that need file paths
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        if method == "camelot":
            tables = extract_with_camelot(tmp_path, pages)
            return tables, [], "camelot"
        elif method == "tabula":
            tables = extract_with_tabula(tmp_path, pages)
            return tables, [], "tabula"
        elif method == "pdfplumber":
            tables, text_pages = extract_with_pdfplumber(pdf_bytes, pages)
            return tables, text_pages, "pdfplumber"
        else:
            # Auto strategy: Camelot → Tabula → pdfplumber
            if _CAMEL0T_AVAILABLE:
                tables = extract_with_camelot(tmp_path, pages)
                if len(tables) > 0:
                    return tables, [], "camelot"
            if _TABULA_AVAILABLE:
                tables = extract_with_tabula(tmp_path, pages)
                if len(tables) > 0:
                    return tables, [], "tabula"
            if _PDFPLUMBER_AVAILABLE:
                tables, text_pages = extract_with_pdfplumber(pdf_bytes, pages)
                return tables, text_pages, "pdfplumber"
            return [], [], "none"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def build_excel_bytes(
    tables: List[pd.DataFrame],
    raw_text_pages: Optional[List[Tuple[int, str]]],
    source_filename: str,
    method_used: str,
    include_text_sheet: bool,
) -> bytes:
    """
    Create an in-memory .xlsx with:
      - One sheet per table: Table_1, Table_2, ...
      - Optional 'Raw_Text' sheet
      - 'Summary' sheet
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Summary
        summary = pd.DataFrame(
            {
                "Item": ["Source file", "Converted on", "Method used", "Tables found"],
                "Value": [
                    source_filename,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    method_used,
                    len(tables),
                ],
            }
        )
        summary.to_excel(writer, index=False, sheet_name=_normalise_sheet_name("Summary"))

        # Tables
        if len(tables) > 0:
            for idx, df in enumerate(tables, start=1):
                sheet = _normalise_sheet_name(f"Table_{idx}")
                # Avoid fully empty sheets
                if df is None or df.empty:
                    continue
                df.to_excel(writer, index=False, sheet_name=sheet)

        # Raw text (optional)
        if include_text_sheet and raw_text_pages:
            text_rows = []
            for pnum, text in raw_text_pages:
                # Keep lines; Excel cell handles long strings
                text_rows.append({"Page": pnum, "Text": text})
            text_df = pd.DataFrame(text_rows)
            text_df.to_excel(writer, index=False, sheet_name=_normalise_sheet_name("Raw_Text"))

    output.seek(0)
    return output.getvalue()


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(
    page_title="PDF → Excel Converter",
    page_icon="📄➡️📊",
    layout="centered",
)

st.title("📄→📊 PDF to Excel Converter")
st.write(
    "Upload a PDF and I’ll extract any tables into an Excel workbook. "
    "Works best with vector PDFs (exported from apps like Word/Excel). "
    "If no tables are detected, you can include a fallback sheet with the raw text."
)

with st.sidebar:
    st.header("Options")
    method = st.radio(
        "Extraction method",
        options=["Auto (recommended)", "Camelot", "Tabula", "pdfplumber"],
        help=(
            "Auto tries Camelot → Tabula → pdfplumber. "
            "Camelot typically works best when the PDF has visible lines; Tabula requires Java. "
            "pdfplumber is pure-Python and a safe fallback."
        ),
    )
    pages_spec = st.text_input(
        "Pages to read",
        value="all",
        help="Examples: all, 1, 1-3, 1,3,5-7",
    )
    include_text = st.checkbox("Include raw text as a sheet (fallback)", value=True)
    st.caption(
        "Tip: For scanned/image-only PDFs you may need OCR first (e.g., Adobe, Tesseract) "
        "before table extraction will work."
    )

uploaded = st.file_uploader("Upload a PDF file", type=["pdf"])

# Capability hinting
capabilities = []
if _CAMEL0T_AVAILABLE:
    capabilities.append("Camelot ✓")
else:
    capabilities.append("Camelot ✗")
if _TABULA_AVAILABLE:
    capabilities.append("Tabula ✓")
else:
    capabilities.append("Tabula ✗")
if _PDFPLUMBER_AVAILABLE:
    capabilities.append("pdfplumber ✓")
else:
    capabilities.append("pdfplumber ✗")
st.caption("Available extractors: " + " • ".join(capabilities))

if uploaded is not None:
    st.success(f"Loaded: **{uploaded.name}** ({uploaded.size/1024:.1f} KB)")
    pdf_bytes = uploaded.read()

    st.session_state.setdefault("cache", {})

    chosen_method = {
        "Auto (recommended)": "auto",
        "Camelot": "camelot",
        "Tabula": "tabula",
        "pdfplumber": "pdfplumber",
    }[method]

    # Normalise pages input
    def _parse_pages_spec(spec: str) -> str:
        if not spec or spec.strip().lower() in ("all", "*"):
            return "all"
        cleaned = spec.replace(" ", "")
        import re as _re
        if _re.fullmatch(r"(\d+(-\d+)?)(,(\d+(-\d+)?))*", cleaned or ""):
            return cleaned
        return "all"

    normalised_pages = _parse_pages_spec(pages_spec)

    if st.button("Convert to Excel"):
        with st.spinner("Extracting tables…"):
            tables, text_pages, method_used = extract_tables(
                pdf_bytes=pdf_bytes, pages=normalised_pages, method=chosen_method
            )

        if method_used == "none":
            st.error(
                "No extraction library is available. Please install at least one of: "
                "`camelot-py`, `tabula-py` (Java required), or `pdfplumber`."
            )
        else:
            if len(tables) > 0:
                st.success(f"Done. Found **{len(tables)}** table(s) using **{method_used}**.")
                # Show preview of up to 3 tables
                st.subheader("Quick preview")
                for i, df in enumerate(tables[:3], start=1):
                    st.caption(f"Table {i} — first 10 rows")
                    st.dataframe(df.head(10), use_container_width=True)
            else:
                st.info(
                    "No tables found. I’ve produced an Excel with a Summary sheet "
                    "and (optionally) a Raw_Text sheet."
                )

            xlsx_bytes = build_excel_bytes(
                tables=tables,
                raw_text_pages=text_pages,
                source_filename=uploaded.name,
                method_used=method_used,
                include_text_sheet=include_text,
            )

            download_name = re.sub(r"\.pdf$", "", uploaded.name, flags=re.IGNORECASE) or "converted"
            download_name = f"{download_name}.xlsx"
            st.download_button(
                label="Download Excel (.xlsx)",
                data=xlsx_bytes,
                file_name=download_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
else:
    st.info("Choose a PDF to get started.")


# Footer note
st.markdown(
    "<hr/>"
    "<small>"
    "Notes: Camelot and Tabula work best on digitally-generated PDFs. Scanned PDFs typically need OCR first. "
    "Camelot requires Ghostscript; Tabula needs Java. If you’re unsure, use ‘Auto’ and include raw text."
    "</small>",
    unsafe_allow_html=True,
)
