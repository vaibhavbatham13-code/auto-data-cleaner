import io
import re

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Auto Data Cleaner", page_icon="🧹", layout="wide")


# ----------------------------- Helpers -----------------------------
NULL_LIKE = {"", "na", "n/a", "nan", "null", "none", "nil", "-", "--", "?", "missing", "undefined"}


def load_file(uploaded, sheet_name=None):
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        raw = uploaded.getvalue()
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
            except Exception:
                continue
        raise ValueError("Could not read the CSV file with common encodings.")
    return pd.read_excel(uploaded, sheet_name=sheet_name)


def clean_column_name(col):
    col = str(col).strip().lower()
    col = re.sub(r"[^\w]+", "_", col)
    return col.strip("_") or "column"


def make_unique(cols):
    seen, out = {}, []
    for c in cols:
        if c in seen:
            seen[c] += 1
            out.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out


def clean_dataframe(df, opts):
    log = []
    df = df.copy()
    original_shape = df.shape

    # 1. Column names
    if opts["clean_cols"]:
        df.columns = make_unique([clean_column_name(c) for c in df.columns])
        log.append("Standardised column names (lowercase, underscores, unique).")

    # 2. Drop fully empty rows / columns
    before = df.shape
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.shape != before:
        log.append(f"Dropped empty rows/columns: {before[0]-df.shape[0]} rows, {before[1]-df.shape[1]} columns.")

    # 3. Trim whitespace and convert null-like strings to NaN
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    for c in obj_cols:
        df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)
        df[c] = df[c].apply(lambda x: np.nan if isinstance(x, str) and x.lower() in NULL_LIKE else x)
    if len(obj_cols):
        log.append("Trimmed whitespace and converted placeholders (NA, null, '-', '?', ...) to missing.")

    # 4. Smart type conversion
    converted = []
    for c in df.select_dtypes(include=["object", "string"]).columns:
        s = df[c].dropna()
        if s.empty:
            continue
        # numeric (handles currency symbols, commas, percent)
        cleaned = s.astype(str).str.replace(r"[,$€£₹%\s]", "", regex=True)
        num = pd.to_numeric(cleaned, errors="coerce")
        if num.notna().mean() >= 0.9:
            full = df[c].astype(str).str.replace(r"[,$€£₹%\s]", "", regex=True)
            df[c] = pd.to_numeric(full.where(df[c].notna()), errors="coerce")
            converted.append(f"{c} → number")
            continue
        # datetime
        if s.astype(str).str.contains(r"[-/:]|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", case=False, regex=True).mean() > 0.8:
            try:
                dt = pd.to_datetime(df[c], errors="coerce", dayfirst=opts["dayfirst"])
                if dt.notna().sum() / max(len(s), 1) >= 0.9:
                    df[c] = dt
                    converted.append(f"{c} → datetime")
            except Exception:
                pass
    if converted:
        log.append("Converted data types: " + ", ".join(converted) + ".")

    # 5. Duplicates
    if opts["drop_dupes"]:
        n = df.duplicated().sum()
        df = df.drop_duplicates().reset_index(drop=True)
        log.append(f"Removed {n} duplicate rows.")

    # 6. Drop columns with too many missing values
    thresh = opts["col_thresh"] / 100
    high_missing = [c for c in df.columns if df[c].isna().mean() > thresh]
    if high_missing:
        df = df.drop(columns=high_missing)
        log.append(f"Dropped columns with >{opts['col_thresh']}% missing: {', '.join(high_missing)}.")

    # 7. Outliers (IQR capping)
    if opts["cap_outliers"]:
        capped = []
        for c in df.select_dtypes(include=[np.number]).columns:
            q1, q3 = df[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0 or pd.isna(iqr):
                continue
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = ((df[c] < lo) | (df[c] > hi)).sum()
            if n_out:
                df[c] = df[c].clip(lo, hi)
                capped.append(f"{c} ({n_out})")
        if capped:
            log.append("Capped outliers (IQR method): " + ", ".join(capped) + ".")

    # 8. Fill missing values -> zero nulls
    filled = []
    for c in df.columns:
        n_missing = df[c].isna().sum()
        if n_missing == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            if opts["num_strategy"] == "Median":
                val = df[c].median()
            elif opts["num_strategy"] == "Mean":
                val = df[c].mean()
            elif opts["num_strategy"] == "Mode":
                val = df[c].mode().iloc[0] if not df[c].mode().empty else 0
            else:
                val = 0
            if pd.isna(val):
                val = 0
            if pd.api.types.is_integer_dtype(df[c]) or (df[c].dropna() % 1 == 0).all():
                val = round(val)
            df[c] = df[c].fillna(val)
            filled.append(f"{c}: {n_missing} → {val}")
        elif pd.api.types.is_datetime64_any_dtype(df[c]):
            if df[c].notna().any():
                val = df[c].median()
            else:
                val = pd.Timestamp("1970-01-01")
            df[c] = df[c].fillna(val)
            filled.append(f"{c}: {n_missing} → {str(val)[:10]}")
        else:
            mode = df[c].mode()
            if opts["cat_strategy"] == "Mode" and not mode.empty:
                val = mode.iloc[0]
            else:
                val = "Unknown"
            df[c] = df[c].fillna(val)
            filled.append(f"{c}: {n_missing} → '{val}'")
    if filled:
        log.append("Filled missing values — " + "; ".join(filled) + ".")

    # 9. Final safety net: guarantee no nulls
    if df.isna().any().any():
        for c in df.columns[df.isna().any()]:
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(0)
            elif pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = df[c].fillna(pd.Timestamp("1970-01-01"))
            else:
                df[c] = df[c].fillna("Unknown")
        log.append("Applied safety-net fill so no null values remain.")

    log.append(f"Shape: {original_shape[0]}×{original_shape[1]} → {df.shape[0]}×{df.shape[1]}.")
    return df, log


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Clean Data")
    return buf.getvalue()


def profile(df):
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "missing_%": (df.isna().mean() * 100).round(2),
            "unique": df.nunique(),
        }
    )


# ----------------------------- UI -----------------------------
st.title("🧹 Auto Data Cleaner")
st.caption("Upload a CSV or Excel file → automatic cleaning → zero null values → download.")

with st.sidebar:
    st.header("⚙️ Cleaning options")
    clean_cols = st.checkbox("Standardise column names", True)
    drop_dupes = st.checkbox("Remove duplicate rows", True)
    cap_outliers = st.checkbox("Cap numeric outliers (IQR)", False)
    dayfirst = st.checkbox("Dates are day-first (DD/MM/YYYY)", True)
    col_thresh = st.slider("Drop columns with missing % above", 50, 100, 90)
    num_strategy = st.selectbox("Fill numeric missing with", ["Median", "Mean", "Mode", "Zero"])
    cat_strategy = st.selectbox("Fill text missing with", ["Mode", "Unknown"])

uploaded = st.file_uploader("Upload your file", type=["csv", "xlsx", "xls"])

if uploaded:
    try:
        sheet = None
        if uploaded.name.lower().endswith((".xlsx", ".xls")):
            xls = pd.ExcelFile(uploaded)
            sheet = st.selectbox("Select sheet", xls.sheet_names) if len(xls.sheet_names) > 1 else xls.sheet_names[0]
            uploaded.seek(0)
        raw_df = load_file(uploaded, sheet)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()

    if raw_df.empty:
        st.warning("The file appears to be empty.")
        st.stop()

    opts = dict(
        clean_cols=clean_cols,
        drop_dupes=drop_dupes,
        cap_outliers=cap_outliers,
        dayfirst=dayfirst,
        col_thresh=col_thresh,
        num_strategy=num_strategy,
        cat_strategy=cat_strategy,
    )
    clean_df, log = clean_dataframe(raw_df, opts)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original rows", f"{raw_df.shape[0]:,}")
    c2.metric("Clean rows", f"{clean_df.shape[0]:,}")
    c3.metric("Nulls before", f"{int(raw_df.isna().sum().sum()):,}")
    c4.metric("Nulls after", f"{int(clean_df.isna().sum().sum()):,}")

    tab1, tab2, tab3, tab4 = st.tabs(["Original", "Cleaned", "Cleaning report", "Column profile"])
    with tab1:
        st.dataframe(raw_df.head(200), use_container_width=True)
    with tab2:
        st.dataframe(clean_df.head(200), use_container_width=True)
    with tab3:
        for item in log:
            st.markdown(f"- {item}")
    with tab4:
        left, right = st.columns(2)
        left.subheader("Before")
        left.dataframe(profile(raw_df), use_container_width=True)
        right.subheader("After")
        right.dataframe(profile(clean_df), use_container_width=True)

    st.divider()
    st.subheader("⬇️ Download clean data")
    base = uploaded.name.rsplit(".", 1)[0]
    d1, d2 = st.columns(2)
    d1.download_button("Download as CSV", to_csv_bytes(clean_df), f"{base}_clean.csv", "text/csv", use_container_width=True)
    d2.download_button(
        "Download as Excel",
        to_excel_bytes(clean_df),
        f"{base}_clean.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
else:
    st.info("👆 Upload a CSV or Excel file to get started.")
