import io
import os
import re
import string
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D


st.set_page_config(page_title="Solar Cell Analysis", layout="wide")


def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str) -> bytes:
    buf = io.BytesIO()
    # Prefer xlsxwriter, but gracefully fall back for environments where it isn't installed.
    try:
        writer_ctx = pd.ExcelWriter(buf, engine="xlsxwriter")
    except Exception:
        writer_ctx = pd.ExcelWriter(buf, engine="openpyxl")

    with writer_ctx as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    buf.seek(0)
    return buf.getvalue()


def decode_bytes(raw: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def parse_simple_filename(filename: str) -> Dict[str, str]:
    stem = filename.rsplit(".", 1)[0]
    toks = stem.split("_")
    out = {
        "device": "Unknown",
        "batch": "Unknown",
        "sample": "Unknown",
        "measurement": "Unknown",
        "day": "Unknown",
    }
    if not toks:
        return out

    out["device"] = toks[0]
    m = re.match(r"^([A-Za-z]+)(\d+)$", out["device"])
    if m:
        out["batch"], out["sample"] = m.group(1), m.group(2)

    if len(toks) >= 4 and toks[1].upper() == "MPP":
        out["measurement"] = f"{toks[1]}_{toks[2]}"
        out["day"] = toks[3]
    else:
        if len(toks) > 1:
            out["measurement"] = toks[1]
        if len(toks) > 2:
            out["day"] = toks[2]
    return out


def parse_split_filename(filename: str, sep: str, idx_device: int, idx_measurement: int, idx_day: int) -> Dict[str, str]:
    stem = filename.rsplit(".", 1)[0]
    toks = [t for t in re.split(r"\s+", stem) if t] if sep.lower() == "whitespace" else stem.split(sep)
    out = {
        "device": toks[idx_device] if idx_device < len(toks) else "Unknown",
        "measurement": toks[idx_measurement] if idx_measurement < len(toks) else "Unknown",
        "day": toks[idx_day] if idx_day < len(toks) else "Unknown",
        "batch": "Unknown",
        "sample": "Unknown",
    }
    m = re.match(r"^([A-Za-z]+)(\d+)$", str(out["device"]))
    if m:
        out["batch"], out["sample"] = m.group(1), m.group(2)
    return out


def parse_regex_filename(filename: str, pattern: str) -> Dict[str, str]:
    stem = filename.rsplit(".", 1)[0]
    out = {"device": "Unknown", "measurement": "Unknown", "day": "Unknown", "batch": "Unknown", "sample": "Unknown"}
    try:
        m = re.search(pattern, stem)
    except re.error:
        return out
    if not m:
        return out
    g = m.groupdict()
    out["device"] = str(g.get("device", out["device"]))
    out["measurement"] = str(g.get("measurement", out["measurement"]))
    out["day"] = str(g.get("day", out["day"]))
    out["batch"] = str(g.get("batch", out["batch"]))
    out["sample"] = str(g.get("sample", out["sample"]))
    if out["batch"] == "Unknown" or out["sample"] == "Unknown":
        mm = re.match(r"^([A-Za-z]+)(\d+)$", out["device"])
        if mm:
            if out["batch"] == "Unknown":
                out["batch"] = mm.group(1)
            if out["sample"] == "Unknown":
                out["sample"] = mm.group(2)
    return out


def parse_single_file_payload(
    payload: Tuple[str, bytes],
    parser_mode: str,
    sep_mode: str,
    skiprows: int,
    split_sep: str,
    idx_device: int,
    idx_measurement: int,
    idx_day: int,
    regex_pattern: str,
) -> Tuple[Dict[str, str], List[pd.DataFrame], List[Dict[str, str]]]:
    fname, raw = payload
    text = decode_bytes(raw)

    if parser_mode == "Simple default (H03_L02_D02)":
        meta = parse_simple_filename(fname)
    elif parser_mode == "Split by separator":
        meta = parse_split_filename(fname, split_sep, idx_device, idx_measurement, idx_day)
    else:
        meta = parse_regex_filename(fname, regex_pattern)

    meta_row = {"file": fname, **meta}
    jv_parts: List[pd.DataFrame] = []
    box_rows: List[Dict[str, str]] = []

    # JV long parsing
    try:
        sep = "\t" if sep_mode == "Tab" else r"\s+"
        df = pd.read_csv(io.StringIO(text), sep=sep, skiprows=skiprows, header=None, engine="python").dropna(axis=1, how="all")
        n_pairs = max((df.shape[1] - 1) // 2, 0)
        labels = list(string.ascii_lowercase)
        x = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        for i in range(n_pairs):
            p = labels[i] if i < len(labels) else f"p{i+1}"
            rc, fc = 1 + 2 * i, 2 + 2 * i
            if fc >= df.shape[1]:
                continue
            y_r = pd.to_numeric(df.iloc[:, rc], errors="coerce")
            y_f = pd.to_numeric(df.iloc[:, fc], errors="coerce")
            tmp_r = pd.DataFrame(
                {
                    "file": fname,
                    "device": meta["device"],
                    "batch": meta["batch"],
                    "sample": meta["sample"],
                    "measurement": meta["measurement"],
                    "day": meta["day"],
                    "pixel": p,
                    "scan": "Reverse",
                    "voltage": x,
                    "current": y_r,
                }
            )
            tmp_f = tmp_r.copy()
            tmp_f["scan"] = "Forward"
            tmp_f["current"] = y_f
            jv_parts.extend([tmp_r, tmp_f])
    except Exception:
        pass

    # Box summary parsing
    lines = text.splitlines()
    if len(lines) >= 13:
        params = ["Jsc", "Voc", "FF", "PCE"]
        for i, line in enumerate(lines[9:13]):
            vals = []
            for p in line.split("\t")[1:-1]:
                try:
                    vals.append(float(p))
                except ValueError:
                    pass
            rev = [v for idx, v in enumerate(vals) if idx % 2 == 0]
            fwd = [v for idx, v in enumerate(vals) if idx % 2 == 1]
            if params[i] == "Jsc":
                rev = [-x for x in rev]
                fwd = [-x for x in fwd]
            for v in rev:
                box_rows.append(
                    {
                        "file": fname,
                        "Device": meta["device"],
                        "Measurement": meta["measurement"],
                        "Day": meta["day"],
                        "Parameter": params[i],
                        "Value": v,
                        "Scan": "Reverse",
                    }
                )
            for v in fwd:
                box_rows.append(
                    {
                        "file": fname,
                        "Device": meta["device"],
                        "Measurement": meta["measurement"],
                        "Day": meta["day"],
                        "Parameter": params[i],
                        "Value": v,
                        "Scan": "Forward",
                    }
                )

    return meta_row, jv_parts, box_rows


@st.cache_data(show_spinner=False)
def parse_raw_files_cached(
    file_payloads: List[Tuple[str, bytes]],
    parser_mode: str,
    sep_mode: str,
    skiprows: int,
    split_sep: str,
    idx_device: int,
    idx_measurement: int,
    idx_day: int,
    regex_pattern: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return metadata_df, jv_long_df, box_df (without group mapping)."""
    meta_rows: List[Dict[str, str]] = []
    jv_chunks: List[pd.DataFrame] = []
    box_rows: List[Dict[str, str]] = []

    worker = partial(
        parse_single_file_payload,
        parser_mode=parser_mode,
        sep_mode=sep_mode,
        skiprows=skiprows,
        split_sep=split_sep,
        idx_device=idx_device,
        idx_measurement=idx_measurement,
        idx_day=idx_day,
        regex_pattern=regex_pattern,
    )

    if len(file_payloads) > 1:
        max_workers = min(len(file_payloads), max(2, min(8, os.cpu_count() or 4)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(worker, file_payloads))
    else:
        results = [worker(p) for p in file_payloads]

    for meta_row, jv_parts, one_box_rows in results:
        meta_rows.append(meta_row)
        if jv_parts:
            jv_chunks.extend(jv_parts)
        if one_box_rows:
            box_rows.extend(one_box_rows)

    meta_df = pd.DataFrame(meta_rows)
    jv_df = pd.concat(jv_chunks, ignore_index=True) if jv_chunks else pd.DataFrame()
    box_df = pd.DataFrame(box_rows)

    if not jv_df.empty:
        jv_df = jv_df.dropna(subset=["voltage", "current"])
    return meta_df, jv_df, box_df


def style_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].fillna("Unknown").astype(str)
    return out


def bump_y(y: float, used: List[float], dy: float) -> float:
    yy = y
    while any(abs(yy - u) < dy for u in used):
        yy += dy
    used.append(yy)
    return yy


@st.cache_data(show_spinner=False)
def jv_excel_bytes_cached(jv_df: pd.DataFrame) -> bytes:
    return df_to_excel_bytes(style_df(jv_df), "JV_Curves")


@st.cache_data(show_spinner=False)
def box_excel_by_parameter_bytes_cached(box_filtered_df: pd.DataFrame, group_order: List[str]) -> bytes:
    buf = io.BytesIO()
    param_order = ["Jsc", "FF", "Voc", "PCE"]
    try:
        writer_ctx = pd.ExcelWriter(buf, engine="xlsxwriter")
    except Exception:
        writer_ctx = pd.ExcelWriter(buf, engine="openpyxl")

    with writer_ctx as writer:
        for p in param_order:
            dp = box_filtered_df[box_filtered_df["Parameter"] == p]
            col_data = {}
            for g in group_order:
                vals = dp[dp["Group"].astype(str) == str(g)]["Value"].dropna().tolist()
                col_data[str(g)] = vals

            max_len = max((len(v) for v in col_data.values()), default=0)
            if max_len == 0:
                sheet_df = pd.DataFrame(columns=[str(g) for g in group_order])
            else:
                padded = {k: v + [np.nan] * (max_len - len(v)) for k, v in col_data.items()}
                sheet_df = pd.DataFrame(padded)

            sheet_df.to_excel(writer, index=False, sheet_name=p)
    buf.seek(0)
    return buf.getvalue()


st.markdown(
    """
<style>
div[role="radiogroup"] label p { font-size: 1.15rem !important; font-weight: 700 !important; }
.big-card {padding:14px; border:1px solid #e5e7eb; border-radius:12px; margin-bottom:10px; background:#fafafa;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Solar Cell Analysis App")
st.markdown('<div class="big-card"><h3>Home / Setup</h3><p><b>How do you name your files?</b> Configure this first, then run JV or Boxplots.</p></div>', unsafe_allow_html=True)

up_col1, up_col2 = st.columns(2)
with up_col1:
    uploaded_files = st.file_uploader("1) Upload raw files (.txt/.dat)", type=["txt", "dat"], accept_multiple_files=True)
with up_col2:
    mapping_file = st.file_uploader("2) Upload names.xlsx (optional)", type=["xlsx"])

if not uploaded_files:
    st.info("Upload raw files to continue.")
    st.stop()

parser_mode = st.radio(
    "3) File naming scheme",
    ["Simple default (H03_L02_D02)", "Split by separator", "Regex (advanced)"],
    horizontal=True,
)

split_sep = "_"
idx_device, idx_measurement, idx_day = 0, 1, 2
regex_pattern = r"(?P<device>[^_]+)_(?P<measurement>[^_]+)_(?P<day>[^_]+)"

if parser_mode == "Split by separator":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        split_sep = st.text_input("Separator", value="_")
    with c2:
        idx_device = st.number_input("Device index", min_value=0, max_value=20, value=0, step=1)
    with c3:
        idx_measurement = st.number_input("Measurement index", min_value=0, max_value=20, value=1, step=1)
    with c4:
        idx_day = st.number_input("Day index", min_value=0, max_value=20, value=2, step=1)
elif parser_mode == "Regex (advanced)":
    regex_pattern = st.text_input("Regex with named groups (device, measurement, day)", value=regex_pattern)

skiprows = st.number_input("JV header rows to skip", min_value=0, max_value=1000, value=21, step=1)
sep_mode = st.selectbox("JV table separator", ["Tab", "Whitespace"], index=0)

file_payloads = [(f.name, f.getvalue()) for f in uploaded_files]

with st.spinner("Parsing and indexing data (parallel per file)..."):
    meta_df, jv_df, box_df = parse_raw_files_cached(
        file_payloads,
        parser_mode,
        sep_mode,
        int(skiprows),
        split_sep,
        int(idx_device),
        int(idx_measurement),
        int(idx_day),
        regex_pattern,
    )

if meta_df.empty:
    st.error("Could not parse uploaded files.")
    st.stop()

raw_samples = sorted(meta_df["device"].dropna().astype(str).unique().tolist())
excel_map_df = pd.DataFrame(columns=["sample", "parameter"])

if mapping_file is not None:
    xls = pd.ExcelFile(io.BytesIO(mapping_file.getvalue()))
    sheet = st.selectbox("names.xlsx sheet", xls.sheet_names)
    tmp = pd.read_excel(io.BytesIO(mapping_file.getvalue()), sheet_name=sheet, header=None, usecols=[0, 1]).dropna(subset=[0, 1])
    excel_map_df = tmp.rename(columns={0: "sample", 1: "parameter"})
    excel_map_df["sample"] = excel_map_df["sample"].astype(str)
    excel_map_df["parameter"] = excel_map_df["parameter"].astype(str)

st.markdown("**4) Sample → group mapping**")
manual_map = st.checkbox("Edit mapping manually", value=False)
if manual_map:
    init = excel_map_df.copy() if not excel_map_df.empty else pd.DataFrame({"sample": raw_samples, "parameter": ["Unknown"] * len(raw_samples)})
    map_df = st.data_editor(init, num_rows="dynamic", use_container_width=True)
else:
    map_df = excel_map_df.copy()

if map_df.empty:
    mapping_dict = {s: s for s in raw_samples}
else:
    map_df = map_df.dropna(subset=["sample", "parameter"])
    map_df["sample"] = map_df["sample"].astype(str)
    map_df["parameter"] = map_df["parameter"].astype(str)
    mapping_dict = dict(zip(map_df["sample"], map_df["parameter"]))

missing = sorted(set(raw_samples) - set(mapping_dict.keys()))
extra = sorted(set(mapping_dict.keys()) - set(raw_samples))
if missing:
    st.warning(f"Missing mapping for: {', '.join(missing)}")
if extra:
    st.warning(f"Mapping includes unknown samples: {', '.join(extra)}")
if not missing and not extra:
    st.success("Mapping table checks out ✅")

meta_df = meta_df.copy()
meta_df["group"] = meta_df["device"].map(mapping_dict).fillna("Unknown")
if not jv_df.empty:
    jv_df = jv_df.merge(meta_df[["file", "group"]], on="file", how="left")
if not box_df.empty:
    box_df["Group"] = box_df["Device"].map(mapping_dict).fillna("Unknown")

st.markdown("**Parsed indexing preview**")
st.dataframe(style_df(meta_df[["file", "device", "batch", "sample", "measurement", "day", "group"]]), use_container_width=True)

section = st.radio("Run section", ["📈 JV Curves", "📦 Boxplots"], horizontal=True)

if section == "📈 JV Curves":
    st.subheader("JV Curves")
    if jv_df.empty:
        st.warning("No JV table data parsed.")
        st.stop()

    m_opts = sorted(meta_df["measurement"].astype(str).unique().tolist())
    d_opts = sorted(meta_df["day"].astype(str).unique().tolist())
    s_opts = sorted(meta_df["device"].astype(str).unique().tolist())
    g_opts = sorted(meta_df["group"].astype(str).unique().tolist())

    sel_m = st.multiselect("Measurement", m_opts, default=["L02"] if "L02" in m_opts else m_opts)
    sel_d = st.multiselect("Day", d_opts, default=d_opts)
    sel_s = st.multiselect("Sample", s_opts, default=s_opts)
    sel_g = st.multiselect("Group", g_opts, default=g_opts)

    meta_f = meta_df[
        meta_df["measurement"].isin(sel_m)
        & meta_df["day"].isin(sel_d)
        & meta_df["device"].isin(sel_s)
        & meta_df["group"].isin(sel_g)
    ]
    files = meta_f["file"].tolist()
    if not files:
        st.warning("No files match filters.")
        st.stop()

    mode = st.radio("Mode", ["Single file", "Overlay"], horizontal=True)
    pix_opts = sorted(jv_df[jv_df["file"].isin(files)]["pixel"].astype(str).unique().tolist())
    sel_pix = st.multiselect("Pixels", pix_opts, default=pix_opts[:6])

    xlim_on = st.checkbox("Use fixed limits", value=True)
    x_min = st.number_input("X min [V]", value=-0.10)
    x_max = st.number_input("X max [V]", value=1.25)
    y_min = st.number_input("Y min [mA/cm²]", value=-25.0)
    y_max = st.number_input("Y max [mA/cm²]", value=5.0)

    if mode == "Single file":
        files_plot = [st.selectbox("File", files)]
    else:
        ov_groups = sorted(meta_f["group"].astype(str).unique().tolist())
        sel_ov_groups = st.multiselect("Overlay groups (tickbox selector)", ov_groups, default=ov_groups)
        files_plot = meta_f[meta_f["group"].isin(sel_ov_groups)]["file"].tolist()
        files_plot = st.multiselect("Overlay files", files_plot, default=files_plot)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("tab20")
    keys = sorted(set((f, p) for f in files_plot for p in sel_pix))
    key_color = {k: cmap(i % 20) for i, k in enumerate(keys)}
    meta_map = meta_f.set_index("file")[["group", "device", "measurement", "day"]].to_dict("index")

    sub = jv_df[jv_df["file"].isin(files_plot) & jv_df["pixel"].isin(sel_pix)]
    for (f, p, sc), chunk in sub.groupby(["file", "pixel", "scan"]):
        mrow = meta_map.get(f, {"group": "Unknown", "device": "Unknown", "measurement": "Unknown", "day": "Unknown"})
        label = f"{mrow['group']} | {mrow['device']} | {mrow['measurement']} | {mrow['day']} | {p} {sc[:3].lower()}"
        ax.plot(
            chunk["voltage"],
            chunk["current"],
            color=key_color[(f, p)],
            linestyle="-" if sc == "Reverse" else "--",
            alpha=1.0 if sc == "Reverse" else 0.85,
            label=label,
        )

    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Voltage [V]")
    ax.set_ylabel("Current density [mA/cm²]")
    if xlim_on:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    n_lines = len(sub.groupby(["file", "pixel", "scan"]))
    if n_lines <= 40:
        ax.legend(fontsize=7, loc="best")
    else:
        st.info(f"Legend hidden for performance/readability ({n_lines} curves).")
    fig.tight_layout()
    st.pyplot(fig)

    if st.button("Prepare clean JV curves Excel", key="prepare_jv_excel"):
        st.session_state["jv_excel_ready"] = True
    if st.session_state.get("jv_excel_ready", False):
        st.download_button(
            "Download clean JV curves Excel",
            data=jv_excel_bytes_cached(jv_df),
            file_name="clean_jv_curves.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_jv_excel",
        )

else:
    st.subheader("Boxplots")
    if box_df.empty:
        st.warning("No box summary parsed.")
        st.stop()

    m_opts = sorted(box_df["Measurement"].astype(str).unique().tolist())
    sel_m = st.selectbox("Measurement", m_opts, index=m_opts.index("L02") if "L02" in m_opts else 0)

    d = box_df[box_df["Measurement"] == sel_m].copy()
    d_opts = sorted(d["Day"].astype(str).unique().tolist())
    s_opts = sorted(d["Device"].astype(str).unique().tolist())
    g_opts = sorted(d["Group"].astype(str).unique().tolist())
    sel_d = st.multiselect("Day", d_opts, default=d_opts)
    sel_s = st.multiselect("Sample", s_opts, default=s_opts)
    sel_g = st.multiselect("Group", g_opts, default=g_opts)
    d = d[d["Day"].isin(sel_d) & d["Device"].isin(sel_s) & d["Group"].isin(sel_g)]

    box_alpha = st.slider("Box transparency", 0.2, 1.0, 0.85, 0.05)
    fwd_alpha = max(0.1, box_alpha - 0.2)
    show_points = st.checkbox("Show points", value=False)
    show_median = st.checkbox("Show median labels", value=False)
    show_max = st.checkbox("Show highest labels", value=False)

    groups = sorted(d["Group"].astype(str).unique().tolist())
    order_mode = st.radio("X-axis order", ["Excel order", "Alphabetical", "Custom"], horizontal=True)
    if order_mode == "Excel order" and not map_df.empty:
        excel_order = [g for g in map_df["parameter"].drop_duplicates().astype(str).tolist() if g in groups]
        groups = excel_order + [g for g in groups if g not in excel_order]
    elif order_mode == "Alphabetical":
        groups = sorted(groups)
    else:
        custom = st.text_input("Custom order (comma-separated)", value=", ".join(groups))
        c = [x.strip() for x in custom.split(",") if x.strip()]
        groups = [g for g in c if g in groups] + [g for g in groups if g not in c]

    params = ["Jsc", "Voc", "FF", "PCE"]
    palette = {g: plt.get_cmap("Set2")(i % 8) for i, g in enumerate(groups)}
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for idx, p in enumerate(params):
        ax = axes[idx // 2, idx % 2]
        dp = d[d["Parameter"] == p]
        if dp.empty:
            ax.set_title(f"{p} (no data)")
            continue
        ymin, ymax = dp["Value"].min(), dp["Value"].max()
        yr = max(ymax - ymin, 1e-9)
        used = []

        for k, g in enumerate(groups):
            base = palette[g]
            for scan, xpos, alpha in [("Reverse", k - 0.18, box_alpha), ("Forward", k + 0.18, fwd_alpha)]:
                vals = dp[(dp["Group"] == g) & (dp["Scan"] == scan)]["Value"].dropna().values
                if len(vals) == 0:
                    continue
                b = ax.boxplot([vals], positions=[xpos], widths=0.28, patch_artist=True, showfliers=False, medianprops={"color": "black", "linewidth": 1.4})
                rgba = to_rgba(base, alpha=alpha)
                for patch in b["boxes"]:
                    patch.set_facecolor(rgba)
                    patch.set_edgecolor(base)
                    patch.set_linewidth(1.7)
                for ln in b["whiskers"] + b["caps"]:
                    ln.set_color(base)
                    ln.set_linewidth(1.3)
                if show_points:
                    jit = np.random.uniform(-0.035, 0.035, size=len(vals))
                    ax.scatter(np.full(len(vals), xpos) + jit, vals, s=14, color=to_rgba(base, min(1, alpha + 0.1)), edgecolors="white", linewidths=0.25, zorder=3)

                med = float(np.median(vals))
                maxv = float(np.max(vals))
                if show_median:
                    yy = bump_y(med + 0.02 * yr, used, 0.04 * yr)
                    ax.text(xpos, yy, f"med {med:.2f}", ha="center", va="bottom", fontsize=8, clip_on=True, bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, boxstyle="round,pad=0.2"))
                if show_max:
                    yy = bump_y(maxv + 0.08 * yr, used, 0.05 * yr)
                    ax.text(xpos, yy, f"max {maxv:.2f}", ha="center", va="bottom", fontsize=8, clip_on=True, bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, boxstyle="round,pad=0.2"))

        ax.set_facecolor("#fafafa")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(f"Parameter: {p}")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=40, ha="right")
        ax.set_ylim(ymin - 0.1 * yr, ymax + (0.30 * yr if show_max else 0.2 * yr))

    axes[0, 1].legend(
        handles=[
            Line2D([0], [0], color="black", linewidth=2, linestyle="-", label="Reverse"),
            Line2D([0], [0], color="black", linewidth=2, linestyle="-", alpha=fwd_alpha, label="Forward"),
        ],
        title="Scan",
        frameon=True,
    )
    fig.suptitle(f"Measurement: {sel_m}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    st.pyplot(fig)

    if st.button("Prepare clean boxplot Excel (one sheet per parameter)", key="prepare_box_excel"):
        st.session_state["box_excel_ready"] = True
    if st.session_state.get("box_excel_ready", False):
        st.download_button(
            "Download clean boxplot data",
            data=box_excel_by_parameter_bytes_cached(d, groups),
            file_name="clean_boxplot_data_by_parameter.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_box_excel",
        )
