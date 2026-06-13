import os, io, zipfile, tempfile, math, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
import plotly.graph_objects as go

from scipy import signal, sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import make_interp_spline, CubicSpline
from scipy.spatial.distance import pdist, squareform


st.set_page_config(page_title="VRC / HRV RRi Analyzer Pro v3.8", layout="wide")


# ============================================================
# CONFIGURACIÓN
# ============================================================

LAMBDA_DEFAULT = 500
FS_INTERP = 4.0

APPLY_LAMBDA = {
    "PSD": True,
    "SampEn": True,
    "ApEn": True,
    "MSE": True,
    "SDNN": False,
    "RMSSD": False,
    "SD1": False,
    "SD2": False,
    "DFA": False,
    "RQA": False,
    "D2": False,
    "HVG": False,
}

DOMAIN_GROUPS = {
    "Amplitud": ["SDNN", "SD2", "TOTAL"],
    "Vagal": ["RMSSD", "SD1", "HF", "pNN50"],
    "Complejidad": ["DFA_alpha1", "DFA_alpha2", "ApEn", "SampEn", "D2", "ShanEn"],
    "Recurrencia": ["REC", "DET", "Lmean", "Lmax"],
}

# Fases del protocolo:
# 1 ventana basal, 6 ventanas de ejercicio y 3 ventanas de recuperación.
PHASES = ["Basal"] + [f"E{i}" for i in range(1, 7)] + [f"R{i}" for i in range(1, 4)]
PHASE_GROUP = {
    "Basal": "Basal",
    **{f"E{i}": "Ejercicio" for i in range(1, 7)},
    **{f"R{i}": "Recuperación" for i in range(1, 4)},
}
PHASE_COLORS = {
    "Basal": "rgba(0,150,255,0.20)",
    "Ejercicio": "rgba(255,140,0,0.18)",
    "Recuperación": "rgba(0,200,100,0.18)",
}


# ============================================================
# UTILIDADES
# ============================================================

def read_rri_file(uploaded_file):
    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="ignore")
    vals = []

    for line in text.replace(";", "\n").replace("\t", "\n").splitlines():
        line = line.strip().replace(",", ".")
        if not line:
            continue
        for p in line.split():
            try:
                vals.append(float(p))
            except Exception:
                pass

    rr = np.asarray(vals, dtype=float)
    rr = rr[np.isfinite(rr)]

    if len(rr) == 0:
        raise ValueError("No se han detectado RRi numéricos.")

    # Si vienen en milisegundos, convertir a segundos
    if np.nanmedian(rr) > 10:
        rr = rr / 1000.0

    # Filtro fisiológico amplio
    rr = rr[(rr >= 0.3) & (rr <= 2.0)]

    if len(rr) == 0:
        raise ValueError("Tras el filtrado fisiológico no quedan RRi válidos.")

    return rr


def sec_to_hms(seconds):
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def hms_to_sec(s):
    parts = str(s).split(":")
    parts = [float(p) for p in parts]

    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def cumulative_time(rr):
    return np.cumsum(rr)


def cut_segment(rr, start_s, end_s):
    t = cumulative_time(rr)
    return rr[(t >= start_s) & (t <= end_s)]


def sanitize_record_name(name):
    name = Path(str(name)).stem
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", name).strip("_")
    return name or "registro"


def calculate_metrics_for_windows(rr, windows, min_rr=30):
    segments = {name: cut_segment(rr, s, e) for name, (s, e) in windows.items()}
    valid_segments = {name: len(seg) >= min_rr for name, seg in segments.items()}
    results = {}
    graphs = {}
    for name, seg in segments.items():
        if not valid_segments.get(name, False):
            continue
        res, G = calculate_all(seg)
        results[name] = res
        graphs[name] = G
    if results:
        metrics_df = pd.DataFrame(results).T
        metrics_df = metrics_df.reindex([ph for ph in PHASES if ph in metrics_df.index])
    else:
        metrics_df = pd.DataFrame()
    return metrics_df, segments, valid_segments, graphs


def add_record_column(metrics_df, record_name):
    df = metrics_df.copy()
    df.insert(0, "Registro", record_name)
    df.insert(1, "Fase", df.index)
    return df.reset_index(drop=True)


def build_comparison_tables(records_results):
    dfs = []
    for rec_name, df in records_results.items():
        if df is not None and not df.empty:
            dfs.append(add_record_column(df, rec_name))
    if not dfs:
        return pd.DataFrame(), {}
    long_df = pd.concat(dfs, ignore_index=True)
    pivot_tables = {}
    numeric_cols = [c for c in long_df.columns if c not in ["Registro", "Fase", "Graph_type", "Graph_interpretation"] and pd.api.types.is_numeric_dtype(long_df[c])]
    for var in numeric_cols:
        pivot_tables[var] = long_df.pivot_table(index="Fase", columns="Registro", values=var, aggfunc="first").reindex([ph for ph in PHASES if ph in long_df["Fase"].unique()])
    return long_df, pivot_tables


def plot_compare_variable(pivot_df, variable, out_path):
    if pivot_df is None or pivot_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5)); ax.text(0.5,0.5,"No hay datos para comparar",ha="center",va="center"); ax.axis("off"); fig.savefig(out_path,dpi=300,bbox_inches="tight"); plt.close(fig); return
    phases=list(pivot_df.index); x=np.arange(len(phases)); fig, ax=plt.subplots(figsize=(14,7))
    for col in pivot_df.columns:
        y=pivot_df[col].astype(float).values; xs,ys=smooth_line(x,y)
        if len(xs)>0: ax.plot(xs,ys,linewidth=2.2,label=str(col))
        ax.scatter(x,y,s=55)
    ax.set_xticks(x); ax.set_xticklabels(phases); ax.set_title(f"Comparativa entre registros: {variable}"); ax.set_ylabel(variable); ax.grid(axis="y",alpha=0.25); ax.legend()
    fig.savefig(out_path,dpi=300,bbox_inches="tight"); plt.close(fig)


def plot_compare_domains(records_results, method, out_path):
    dom_records={}
    for rec_name, df in records_results.items():
        if df is not None and not df.empty and "Basal" in df.index:
            dom_records[rec_name]=domain_values(df, method=method)
    if not dom_records:
        fig, ax=plt.subplots(figsize=(10,5)); ax.text(0.5,0.5,"No hay dominios comparables. Se necesita Basal válido.",ha="center",va="center"); ax.axis("off"); fig.savefig(out_path,dpi=300,bbox_inches="tight"); plt.close(fig); return pd.DataFrame()
    rows=[]
    for rec_name, dom_df in dom_records.items():
        tmp=dom_df.copy(); tmp.insert(0,"Registro",rec_name); tmp.insert(1,"Fase",tmp.index); rows.append(tmp.reset_index(drop=True))
    long_dom=pd.concat(rows, ignore_index=True)
    fig, axes=plt.subplots(2,2,figsize=(16,10)); axes=axes.flatten()
    for ax, dom in zip(axes, DOMAIN_GROUPS.keys()):
        pivot=long_dom.pivot_table(index="Fase", columns="Registro", values=dom, aggfunc="first").reindex([ph for ph in PHASES if ph in long_dom["Fase"].unique()])
        phases=list(pivot.index); x=np.arange(len(phases))
        for rec in pivot.columns:
            y=pivot[rec].astype(float).values; xs,ys=smooth_line(x,y)
            if len(xs)>0: ax.plot(xs,ys,linewidth=2.0,label=str(rec))
            ax.scatter(x,y,s=35)
        ax.axhline(100,linestyle="--",alpha=0.5); ax.set_xticks(x); ax.set_xticklabels(phases,rotation=45); ax.set_title(dom); ax.grid(axis="y",alpha=0.25)
    axes[0].legend(fontsize=8); fig.suptitle("Comparativa de dominios normalizados. Basal = 100% en cada registro",fontsize=16); plt.tight_layout(); fig.savefig(out_path,dpi=300,bbox_inches="tight"); plt.close(fig)
    return long_dom


# ============================================================
# CORRECCIÓN DE ARTEFACTOS TIPO KUBIOS APROXIMADA
# ============================================================

def correct_artifacts_kubios_like(rr, level="none", window=5):
    """
    Corrección aproximada tipo Kubios.
    No replica el algoritmo propietario exacto.

    Método:
    - Mediana local.
    - Umbral absoluto en segundos.
    - Interpolación lineal de RRi marcados.
    """
    rr = np.asarray(rr, dtype=float)
    rr_corr = rr.copy()
    n = len(rr)

    if level == "none" or n < 10:
        return rr_corr, np.zeros(n, dtype=bool), {
            "level": level,
            "n_artifacts": 0,
            "percent_artifacts": 0.0
        }

    thresholds = {
        "very low": 0.45,
        "low": 0.35,
        "medium": 0.25,
        "strong": 0.15,
        "very strong": 0.05,
    }

    th = thresholds.get(level, 0.25)

    local = pd.Series(rr).rolling(window=window, center=True, min_periods=1).median().to_numpy()
    abs_dev = np.abs(rr - local)

    artifacts = abs_dev > th

    # Protección contra sobrecorrección
    if np.mean(artifacts) > 0.30:
        artifacts[:] = False

    idx = np.arange(n)
    good = ~artifacts

    if np.sum(good) >= 2 and np.sum(artifacts) > 0:
        rr_corr[artifacts] = np.interp(idx[artifacts], idx[good], rr[good])

    info = {
        "level": level,
        "n_artifacts": int(np.sum(artifacts)),
        "percent_artifacts": float(100 * np.mean(artifacts))
    }

    return rr_corr, artifacts, info


# ============================================================
# DETRENDING SMOOTHNESS PRIORS
# ============================================================

def smoothness_priors_detrend(y, lam=500):
    """
    Smoothness priors detrending.
    Devuelve la señal detrendida: y - tendencia.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    if n < 5:
        return y

    I = sparse.eye(n, format="csc")
    e = np.ones(n)
    D2 = sparse.diags(
        [e[:-2], -2 * e[:-2], e[:-2]],
        [0, 1, 2],
        shape=(n - 2, n),
        format="csc"
    )

    trend = spsolve(I + (lam ** 2) * (D2.T @ D2), y)
    return y - trend


def interpolate_rr(rr, fs=FS_INTERP, apply_lambda=False, lam=500):
    """
    Interpolación cúbica a 4 Hz, más parecida a Kubios que np.interp.
    Si apply_lambda=True, aplica Smoothness Priors sobre la serie interpolada.
    """
    t = cumulative_time(rr)

    if len(t) < 5:
        return np.array([]), np.array([])

    t = t - t[0]
    x = rr.copy()

    # Evitar duplicados temporales
    keep = np.r_[True, np.diff(t) > 0]
    t = t[keep]
    x = x[keep]

    if len(t) < 5:
        return np.array([]), np.array([])

    ti = np.arange(0, t[-1], 1 / fs)

    if len(ti) < 5:
        return np.array([]), np.array([])

    cs = CubicSpline(t, x, bc_type="natural")
    xi = cs(ti)

    if apply_lambda:
        xi = smoothness_priors_detrend(xi, lam)

    return ti, xi


# ============================================================
# MÉTRICAS HRV
# ============================================================

def time_metrics(rr):
    rr_ms = rr * 1000
    diff = np.diff(rr_ms)

    mean_rr = np.mean(rr_ms)
    mean_hr = 60000 / mean_rr if mean_rr > 0 else np.nan
    sdnn = np.std(rr_ms, ddof=1) if len(rr_ms) > 1 else np.nan
    rmssd = np.sqrt(np.mean(diff ** 2)) if len(diff) > 0 else np.nan
    nn50 = np.sum(np.abs(diff) > 50)
    pnn50 = 100 * nn50 / len(diff) if len(diff) > 0 else np.nan

    sd1 = np.sqrt(0.5) * np.std(diff, ddof=1) if len(diff) > 1 else np.nan
    sd2 = np.sqrt(max(0, 2 * sdnn ** 2 - sd1 ** 2)) if np.isfinite(sdnn) and np.isfinite(sd1) else np.nan

    return {
        "MeanRR": mean_rr,
        "MeanHR": mean_hr,
        "SDNN": sdnn,
        "RMSSD": rmssd,
        "NN50": nn50,
        "pNN50": pnn50,
        "SD1": sd1,
        "SD2": sd2,
    }


def psd_metrics(rr):
    """
    PSD tipo Kubios:
    - Interpolación cúbica 4 Hz
    - Smoothness Priors λ=500 si APPLY_LAMBDA["PSD"] = True
    - Welch con ventana de 256 s y solapamiento del 50 %
    """
    ti, xi = interpolate_rr(
        rr,
        fs=FS_INTERP,
        apply_lambda=APPLY_LAMBDA["PSD"],
        lam=LAMBDA_DEFAULT
    )

    if len(xi) < 32:
        return {
            "VLF": np.nan,
            "LF": np.nan,
            "HF": np.nan,
            "TOTAL": np.nan,
            "LF_HF": np.nan
        }

    xi_ms = xi * 1000
    xi_ms = xi_ms - np.mean(xi_ms)

    nperseg = int(256 * FS_INTERP)  # 256 s × 4 Hz = 1024 muestras
    nperseg = min(nperseg, len(xi_ms))
    noverlap = int(0.5 * nperseg)

    f, pxx = signal.welch(
        xi_ms,
        fs=FS_INTERP,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density"
    )

    def bp(lo, hi):
        mask = (f >= lo) & (f < hi)
        return np.trapezoid(pxx[mask], f[mask]) if np.any(mask) else 0

    vlf = bp(0.0033, 0.04)
    lf = bp(0.04, 0.15)
    hf = bp(0.15, 0.40)
    total = vlf + lf + hf

    return {
        "VLF": vlf,
        "LF": lf,
        "HF": hf,
        "TOTAL": total,
        "LF_HF": lf / hf if hf > 0 else np.nan
    }


def _phi_apen(x, m, r):
    n = len(x)

    if n <= m + 1:
        return np.nan

    pats = np.array([x[i:i + m] for i in range(n - m + 1)])
    C = []

    for p in pats:
        dist = np.max(np.abs(pats - p), axis=1)
        C.append(np.mean(dist <= r))

    C = np.asarray(C)
    C = C[C > 0]

    return np.mean(np.log(C)) if len(C) > 0 else np.nan


def apen_calc(x, m=2, r_ratio=0.2):
    x = np.asarray(x, dtype=float)

    if APPLY_LAMBDA["ApEn"]:
        x = smoothness_priors_detrend(x, LAMBDA_DEFAULT)

    r = r_ratio * np.std(x, ddof=1)

    if not np.isfinite(r) or r == 0:
        return np.nan

    return _phi_apen(x, m, r) - _phi_apen(x, m + 1, r)


def sampen_base(x, m=2, r_ratio=0.2):
    x = np.asarray(x, dtype=float)
    n = len(x)

    if n <= m + 2:
        return np.nan

    r = r_ratio * np.std(x, ddof=1)

    if r == 0 or not np.isfinite(r):
        return np.nan

    def count(mm):
        pats = np.array([x[i:i + mm] for i in range(n - mm + 1)])
        c = 0

        for i in range(len(pats)):
            if len(pats[i + 1:]) == 0:
                continue
            dist = np.max(np.abs(pats[i + 1:] - pats[i]), axis=1)
            c += np.sum(dist <= r)

        return c

    B = count(m)
    A = count(m + 1)

    if A == 0 or B == 0:
        return np.nan

    return -np.log(A / B)


def sampen_calc(x, m=2, r_ratio=0.2):
    x = np.asarray(x, dtype=float)

    if APPLY_LAMBDA["SampEn"]:
        x = smoothness_priors_detrend(x, LAMBDA_DEFAULT)

    return sampen_base(x, m, r_ratio)


def mse_calc(x, max_scale=20):
    x = np.asarray(x, dtype=float)

    if APPLY_LAMBDA["MSE"]:
        x = smoothness_priors_detrend(x, LAMBDA_DEFAULT)

    out = {}

    for scale in range(1, max_scale + 1):
        n = len(x) // scale

        if n < 10:
            out[f"MSE{scale}"] = np.nan
        else:
            cg = x[:n * scale].reshape(n, scale).mean(axis=1)
            out[f"MSE{scale}"] = sampen_base(cg)

    return out


def dfa_calc(x):
    x = np.asarray(x, dtype=float)
    n = len(x)

    if n < 50:
        return np.nan, np.nan

    y = np.cumsum(x - np.mean(x))
    scales = np.unique(
        np.floor(
            np.logspace(np.log10(4), np.log10(max(5, n // 4)), 18)
        ).astype(int)
    )

    ss = []
    ff = []

    for s in scales:
        if s < 4 or n // s < 2:
            continue

        rms = []

        for i in range(n // s):
            seg = y[i * s:(i + 1) * s]
            t = np.arange(s)
            co = np.polyfit(t, seg, 1)
            rms.append(np.sqrt(np.mean((seg - np.polyval(co, t)) ** 2)))

        val = np.sqrt(np.mean(np.asarray(rms) ** 2))

        if val > 0:
            ss.append(s)
            ff.append(val)

    ss = np.asarray(ss)
    ff = np.asarray(ff)

    if len(ss) < 4:
        return np.nan, np.nan

    m1 = (ss >= 4) & (ss <= 16)
    m2 = ss > 16

    a1 = np.polyfit(np.log(ss[m1]), np.log(ff[m1]), 1)[0] if np.sum(m1) >= 2 else np.nan
    a2 = np.polyfit(np.log(ss[m2]), np.log(ff[m2]), 1)[0] if np.sum(m2) >= 2 else np.nan

    return a1, a2


def d2_calc(x, emb_dim=2, tau=1):
    x = np.asarray(x, float)
    n = len(x) - (emb_dim - 1) * tau

    if n < 30:
        return np.nan

    X = np.array([x[i:i + emb_dim * tau:tau] for i in range(n)])
    d = pdist(X)
    d = d[d > 0]

    if len(d) < 20:
        return np.nan

    radii = np.logspace(
        np.log10(np.percentile(d, 5)),
        np.log10(np.percentile(d, 60)),
        20
    )

    C = np.array([np.mean(d < r) for r in radii])
    mask = (C > 0) & (C < 1)

    if np.sum(mask) < 5:
        return np.nan

    return np.polyfit(np.log(radii[mask]), np.log(C[mask]), 1)[0]


def rqa_calc(x, emb_dim=10, tau=1, l_min=2):
    """
    RQA aproximada a configuración Kubios:
    m=10, tau=1, r=sqrt(m)*SD.
    """
    x = np.asarray(x, dtype=float)
    n = len(x) - (emb_dim - 1) * tau

    if n < 20:
        return {
            "REC": np.nan,
            "DET": np.nan,
            "Lmean": np.nan,
            "Lmax": np.nan,
            "ShanEn": np.nan
        }

    X = np.array([x[i:i + emb_dim * tau:tau] for i in range(n)])
    D = squareform(pdist(X))

    radius = np.sqrt(emb_dim) * np.std(x, ddof=1)

    R = (D <= radius).astype(int)
    np.fill_diagonal(R, 0)

    rec = 100 * R.sum() / (n * n - n)

    lens = []

    for k in range(-n + 1, n):
        diag = np.diag(R, k=k)
        c = 0

        for val in diag:
            if val == 1:
                c += 1
            else:
                if c >= l_min:
                    lens.append(c)
                c = 0

        if c >= l_min:
            lens.append(c)

    if len(lens) == 0:
        return {
            "REC": rec,
            "DET": 0,
            "Lmean": 0,
            "Lmax": 0,
            "ShanEn": 0
        }

    lens = np.asarray(lens)
    det = 100 * lens.sum() / R.sum() if R.sum() > 0 else 0

    vals, counts = np.unique(lens, return_counts=True)
    p = counts / counts.sum()

    return {
        "REC": rec,
        "DET": det,
        "Lmean": np.mean(lens),
        "Lmax": np.max(lens),
        "ShanEn": -np.sum(p * np.log(p))
    }


# ============================================================
# HVG Y TEORÍA DE GRAFOS
# ============================================================

def hvg_graph(x):
    x = np.asarray(x, float)
    n = len(x)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n - 1):
        G.add_edge(i, i + 1)

        for j in range(i + 2, n):
            if np.max(x[i + 1:j]) < min(x[i], x[j]):
                G.add_edge(i, j)

    return G


def hvg_lambda(G):
    deg = np.array([d for _, d in G.degree()])
    vals, counts = np.unique(deg, return_counts=True)
    p = counts / counts.sum()
    mask = (vals > 1) & (p > 0)

    if np.sum(mask) < 2:
        return np.nan

    return -np.polyfit(vals[mask], np.log(p[mask]), 1)[0]


def classify_graph_structure(G):
    """
    Clasificación orientativa del HVG:
    - Small World
    - Scale-Free / libre de escala
    - Mixta
    - Intermedia / no concluyente
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if n < 20 or m == 0:
        return {
            "Graph_type": "No evaluable",
            "Small_world_index": np.nan,
            "Scale_free_score": np.nan,
            "Graph_interpretation": "Grafo demasiado pequeño para clasificar."
        }

    degrees = np.array([d for _, d in G.degree()], dtype=float)
    k_mean = np.mean(degrees)
    k_max = np.max(degrees)
    clustering = nx.average_clustering(G)

    if nx.is_connected(G):
        path_length = nx.average_shortest_path_length(G)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        subG = G.subgraph(largest_cc)
        path_length = nx.average_shortest_path_length(subG)

    try:
        G_rand = nx.gnm_random_graph(n, m, seed=42)
        c_rand = nx.average_clustering(G_rand)

        if nx.is_connected(G_rand):
            l_rand = nx.average_shortest_path_length(G_rand)
        else:
            largest_cc_rand = max(nx.connected_components(G_rand), key=len)
            subG_rand = G_rand.subgraph(largest_cc_rand)
            l_rand = nx.average_shortest_path_length(subG_rand)

        if c_rand > 0 and l_rand > 0 and path_length > 0:
            small_world_index = (clustering / c_rand) / (path_length / l_rand)
        else:
            small_world_index = np.nan

    except Exception:
        c_rand = np.nan
        small_world_index = np.nan

    hub_ratio = k_max / k_mean if k_mean > 0 else np.nan
    hubs = np.sum(degrees >= np.percentile(degrees, 90))
    hub_fraction = hubs / n if n > 0 else np.nan
    scale_free_score = hub_ratio

    is_small_world = (
        np.isfinite(small_world_index)
        and small_world_index > 1.2
        and np.isfinite(c_rand)
        and clustering > c_rand
    )

    is_scale_free = (
        np.isfinite(scale_free_score)
        and scale_free_score > 3.0
        and np.isfinite(hub_fraction)
        and hub_fraction < 0.20
    )

    if is_small_world and is_scale_free:
        graph_type = "Mixta: Small World con hubs"
        interpretation = (
            "Agrupamiento local y caminos relativamente cortos, "
            "con algunos nodos muy conectados."
        )
    elif is_small_world:
        graph_type = "Small World"
        interpretation = (
            "Clustering alto y caminos relativamente cortos. "
            "Organización local con comunicación global eficiente."
        )
    elif is_scale_free:
        graph_type = "Scale-Free / libre de escala"
        interpretation = (
            "Pocos hubs muy conectados y muchos nodos pequeños. "
            "Dependencia de nodos centrales."
        )
    else:
        graph_type = "Intermedia / no concluyente"
        interpretation = (
            "No cumple claramente criterios de Small World ni de Scale-Free."
        )

    return {
        "Graph_type": graph_type,
        "Small_world_index": small_world_index,
        "Scale_free_score": scale_free_score,
        "Graph_interpretation": interpretation
    }


def hvg_metrics(rr):
    G = hvg_graph(rr)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    deg = np.array([d for _, d in G.degree()])

    out = {
        "HVG_nodes": n,
        "HVG_edges": m,
        "HVG_degree_mean": 2 * m / n if n else np.nan,
        "HVG_degree_max": np.max(deg) if len(deg) else np.nan,
        "HVG_hubs_p90": int(np.sum(deg >= np.percentile(deg, 90))) if len(deg) else 0,
        "HVG_clustering": nx.average_clustering(G) if n else np.nan,
        "HVG_density": nx.density(G) if n else np.nan,
        "HVG_lambda": hvg_lambda(G)
    }

    if n > 1 and nx.is_connected(G):
        out["HVG_path_length"] = nx.average_shortest_path_length(G)
        out["HVG_diameter"] = nx.diameter(G)
    else:
        out["HVG_path_length"] = np.nan
        out["HVG_diameter"] = np.nan

    out.update(classify_graph_structure(G))

    return out, G


def calculate_all(rr):
    rr_ms = rr * 1000
    out = {}

    out.update(time_metrics(rr))
    out.update(psd_metrics(rr))

    a1, a2 = dfa_calc(rr_ms)
    out["DFA_alpha1"] = a1
    out["DFA_alpha2"] = a2

    out["ApEn"] = apen_calc(rr_ms)
    out["SampEn"] = sampen_calc(rr_ms)
    out["D2"] = d2_calc(rr_ms)

    out.update(rqa_calc(rr_ms))
    out.update(mse_calc(rr_ms, 20))

    hvg_out, G = hvg_metrics(rr)
    out.update(hvg_out)

    return out, G


# ============================================================
# GRÁFICOS
# ============================================================

def smooth_line(x, y, points=100):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(y)

    if np.sum(mask) < 3:
        return x[mask], y[mask]

    xs = np.linspace(x[mask].min(), x[mask].max(), points)
    ys = make_interp_spline(x[mask], y[mask], k=2)(xs)

    return xs, ys


def plot_6_panel(metrics_df, out_path):
    phases = list(metrics_df.index)
    x = np.arange(len(phases))

    panels = [
        ("1) RMSSD, SDNN, pNN50", ["RMSSD", "SDNN", "pNN50"]),
        ("2) VLF, LF, HF, TOTAL", ["VLF", "LF", "HF", "TOTAL"]),
        ("3) SD1, SD2", ["SD1", "SD2"]),
        ("4) DFA α1, α2, D2, ApEn, SampEn", ["DFA_alpha1", "DFA_alpha2", "D2", "ApEn", "SampEn"]),
        ("5) Recurrence Plot", ["Lmean", "Lmax", "REC", "DET", "ShanEn"]),
        ("6) MSE 1-20", [f"MSE{i}" for i in range(1, 21)]),
    ]

    palette = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)

    all_vars = []

    for _, vv in panels:
        for v in vv:
            if v not in all_vars:
                all_vars.append(v)

    color_map = {v: palette[i % len(palette)] for i, v in enumerate(all_vars)}

    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    axes = axes.flatten()

    for ax, (title, vars_) in zip(axes, panels):
        vars_ = [v for v in vars_ if v in metrics_df.columns]
        width = min(0.8 / max(1, len(vars_)), 0.18)
        offsets = (np.arange(len(vars_)) - (len(vars_) - 1) / 2) * width

        for i, v in enumerate(vars_):
            y = metrics_df[v].astype(float).values
            c = color_map[v]

            ax.bar(x + offsets[i], y, width=width, alpha=0.45, label=v, color=c, edgecolor=c)

            xs, ys = smooth_line(x, y)
            ax.plot(xs, ys, linewidth=2.0, color=c)
            ax.scatter(x, y, s=28, color=c, edgecolor="black", linewidth=0.4, zorder=3)

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(phases)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7)

    fig.suptitle("VRC: barras verticales + tendencia suavizada", fontsize=18)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_single_phase(metrics_df, phase, out_path):
    if phase not in metrics_df.index:
        raise ValueError(f"Fase no encontrada: {phase}")

    panels = [
        ("1) Tiempo", ["RMSSD", "SDNN", "pNN50"]),
        ("2) Frecuencia", ["VLF", "LF", "HF", "TOTAL"]),
        ("3) Poincaré", ["SD1", "SD2"]),
        ("4) Complejidad", ["DFA_alpha1", "DFA_alpha2", "D2", "ApEn", "SampEn"]),
        ("5) Recurrence Plot", ["Lmean", "Lmax", "REC", "DET", "ShanEn"]),
        ("6) MSE 1-20", [f"MSE{i}" for i in range(1, 21)]),
    ]

    palette = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)

    all_vars = []

    for _, vv in panels:
        for v in vv:
            if v not in all_vars:
                all_vars.append(v)

    color_map = {v: palette[i % len(palette)] for i, v in enumerate(all_vars)}

    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    axes = axes.flatten()

    for ax, (title, vars_) in zip(axes, panels):
        vars_ = [v for v in vars_ if v in metrics_df.columns]
        vals = [
            float(metrics_df.loc[phase, v]) if np.isfinite(metrics_df.loc[phase, v]) else np.nan
            for v in vars_
        ]

        x = np.arange(len(vars_))
        colors = [color_map[v] for v in vars_]

        ax.bar(x, vals, color=colors, alpha=0.65, edgecolor=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(vars_, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{title} — {phase}")
        ax.grid(axis="y", alpha=0.25)

        for xi, yi in zip(x, vals):
            if np.isfinite(yi):
                ax.text(xi, yi, f"{yi:.2f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle(f"VRC: parámetros de la fase {phase}", fontsize=18)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_poincare(rr_segments, out_path):
    valid_segments = {
        phase: np.asarray(rr, dtype=float)
        for phase, rr in rr_segments.items()
        if rr is not None and len(rr) >= 3
    }

    if len(valid_segments) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No hay segmentos válidos para Poincaré", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    phases = list(valid_segments.keys())
    n = len(phases)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))

    if n == 1:
        axes = [axes]

    for ax, phase in zip(axes, phases):
        rr = valid_segments[phase] * 1000.0

        x = rr[:-1]
        y = rr[1:]

        diff = np.diff(rr)
        sdnn = np.std(rr, ddof=1) if len(rr) > 1 else np.nan
        sd1 = np.sqrt(0.5) * np.std(diff, ddof=1) if len(diff) > 1 else np.nan
        sd2 = np.sqrt(max(0, 2 * sdnn ** 2 - sd1 ** 2)) if np.isfinite(sdnn) and np.isfinite(sd1) else np.nan

        ax.scatter(x, y, s=12, alpha=0.6)

        lim_min = min(np.min(x), np.min(y))
        lim_max = max(np.max(x), np.max(y))
        margin = (lim_max - lim_min) * 0.08 if lim_max > lim_min else 10

        ax.plot(
            [lim_min - margin, lim_max + margin],
            [lim_min - margin, lim_max + margin],
            linestyle="--",
            linewidth=1
        )

        ax.set_xlim(lim_min - margin, lim_max + margin)
        ax.set_ylim(lim_min - margin, lim_max + margin)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)

        ax.set_title(f"{phase}\nSD1={sd1:.2f} ms | SD2={sd2:.2f} ms")
        ax.set_xlabel("RR(n) ms")
        ax.set_ylabel("RR(n+1) ms")

    fig.suptitle("Diagrama de Poincaré por fase", fontsize=18)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def domain_values(metrics_df, method="median"):
    out = {}
    base = metrics_df.iloc[0]

    for dom, vars_ in DOMAIN_GROUPS.items():
        vals_phase = []

        for ph in metrics_df.index:
            vals = []

            for v in vars_:
                if v not in metrics_df.columns:
                    continue

                b = base[v]
                x = metrics_df.loc[ph, v]

                if not np.isfinite(b) or b == 0 or not np.isfinite(x):
                    continue

                vals.append(100 * x / b)

            if len(vals) == 0:
                vals_phase.append(np.nan)
            else:
                vals_phase.append(np.nanmedian(vals) if method == "median" else np.nanmean(vals))

        out[dom] = vals_phase

    return pd.DataFrame(out, index=metrics_df.index)


def plot_domains(metrics_df, out_path, method="median"):
    dom_df = domain_values(metrics_df, method)
    phases = list(dom_df.index)
    x = np.arange(len(phases))

    fig, ax = plt.subplots(figsize=(13, 7))

    for col in dom_df.columns:
        y = dom_df[col].values
        xs, ys = smooth_line(x, y)

        ax.plot(xs, ys, linewidth=2.8, label=col)
        ax.scatter(x, y, s=70)

        for xi, yi in zip(x[1:], y[1:]):
            if np.isfinite(yi):
                ax.text(xi, yi + 3, f"{yi:.1f}", ha="center", fontsize=10)

    ax.axhline(100, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.set_ylabel("Índice normalizado (%)")
    ax.set_title(f"Respuesta autonómica y dinámica cardiovascular\nBasal = 100%, cálculo por {method}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    note = (
        "Amplitud: SDNN, SD2, Total Power\n"
        "Vagal: RMSSD, SD1, HF, pNN50*\n"
        "Complejidad: DFA α1, DFA α2, ApEn, SampEn, D2, ShanEn\n"
        "Recurrencia: REC, DET, Lmean, Lmax\n"
        "*Si basal=0 se excluye del cálculo del dominio."
    )

    fig.text(
        0.02,
        0.01,
        note,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="gray")
    )

    plt.subplots_adjust(bottom=0.28)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return dom_df


def plot_hvg_all(graphs, out_path):
    n = len(graphs)

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))

    if n == 1:
        axes = [axes]

    for ax, (name, G) in zip(axes, graphs.items()):
        pos = nx.spring_layout(G, seed=42, k=0.12, iterations=80)
        deg = dict(G.degree())
        sizes = [8 + deg[node] * 8 for node in G.nodes()]

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.18, width=0.5)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, alpha=0.85)

        ax.set_title(f"{name}\nN={G.number_of_nodes()} | E={G.number_of_edges()}")
        ax.axis("off")

    fig.suptitle("Horizontal Visibility Graphs reales por fase", fontsize=18)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# APP v3.7 ROBUSTA: múltiples registros + comparación real
# ============================================================

st.title("VRC / HRV RRi Analyzer Pro v3.8")
st.caption(
    "Segmentación libre, análisis HRV/HVG/Poincaré y comparación entre varios registros. "
    "Protocolo: Basal + E1-E6 + R1-R3."
)

MIN_RR_DEFAULT = 30


def default_windows_for_duration(t_max):
    """Ventanas iniciales distribuidas dentro del registro, sin imponer 5 minutos."""
    t_max = float(max(t_max, 60.0))
    # Si el registro es largo, usa una propuesta razonable de 10 ventanas.
    # Si es corto, reparte todo el registro en 10 ventanas.
    if t_max >= 3900:  # >=65 min
        vals = {
            "Basal": [300, 600],
            "E1": [1200, 1500],
            "E2": [1500, 1800],
            "E3": [1800, 2100],
            "E4": [2100, 2400],
            "E5": [2400, 2700],
            "E6": [2700, 3000],
            "R1": [3000, 3300],
            "R2": [3300, 3600],
            "R3": [3600, 3900],
        }
    else:
        edges = np.linspace(0, t_max, len(PHASES) + 1)
        vals = {ph: [float(edges[i]), float(edges[i + 1])] for i, ph in enumerate(PHASES)}
    return vals


def ensure_windows(t_max):
    """Asegura que existan ventanas para todas las fases y que estén dentro del registro principal."""
    if "windows" not in st.session_state:
        st.session_state.windows = default_windows_for_duration(t_max)
    # Convertir versión antigua si existe
    old = st.session_state.windows.copy()
    if "Ejercicio" in old or "Recuperación" in old:
        new = {"Basal": old.get("Basal", [0, min(300, t_max)])}
        if "Ejercicio" in old:
            s, e = old["Ejercicio"]
            edges = np.linspace(s, e, 7)
            for i in range(1, 7):
                new[f"E{i}"] = [float(edges[i-1]), float(edges[i])]
        if "Recuperación" in old:
            s, e = old["Recuperación"]
            edges = np.linspace(s, e, 4)
            for i in range(1, 4):
                new[f"R{i}"] = [float(edges[i-1]), float(edges[i])]
        st.session_state.windows = new
    defaults = default_windows_for_duration(t_max)
    for ph in PHASES:
        st.session_state.windows.setdefault(ph, defaults[ph])
    # Limitar a rango real sólo para visualización; se permite editar después.
    for ph in PHASES:
        s, e = st.session_state.windows[ph]
        s = min(max(0.0, float(s)), float(t_max))
        e = min(max(0.0, float(e)), float(t_max))
        if e <= s:
            e = min(float(t_max), s + max(10.0, float(t_max) / 100.0))
        st.session_state.windows[ph] = [s, e]


def read_all_uploaded(uploaded_files, artifact_level):
    records = {}
    errors = []
    for uf in uploaded_files:
        try:
            rr_raw_i = read_rri_file(uf)
            rr_i, artifact_mask_i, artifact_info_i = correct_artifacts_kubios_like(
                rr_raw_i, level=artifact_level
            )
            name = sanitize_record_name(uf.name)
            base = name
            k = 2
            while name in records:
                name = f"{base}_{k}"
                k += 1
            records[name] = {
                "filename": uf.name,
                "rr_raw": rr_raw_i,
                "rr": rr_i,
                "artifact_mask": artifact_mask_i,
                "artifact_info": artifact_info_i,
                "t": cumulative_time(rr_i),
                "t_raw": cumulative_time(rr_raw_i),
            }
        except Exception as exc:
            errors.append(f"{uf.name}: {exc}")
    return records, errors


def compute_all_records(record_data, windows, min_rr=30):
    results = {}
    segments_all = {}
    valid_all = {}
    graphs_all = {}
    for rec, data in record_data.items():
        df, segs, valid, graphs = calculate_metrics_for_windows(data["rr"], windows, min_rr=min_rr)
        results[rec] = df
        segments_all[rec] = segs
        valid_all[rec] = valid
        graphs_all[rec] = graphs
    return results, segments_all, valid_all, graphs_all


def windows_table(windows):
    return pd.DataFrame([
        {
            "Fase": ph,
            "Inicio": sec_to_hms(windows[ph][0]),
            "Fin": sec_to_hms(windows[ph][1]),
            "Inicio_min": round(windows[ph][0] / 60, 2),
            "Fin_min": round(windows[ph][1] / 60, 2),
            "Duración": sec_to_hms(windows[ph][1] - windows[ph][0]),
            "Duración_min": round((windows[ph][1] - windows[ph][0]) / 60, 2),
        }
        for ph in PHASES
    ])


def plot_rr_segmentation(record_data, selected_record, windows, show_mode="Principal"):
    fig = go.Figure()
    if show_mode == "Todos superpuestos":
        for rec, data in record_data.items():
            t_i = data["t"] / 60.0
            rr_i = data["rr"] * 1000.0
            opacity = 0.95 if rec == selected_record else 0.35
            fig.add_trace(go.Scatter(
                x=t_i, y=rr_i, mode="lines", name=rec,
                line=dict(width=1), opacity=opacity
            ))
    else:
        data = record_data[selected_record]
        fig.add_trace(go.Scatter(
            x=data["t"] / 60.0,
            y=data["rr"] * 1000.0,
            mode="lines",
            name=selected_record,
        ))

    for name, (s, e) in windows.items():
        group = PHASE_GROUP.get(name, name)
        fig.add_vrect(
            x0=s / 60.0,
            x1=e / 60.0,
            fillcolor=PHASE_COLORS.get(group, "rgba(180,180,180,0.15)"),
            line_width=0,
            annotation_text=name,
            annotation_position="top left",
        )

    fig.update_layout(
        height=520,
        xaxis_title="Tiempo acumulado (min)",
        yaxis_title="RRi (ms)",
        hovermode="x unified",
    )
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def plot_compare_plotly(pivot_df, variable):
    fig = go.Figure()
    for rec in pivot_df.columns:
        fig.add_trace(go.Scatter(
            x=list(pivot_df.index),
            y=pivot_df[rec],
            mode="lines+markers",
            name=str(rec),
        ))
    fig.update_layout(
        height=520,
        title=f"Comparativa entre registros: {variable}",
        xaxis_title="Fase",
        yaxis_title=variable,
        hovermode="x unified",
    )
    return fig


def plot_phase_rr_overlay(record_data, segments_all, phase):
    fig = go.Figure()
    for rec in record_data.keys():
        seg = segments_all.get(rec, {}).get(phase, np.array([]))
        if seg is None or len(seg) == 0:
            continue
        t_seg = np.cumsum(seg) / 60.0
        fig.add_trace(go.Scatter(
            x=t_seg,
            y=seg * 1000.0,
            mode="lines",
            name=rec,
        ))
    fig.update_layout(
        height=430,
        title=f"RRi superpuesto en fase {phase}",
        xaxis_title="Tiempo dentro de la fase (min)",
        yaxis_title="RRi (ms)",
        hovermode="x unified",
    )
    return fig


# ---------------- Sidebar ----------------

uploaded_files = st.sidebar.file_uploader(
    "Sube uno o varios CSV/TXT con RRi",
    type=["csv", "txt"],
    accept_multiple_files=True,
)

method = st.sidebar.selectbox("Dominios", ["median", "mean"], index=0)
artifact_level = st.sidebar.selectbox(
    "Artifact correction",
    ["none", "very low", "low", "medium", "strong", "very strong"],
    index=0,
)
min_rr = st.sidebar.number_input(
    "Mínimo RRi por ventana",
    min_value=5,
    max_value=300,
    value=MIN_RR_DEFAULT,
    step=5,
)

if not uploaded_files:
    st.info("Sube uno o varios archivos RRi para empezar.")
    st.stop()

record_data, load_errors = read_all_uploaded(uploaded_files, artifact_level)
if load_errors:
    st.error("No se pudieron leer algunos archivos:\n\n" + "\n".join(load_errors))
if not record_data:
    st.stop()

record_names = list(record_data.keys())
selected_record = st.sidebar.selectbox("Registro principal", record_names, index=0)
rr = record_data[selected_record]["rr"]
rr_raw = record_data[selected_record]["rr_raw"]
artifact_mask = record_data[selected_record]["artifact_mask"]
artifact_info = record_data[selected_record]["artifact_info"]
t = record_data[selected_record]["t"]
t_raw = record_data[selected_record]["t_raw"]
t_max = float(t.max())

ensure_windows(t_max)

st.sidebar.success(f"{len(record_data)} registro(s) cargado(s)")
st.sidebar.info(f"Principal: {selected_record}\n{len(rr)} RRi | {t_max/60:.1f} min")
if artifact_level != "none":
    st.sidebar.warning(
        f"Artefactos corregidos en principal: {artifact_info['n_artifacts']} "
        f"({artifact_info['percent_artifacts']:.2f}%)"
    )

# Calcular todos los registros una vez por ejecución.
records_results, segments_all, valid_all, graphs_all = compute_all_records(
    record_data,
    st.session_state.windows,
    min_rr=int(min_rr),
)
metrics_df = records_results[selected_record]
segments = segments_all[selected_record]
valid_segments = valid_all[selected_record]
graphs = graphs_all[selected_record]

# Advertencia no bloqueante
bad = [ph for ph, ok in valid_segments.items() if not ok]
if bad:
    st.warning(
        "Algunas ventanas del registro principal no se calculan por tener pocos RRi: "
        + ", ".join(bad)
        + ". El resto sí se analiza. Puedes bajar el mínimo RRi en la barra lateral si lo necesitas."
    )

# ---------------- Tabs ----------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1) Segmentación",
    "2) Análisis principal",
    "3) Comparar registros",
    "4) Gráficas",
    "5) HVG",
    "6) Exportar",
])

with tab1:
    st.subheader("Segmentación visual")
    st.write(
        "Las ventanas son libres: puedes poner la duración que quieras. "
        "La misma segmentación temporal se aplicará a todos los registros para compararlos."
    )

    col_a, col_b = st.columns([1, 2])
    with col_a:
        show_mode = st.radio("Visualización RRi", ["Principal", "Todos superpuestos"], index=0)
    with col_b:
        if st.button("Reiniciar ventanas distribuidas en el registro principal"):
            st.session_state.windows = default_windows_for_duration(t_max)
            st.rerun()

    st.markdown("### Ajustar ventanas con deslizadores")
    max_min = float(t_max / 60.0)
    for fase in PHASES:
        ini_min = min(max(0.0, st.session_state.windows[fase][0] / 60.0), max_min)
        fin_min = min(max(0.0, st.session_state.windows[fase][1] / 60.0), max_min)
        if fin_min <= ini_min:
            fin_min = min(max_min, ini_min + 0.1)
        val = st.slider(
            f"{fase} inicio-fin (min)",
            min_value=0.0,
            max_value=max_min,
            value=(ini_min, fin_min),
            step=0.01,
            key=f"slider_{fase}",
        )
        st.session_state.windows[fase] = [val[0] * 60.0, val[1] * 60.0]

    st.markdown("### Editar ventanas manualmente")
    edit_cols = st.columns(5)
    edited = {}
    for idx, fase in enumerate(PHASES):
        with edit_cols[idx % 5]:
            st.markdown(f"**{fase}**")
            ini_txt = st.text_input(
                f"{fase} inicio",
                value=sec_to_hms(st.session_state.windows[fase][0]),
                key=f"{fase}_ini_txt",
            )
            fin_txt = st.text_input(
                f"{fase} fin",
                value=sec_to_hms(st.session_state.windows[fase][1]),
                key=f"{fase}_fin_txt",
            )
            edited[fase] = [ini_txt, fin_txt]

    if st.button("Aplicar ventanas escritas"):
        ok = True
        new_w = {}
        for fase, (ini_txt, fin_txt) in edited.items():
            try:
                s = hms_to_sec(ini_txt)
                e = hms_to_sec(fin_txt)
                if e <= s:
                    st.warning(f"{fase}: el final debe ser mayor que el inicio.")
                    ok = False
                else:
                    new_w[fase] = [s, e]
            except Exception:
                st.warning(f"{fase}: formato no válido. Usa HH:MM:SS, MM:SS o segundos.")
                ok = False
        if ok:
            st.session_state.windows.update(new_w)
            st.success("Ventanas actualizadas.")
            st.rerun()

    st.plotly_chart(
        plot_rr_segmentation(record_data, selected_record, st.session_state.windows, show_mode),
        use_container_width=True,
    )

    st.markdown("### Ventanas activas")
    st.dataframe(windows_table(st.session_state.windows), use_container_width=True)

    st.markdown("### Ventanas válidas por registro")
    valid_summary = pd.DataFrame(valid_all).T.reindex(columns=PHASES)
    st.dataframe(valid_summary, use_container_width=True)

with tab2:
    st.subheader(f"Análisis HRV del registro principal: {selected_record}")
    if metrics_df.empty:
        st.info("No hay ventanas válidas. Ajusta las ventanas o baja el mínimo de RRi por ventana.")
    else:
        st.markdown("### Parámetros lineales")
        linear_cols = ["MeanRR", "MeanHR", "SDNN", "RMSSD", "NN50", "pNN50", "SD1", "SD2"]
        st.dataframe(metrics_df[[c for c in linear_cols if c in metrics_df.columns]], use_container_width=True)

        st.markdown("### Parámetros frecuenciales")
        freq_cols = ["VLF", "LF", "HF", "TOTAL", "LF_HF"]
        st.dataframe(metrics_df[[c for c in freq_cols if c in metrics_df.columns]], use_container_width=True)

        st.markdown("### Parámetros no lineales / RQA / MSE")
        non_cols = [
            "DFA_alpha1", "DFA_alpha2", "ApEn", "SampEn", "D2",
            "REC", "DET", "Lmean", "Lmax", "ShanEn",
        ] + [f"MSE{i}" for i in range(1, 21)]
        st.dataframe(metrics_df[[c for c in non_cols if c in metrics_df.columns]], use_container_width=True)

with tab3:
    st.subheader("Comparar registros")
    if len(record_data) < 2:
        st.info("Sube dos o más archivos RRi para comparar registros.")
    else:
        long_df, pivot_tables = build_comparison_tables(records_results)
        if long_df.empty:
            st.info("No hay ventanas válidas suficientes para comparar.")
        else:
            st.markdown("### Selección comparativa")
            available_phases = [ph for ph in PHASES if ph in long_df["Fase"].unique()]
            selected_phases = st.multiselect(
                "Fases a comparar",
                available_phases,
                default=available_phases,
            )
            preferred = [
                "MeanHR", "MeanRR", "SDNN", "RMSSD", "pNN50", "SD1", "SD2",
                "VLF", "LF", "HF", "TOTAL", "LF_HF",
                "DFA_alpha1", "DFA_alpha2", "ApEn", "SampEn", "D2",
                "REC", "DET", "Lmean", "Lmax", "ShanEn",
                "HVG_edges", "HVG_degree_mean", "HVG_degree_max", "HVG_hubs_p90",
                "HVG_clustering", "HVG_lambda", "HVG_path_length", "HVG_diameter",
                "Small_world_index", "Scale_free_score",
            ]
            available_vars = [v for v in preferred if v in pivot_tables] + [
                v for v in pivot_tables.keys() if v not in preferred
            ]

            if not available_vars:
                st.warning("No hay variables numéricas comparables entre los registros con las ventanas actuales.")
                st.stop()

            variable = st.selectbox(
                "Variable principal",
                available_vars,
                index=0,
                key="compare_variable_v38"
            )

            # Protección frente a estados antiguos de Streamlit tras cambiar archivos o ventanas.
            if variable not in pivot_tables:
                variable = available_vars[0]

            pivot = pivot_tables.get(variable, pd.DataFrame())
            if pivot.empty:
                st.warning(f"No hay datos comparables para {variable} con las ventanas actuales.")
                st.stop()

            if selected_phases:
                pivot = pivot.loc[[ph for ph in selected_phases if ph in pivot.index]]

            st.markdown(f"### Tabla comparativa: {variable}")
            st.dataframe(pivot, use_container_width=True)
            st.plotly_chart(plot_compare_plotly(pivot, variable), use_container_width=True)

            phase_overlay = st.selectbox("Ver RRi superpuesto en una fase", selected_phases or available_phases, index=0)
            st.plotly_chart(plot_phase_rr_overlay(record_data, segments_all, phase_overlay), use_container_width=True)

            st.markdown("### Tabla larga completa")
            if selected_phases:
                st.dataframe(long_df[long_df["Fase"].isin(selected_phases)], use_container_width=True)
            else:
                st.dataframe(long_df, use_container_width=True)

            st.markdown("### Dominios normalizados por registro")
            tmp_domains_compare = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            dom_long = plot_compare_domains(records_results, method, tmp_domains_compare.name)
            st.image(tmp_domains_compare.name)
            if not dom_long.empty:
                if selected_phases:
                    st.dataframe(dom_long[dom_long["Fase"].isin(selected_phases)], use_container_width=True)
                else:
                    st.dataframe(dom_long, use_container_width=True)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                p_xlsx = tmpdir / "comparativa_registros_hrv.xlsx"
                p_csv = tmpdir / "comparativa_registros_hrv.csv"
                long_df.to_csv(p_csv, index=False)
                with pd.ExcelWriter(p_xlsx) as writer:
                    long_df.to_excel(writer, sheet_name="metricas_largas", index=False)
                    pd.DataFrame(valid_all).T.reindex(columns=PHASES).to_excel(writer, sheet_name="ventanas_validas")
                    for name, piv in list(pivot_tables.items())[:25]:
                        safe = re.sub(r"[^A-Za-z0-9_]+", "_", name)[:25]
                        piv.to_excel(writer, sheet_name=safe)
                st.download_button(
                    "Descargar Excel comparativa",
                    data=p_xlsx.read_bytes(),
                    file_name="comparativa_registros_hrv.xlsx",
                )

with tab4:
    st.subheader("Gráficas del registro principal")
    if metrics_df.empty:
        st.info("No hay ventanas válidas para graficar.")
    else:
        opciones = []
        if len(metrics_df.index) >= 2:
            opciones.append("Todas las fases válidas")
        for fase in PHASES:
            if fase in metrics_df.index:
                opciones.append(f"Sólo {fase}")
        modo_grafica = st.selectbox("Qué quieres graficar", opciones, index=0)

        if modo_grafica == "Todas las fases válidas":
            tmp1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp_poincare = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            plot_6_panel(metrics_df, tmp1.name)
            dom_df = plot_domains(metrics_df, tmp2.name, method=method)
            poincare_segments = {fase: segments[fase] for fase in metrics_df.index if fase in segments}
            plot_poincare(poincare_segments, tmp_poincare.name)
            st.markdown("### Parrilla 6 paneles")
            st.image(tmp1.name)
            st.markdown("### Dominios")
            st.image(tmp2.name)
            st.dataframe(dom_df, use_container_width=True)
            st.markdown("### Poincaré")
            st.image(tmp_poincare.name)
        else:
            phase = modo_grafica.replace("Sólo ", "")
            tmp_single = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp_poincare = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            plot_single_phase(metrics_df, phase, tmp_single.name)
            plot_poincare({phase: segments[phase]}, tmp_poincare.name)
            st.image(tmp_single.name)
            st.dataframe(metrics_df.loc[[phase]], use_container_width=True)
            st.markdown(f"### Poincaré — {phase}")
            st.image(tmp_poincare.name)

with tab5:
    st.subheader("Horizontal Visibility Graph del registro principal")
    if metrics_df.empty:
        st.info("No hay ventanas válidas.")
    else:
        hvg_cols = [c for c in metrics_df.columns if c.startswith("HVG_")]
        graph_cols = ["Graph_type", "Small_world_index", "Scale_free_score", "Graph_interpretation"]
        cols_to_show = hvg_cols + [c for c in graph_cols if c in metrics_df.columns]
        st.dataframe(metrics_df[cols_to_show], use_container_width=True)
        if graphs:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            plot_hvg_all(graphs, tmp.name)
            st.image(tmp.name)

with tab6:
    st.subheader("Exportar")
    if metrics_df.empty:
        st.info("No hay ventanas válidas para exportar.")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            p_metrics = tmpdir / "metricas_hrv_registro_principal.xlsx"
            p_csv = tmpdir / "metricas_hrv_registro_principal.csv"
            p_domains = tmpdir / "dominios_normalizados.csv"
            p_artifacts = tmpdir / "artifact_correction.csv"
            p_6 = tmpdir / "grafica_6_paneles.png"
            p_dom = tmpdir / "grafica_dominios.png"
            p_hvg = tmpdir / "hvg_grafos.png"
            p_poincare = tmpdir / "poincare_plot.png"
            p_zip = tmpdir / "resultados_vrc.zip"

            dom_df = plot_domains(metrics_df, p_dom, method=method)
            if len(metrics_df.index) >= 2:
                plot_6_panel(metrics_df, p_6)
            else:
                plot_single_phase(metrics_df, metrics_df.index[0], p_6)
            if graphs:
                plot_hvg_all(graphs, p_hvg)
            else:
                p_hvg.write_text("No hay grafos válidos", encoding="utf-8")
            poincare_segments = {fase: segments[fase] for fase in metrics_df.index if fase in segments}
            plot_poincare(poincare_segments, p_poincare)

            metrics_df.to_csv(p_csv)
            dom_df.to_csv(p_domains)
            artifact_df = pd.DataFrame({
                "RRi_original_s": rr_raw,
                "RRi_used_s": rr,
                "artifact_corrected": artifact_mask,
            })
            artifact_df.to_csv(p_artifacts, index=False)
            with pd.ExcelWriter(p_metrics) as writer:
                metrics_df.to_excel(writer, sheet_name="metricas")
                dom_df.to_excel(writer, sheet_name="dominios")
                windows_table(st.session_state.windows).to_excel(writer, sheet_name="ventanas", index=False)

            with zipfile.ZipFile(p_zip, "w", zipfile.ZIP_DEFLATED) as z:
                for p in [p_metrics, p_csv, p_domains, p_artifacts, p_6, p_dom, p_hvg, p_poincare]:
                    z.write(p, arcname=p.name)

            st.download_button(
                "Descargar ZIP completo",
                data=p_zip.read_bytes(),
                file_name="resultados_vrc.zip",
                mime="application/zip",
            )
            st.download_button(
                "Descargar Excel",
                data=p_metrics.read_bytes(),
                file_name="metricas_hrv_registro_principal.xlsx",
            )
