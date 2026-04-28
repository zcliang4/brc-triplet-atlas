"""
BrC-Triplet-Atlas  棕碳光敏性质预测平台
Bilingual Streamlit web app with aerosol photochemistry theme.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import base64
import joblib
import numpy as np
import pandas as pd
import requests
from openpyxl import Workbook  # noqa: F401
import streamlit as st
import xgboost as xgb  # noqa: F401
from plotly import express as px
from plotly import graph_objects as go
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from rdkit.Chem import (
    AllChem, Crippen, Descriptors, Fragments, GraphDescriptors,
    Lipinski, MolSurf, QED, rdMolDescriptors,
)
from rdkit.Chem.EState import EState_VSA
from sklearn.exceptions import InconsistentVersionWarning

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BrC-Triplet-Atlas | 棕碳光敏性质预测平台",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Path constants ────────────────────────────────────────────────────────────
# Models live in different subdirectories under the project root.
# On HuggingFace Spaces: app.py is at the Space root, so parent = root.
ROOT = Path(__file__).resolve().parent
PS_ML = ROOT / "PS_ML"

E0_MODEL   = PS_ML / "e0_model-added.model"
E0_SCALER  = PS_ML / "e0_scaler-added.pkl"
E0_FEAT    = PS_ML / "e0_model_features.json"

ET_MODEL   = PS_ML / "et_model-added.model"
ET_SCALER  = PS_ML / "et_scaler-added.pkl"
ET_FEAT    = PS_ML / "et_model_features.json"

OXI_MODEL  = ROOT / "oxi_model-added.model"
OXI_SCALER = ROOT / "oxi_scaler-added.pkl"
OXI_FEAT   = ROOT / "oxi_model_features.json"

PHI_MODEL  = PS_ML / "phi_model.model"
PHI_SCALER = PS_ML / "phi_model.pkl"
PHI_FEAT   = PS_ML / "phi_model_features.json"

G2_PIPELINE = ROOT / "final_3class_G2_model" / "final_g2_pipeline.pkl"

# ─── G2 Absorb predictor (inline, no importlib) ───────────────────────────────
import pickle as _pickle
from rdkit import DataStructs as _DataStructs

# ── Load all ML models once at module level (persists across reruns) ──────────
with warnings.catch_warnings():
    warnings.simplefilter("ignore", InconsistentVersionWarning)
    _model_e0, _scaler_e0 = joblib.load(E0_MODEL), joblib.load(E0_SCALER)
    _model_et, _scaler_et = joblib.load(ET_MODEL), joblib.load(ET_SCALER)
    _model_phi, _scaler_phi = joblib.load(PHI_MODEL), joblib.load(PHI_SCALER)
    _model_oxi, _scaler_oxi = joblib.load(OXI_MODEL), joblib.load(OXI_SCALER)


_G2_PIPE = None

def _load_g2():
    global _G2_PIPE
    if _G2_PIPE is None and G2_PIPELINE.exists():
        with open(G2_PIPELINE, "rb") as _f:
            _G2_PIPE = _pickle.load(_f)
    return _G2_PIPE

_G2_ALL_CHEM = [
    "MolWt","MolLogP","TPSA","NumHDonors","NumHAcceptors",
    "NumRotatableBonds","NumHeteroatoms","NumAromaticRings",
    "NumBridgeheadAtoms","BertzCT","Chi0","Chi0n","Chi0v",
    "Chi1","Chi1n","RingCount","MolMR","NumAliphaticRings",
    "NumSaturatedRings","NumAmideBonds","LabuteASA",
    "HallKierAlpha","Kappa1","Kappa2",
    "FusedRingCount","MethoxyCount","AldehydeKetoneQuinoneCount",
    "NitroGroupCount","PhenolicOHCount",
]

def _g2_desc(mol):
    from rdkit.Chem import Descriptors as _D, rdMolDescriptors as _rMD
    f = {}
    def g(n, v): f[n] = float(v) if v is not None else 0.0
    g("MolWt",             _D.MolWt(mol))
    g("MolLogP",           _D.MolLogP(mol))
    g("TPSA",              _D.TPSA(mol))
    g("NumHDonors",        _D.NumHDonors(mol))
    g("NumHAcceptors",     _D.NumHAcceptors(mol))
    try: g("NumRotatableBonds", _rMD.CalcNumRotatableBonds(mol))
    except: f["NumRotatableBonds"] = 0
    g("NumHeteroatoms",    _D.NumHeteroatoms(mol))
    g("NumAromaticRings",  _D.NumAromaticRings(mol))
    g("NumBridgeheadAtoms",_rMD.CalcNumBridgeheadAtoms(mol))
    g("BertzCT",           _D.BertzCT(mol))
    g("Chi0",              _D.Chi0(mol))
    g("Chi0n",             _D.Chi0n(mol))
    g("Chi0v",             _D.Chi0v(mol))
    g("Chi1",              _D.Chi1(mol))
    g("Chi1n",             _D.Chi1n(mol))
    g("RingCount",         mol.GetRingInfo().NumRings())
    g("MolMR",             _D.MolMR(mol))
    g("NumAliphaticRings", _D.NumAliphaticRings(mol))
    g("NumSaturatedRings", _D.NumSaturatedRings(mol))
    g("NumAmideBonds",     _rMD.CalcNumAmideBonds(mol))
    g("LabuteASA",         _D.LabuteASA(mol))
    g("HallKierAlpha",     _D.HallKierAlpha(mol))
    g("Kappa1",            _D.Kappa1(mol))
    g("Kappa2",            _D.Kappa2(mol))
    try: g("FusedRingCount", _rMD.CalcNumFusedRings(mol))
    except: f["FusedRingCount"] = 0
    ms = Chem.MolFromSmarts("cO[C;H1]")
    g("MethoxyCount", len(mol.GetSubstructMatches(ms)) if ms else 0)
    ald  = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3H1](=O)")))
    ket  = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[!H1]"))) // 2
    quin = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[cR1](=O)[cR1](=O)")))
    g("AldehydeKetoneQuinoneCount", ald + ket + quin)
    ns = Chem.MolFromSmarts("[N+](=O)[O-]")
    g("NitroGroupCount", len(mol.GetSubstructMatches(ns)) if ns else 0)
    phenol = sum(
        1 for a in mol.GetAtoms()
        if a.GetAtomicNum() == 8 and a.GetTotalNumHs() >= 1
        and any(nb.GetAtomicNum() == 6 and nb.GetIsAromatic()
                for nb in a.GetNeighbors())
    )
    g("PhenolicOHCount", phenol)
    return f

def _g2_predict(smiles: str) -> str:
    """Return 'low'/'medium'/'high' or raise on error."""
    pipe = _load_g2()
    if pipe is None:
        raise RuntimeError("G2 pipeline not found")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    desc = _g2_desc(mol)
    chem_idx = pipe["chem_idx"]
    fp_idx   = pipe["fp_idx"]
    chem_vals = np.array([[desc[k] for k in _G2_ALL_CHEM]], dtype=float)
    chem_vals = np.nan_to_num(chem_vals)
    from rdkit.Chem import AllChem as _AC
    fp_raw = _AC.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fp_arr = np.zeros((1, 1024), dtype=int)
    for i in fp_raw.GetOnBits(): fp_arr[0, i] = 1
    d = chem_vals[:, chem_idx]
    f = fp_arr[:, fp_idx] if fp_idx else np.zeros((1, 0))
    X = np.hstack([d, f])
    X_s = pipe["scaler"].transform(X)
    pred = pipe["classifier"].predict(X_s)[0]
    return pipe["class_names"][pred]

# Load feature column names from JSON metadata
E0_FEATURES = json.loads(E0_FEAT.read_text(encoding="utf-8"))["feature_columns"]
ET_FEATURES  = json.loads(ET_FEAT.read_text(encoding="utf-8"))["feature_columns"]
OXI_FEATURES = json.loads(OXI_FEAT.read_text(encoding="utf-8"))["feature_columns"]
PHI_FEATURES = json.loads(PHI_FEAT.read_text(encoding="utf-8"))["feature_columns"]

MORGAN_R2 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
MORGAN_R3 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
CAS_PATTERN = re.compile(r"^\d{2,7}-\d{2}-\d$")

PHI_CLASS_LABELS = {0: "medium", 1: "low", 2: "high"}
PHI_COLORS       = {"low": "#ff7f0e", "medium": "#1f77b4", "high": "#2ca02c", "unknown": "#7f7f7f"}
ABSORB_COLORS    = {"low": "#1f77b4", "medium": "#ff7f0e", "high": "#2ca02c"}

FARADAY = 96.485  # kJ mol⁻¹ V⁻¹

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _maccs(mol, bit_id):
    return int(MACCSkeys.GenMACCSKeys(mol).GetBit(bit_id))

def _ecfp(mol, bit_id, gen):
    return int(gen.GetFingerprint(mol).GetBit(bit_id))

def atom_count(mol, num):
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == num)

def bond_count(mol, btype):
    return sum(1 for b in mol.GetBonds() if b.GetBondType() == btype)

def compute_features(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    ha = max(Descriptors.HeavyAtomCount(mol), 1)
    ta = max(mol.GetNumAtoms(), 1)
    tb = max(mol.GetNumBonds(), 1)
    aro = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    con = sum(1 for b in mol.GetBonds() if b.GetIsConjugated())

    f = {
        "NumHeteroatoms": Lipinski.NumHeteroatoms(mol),
        "MaxPartialCharge": Descriptors.MaxPartialCharge(mol),
        "NumAliphaticCarbocycles": Lipinski.NumAliphaticCarbocycles(mol),
        "Kappa3": GraphDescriptors.Kappa3(mol),
        "Kappa2": GraphDescriptors.Kappa2(mol),
        "Kappa1": GraphDescriptors.Kappa1(mol),
        "HallKierAlpha": GraphDescriptors.HallKierAlpha(mol),
        "qed": QED.qed(mol),
        "MinEStateIndex": Descriptors.MinEStateIndex(mol),
        "MaxAbsPartialCharge": Descriptors.MaxAbsPartialCharge(mol),
        "MinAbsPartialCharge": Descriptors.MinAbsPartialCharge(mol),
        "MinPartialCharge": Descriptors.MinPartialCharge(mol),
        "MaxAbsEStateIndex": Descriptors.MaxAbsEStateIndex(mol),
        "MinAbsEStateIndex": Descriptors.MinAbsEStateIndex(mol),
        "SPS": Descriptors.SPS(mol),
        "Chi1": GraphDescriptors.Chi1(mol),
        "Chi1n": GraphDescriptors.Chi1n(mol),
        "Chi2v": GraphDescriptors.Chi2v(mol),
        "Chi3n": GraphDescriptors.Chi3n(mol),
        "Chi3v": GraphDescriptors.Chi3v(mol),
        "Chi4n": GraphDescriptors.Chi4n(mol),
        "Chi4v": GraphDescriptors.Chi4v(mol),
        "BertzCT": GraphDescriptors.BertzCT(mol),
        "AvgIpc": GraphDescriptors.AvgIpc(mol),
        "MolMR": Crippen.MolMR(mol),
        "Fraction Csp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "TPSA_x": rdMolDescriptors.CalcTPSA(mol),
        "FpDensityMorgan3": Descriptors.FpDensityMorgan3(mol),
        "NumRotatableBonds": Lipinski.NumRotatableBonds(mol),
        "NumAromaticHeterocycles": Lipinski.NumAromaticHeterocycles(mol),
        "Aromatic Ring Count": Lipinski.NumAromaticRings(mol),
        "Aliphatic Ring Count": Lipinski.NumAliphaticRings(mol),
        "Ring Count": rdMolDescriptors.CalcNumRings(mol),
        "NumBridgeheadAtoms": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        "NumSpiroAtoms": rdMolDescriptors.CalcNumSpiroAtoms(mol),
        "NumHDonors": Lipinski.NumHDonors(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "HeteroAtomFraction": Lipinski.NumHeteroatoms(mol) / ha,
        "AromaticAtomCount": aro,
        "AromaticAtomFraction": aro / ta,
        "ConjugatedBondCount": con,
        "ConjugatedBondFraction": con / tb,
        "TripleBondCount": bond_count(mol, Chem.BondType.TRIPLE),
        "UnsaturatedBondCount": bond_count(mol, Chem.BondType.DOUBLE) + bond_count(mol, Chem.BondType.TRIPLE),
        "UnsaturatedBondFraction": (bond_count(mol, Chem.BondType.DOUBLE) + bond_count(mol, Chem.BondType.TRIPLE)) / tb,
        "H Acceptors": Lipinski.NumHAcceptors(mol),
        "O": atom_count(mol, 8),
        "N Atoms": atom_count(mol, 7),
        "single": bond_count(mol, Chem.BondType.SINGLE),
        "double": bond_count(mol, Chem.BondType.DOUBLE),
        "BCUT2D_LOGPHI": Descriptors.BCUT2D_LOGPHI(mol),
        "BCUT2D_LOGPLOW": Descriptors.BCUT2D_LOGPLOW(mol),
        "fr_ester": Fragments.fr_ester(mol),
        "fr_NH0": Fragments.fr_NH0(mol),
        "fr_alkyl_halide": Fragments.fr_alkyl_halide(mol),
        "fr_allylic_oxid": Fragments.fr_allylic_oxid(mol),
        "fr_C_O_noCOO": Fragments.fr_C_O_noCOO(mol),
        "fr_ketone": Fragments.fr_ketone(mol),
        "fr_term_acetylene": Fragments.fr_term_acetylene(mol),
        "fr_bicyclic": Fragments.fr_bicyclic(mol),
        "fr_NH2": Fragments.fr_NH2(mol),
    }

    all_feats = set(E0_FEATURES + ET_FEATURES + OXI_FEATURES + PHI_FEATURES)
    for name in all_feats:
        if name in f:
            continue
        if name.startswith("PEOE_VSA") or name.startswith("SMR_VSA") or name.startswith("SlogP_VSA"):
            f[name] = getattr(MolSurf, name, lambda m: 0.0)(mol)
        elif name.startswith("EState_VSA") or name.startswith("VSA_EState"):
            f[name] = getattr(EState_VSA, name, lambda m: 0.0)(mol)
        elif name.startswith("MACCS_"):
            f[name] = _maccs(mol, int(name.split("_", 1)[1]))
        elif name.startswith("ECFP2_"):
            f[name] = _ecfp(mol, int(name.split("_", 1)[1]), MORGAN_R2)
        elif name.startswith("ECFP3_"):
            f[name] = _ecfp(mol, int(name.split("_", 1)[1]), MORGAN_R3)
        elif name.startswith("ECFP_"):
            f[name] = _ecfp(mol, int(name.split("_", 1)[1]), MORGAN_R2)

    return f


def fetch_pubchem(identifier: str) -> dict:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(identifier)}/property/CanonicalSMILES,ConnectivitySMILES,InChI,IUPACName,Title/JSON"
    r = requests.get(url, timeout=30)
    if r.status_code == 404:
        raise LookupError(f"PubChem not found: {identifier}")
    r.raise_for_status()
    return r.json()["PropertyTable"]["Properties"][0]






def build_plot(
    df: pd.DataFrame,
    phi_filter: list[str] | None = None,
    abs_filter: list[str] | None = None,
    highlighted: list[str] | None = None,
) -> go.Figure:
    """
    Build the interactive scatter plot.
    phi_filter:  list of Phi_ISC labels to show; None = all
    abs_filter:  list of Absorb_Class labels to show; None = all
    highlighted: list of Compound names whose ET and E' dashed lines are shown
                 (scatter point is shown ONLY if the compound also passes BOTH filters above)
    """
    fig = go.Figure()

    hl_set = set(highlighted) if highlighted else set()

    # ── separate highlighted from non-highlighted FIRST ──────────────────────
    # Highlighted compounds are drawn separately (see below)
    # Non-highlighted compounds go through the normal filter
    plot_df = df.copy()
    if phi_filter:
        plot_df = plot_df[plot_df["Phi_ISC_Label"].isin(phi_filter)]
    if abs_filter:
        plot_df = plot_df[plot_df["Absorb_Class"].isin(abs_filter)]

    # ── scatter: non-highlighted compounds only (filtered) ────────────────────
    for label, color in PHI_COLORS.items():
        sub = plot_df[~plot_df["Compound"].isin(hl_set)]
        sub = sub[sub["Phi_ISC_Label"] == label]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["E0_triplet"], y=sub["ET"],
            mode="markers",
            marker=dict(size=12, color=color, opacity=0.85, line=dict(width=1, color="white")),
            name=f"Phi_ISC {label.capitalize()}",
            text=[f"<b>{r.get('Compound','')}</b><br>"
                  f"ET: {r.get('ET',''):.2f} kJ mol⁻¹<br>"
                  f"Triplet E₀: {r.get('E0_triplet',''):.3f} V<br>"
                  f"E′: {r.get('E_ox_SHE',''):.3f} V<br>"
                  f"Phi_ISC: {label}<br>"
                  f"Absorb: {r.get('Absorb_Class','')}"
                  for _, r in sub.iterrows()],
            hoverinfo="text",
        ))

    # ── O₂→¹O₂* threshold line at 94 kJ mol⁻¹ ───────────────────────────────
    fig.add_hline(
        y=94,
        line=dict(color="#dc2626", width=2, dash="dash"),
        annotation_text="Energy for O<sub>2</sub>→<sup>1</sup>O<sub>2</sub>* (94 kJ mol⁻¹)",
        annotation_position="top right",
        annotation_font=dict(color="#dc2626", size=11),
    )

    # ── highlighted compound dashed lines ────────────────────────────────────
    # NOTE: only dashed lines (no scatter points) — scatter points are handled by the filter above
    if highlighted:
        colors_hl = ["#7c3aed","#0891b2","#059669","#d97706","#db2777","#4338ca"]
        for idx, cname in enumerate(highlighted):
            mask = df["Compound"] == cname
            if not mask.any():
                continue
            row_hl = df[mask].iloc[0]
            et_val   = float(row_hl.get("ET", np.nan))
            eox_val  = float(row_hl.get("E_ox_SHE", np.nan))
            hl_color = colors_hl[idx % len(colors_hl)]
            short = cname[:18] + "…" if len(cname) > 18 else cname

            # Dashed lines ONLY — scatter point is NOT drawn here
            if not np.isnan(et_val):
                fig.add_hline(
                    y=et_val,
                    line=dict(color=hl_color, width=1.5, dash="dot"),
                    annotation_text=f"ET={et_val:.1f} ({short})",
                    annotation_position="top left",
                    annotation_font=dict(color=hl_color, size=10),
                )
            if not np.isnan(eox_val):
                fig.add_vline(
                    x=eox_val,
                    line=dict(color=hl_color, width=1.5, dash="dot"),
                    annotation_text=f"E′={eox_val:.3f} ({short})",
                    annotation_position="top right",
                    annotation_font=dict(color=hl_color, size=10),
                )

    fig.update_layout(
        title=dict(
            text="ET vs. Triplet E₀ / ET 与三线态 E₀",
            x=0.5, font=dict(size=18, color="#1f2937"),
        ),
        xaxis=dict(title="Triplet E₀ (V vs SHE)",
                   gridcolor="rgba(0,0,0,0.12)", color="#000000",
                   title_font=dict(color="#000000"),
                   tickfont=dict(color="#000000"),
                   linecolor="#000000", linewidth=1.5,
                   zerolinecolor="rgba(0,0,0,0.3)",
                   range=[0, 2.5]),
        yaxis=dict(title="ET (kJ mol⁻¹)",
                   gridcolor="rgba(0,0,0,0.12)", color="#000000",
                   title_font=dict(color="#000000"),
                   tickfont=dict(color="#000000"),
                   linecolor="#000000", linewidth=1.5,
                   zerolinecolor="rgba(0,0,0,0.3)",
                   range=[0, 400]),
        plot_bgcolor="rgba(255,255,255,0.6)", paper_bgcolor="rgba(255,255,255,0)",
        legend=dict(font=dict(color="#1f2937"), bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.12)", borderwidth=1),
        height=520,
    )
    return fig


# ─── Excel builder ────────────────────────────────────────────────────────────
def _build_excel_bytes(df: pd.DataFrame, errors: list) -> bytes:
    import io
    wb = Workbook()
    ws = wb.active
    ws.title = "Predictions"

    headers = [
        "No.", "Identifier", "Type", "Name", "SMILES", "InChI",
        "Abundance", "ET (kJ/mol)", "E₀ ground (V)", "E₀ triplet (V)",
        "Phi_ISC class", "Phi_ISC confidence", "E_ox (V SHE)", "Absorb class",
    ]
    ws.append(headers)
    for _, r in df.iterrows():
        ws.append([
            r.get("No.", ""),
            r.get("Identifier", ""),
            r.get("IdentifierType", ""),
            r.get("Compound", ""),
            r.get("SMILES", ""),
            r.get("INCHI", ""),
            r.get("Abundance", ""),
            round(r.get("ET", np.nan), 4) if pd.notna(r.get("ET")) else "",
            round(r.get("E0_ground", np.nan), 4) if pd.notna(r.get("E0_ground")) else "",
            round(r.get("E0_triplet", np.nan), 4) if pd.notna(r.get("E0_triplet")) else "",
            r.get("Phi_ISC_Label", ""),
            round(r.get("Phi_ISC_Confidence", np.nan), 4) if pd.notna(r.get("Phi_ISC_Confidence")) else "",
            round(r.get("E_ox_SHE", np.nan), 4) if pd.notna(r.get("E_ox_SHE")) else "",
            r.get("Absorb_Class", ""),
        ])

    if errors:
        ws2 = wb.create_sheet("Errors")
        ws2.append(["Row", "Identifier", "Error"])
        for e in errors:
            ws2.append([e.get("row",""), e.get("identifier",""), e.get("error","")])

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.read()


# ─── Custom CSS – aerosol photochemistry theme ────────────────────────────────

# Load hero background image as base64
_hero_img_path = Path(__file__).parent / "header_bg.png"
_hero_bg_css = "none"
if _hero_img_path.exists():
    _hero_b64 = base64.b64encode(_hero_img_path.read_bytes()).decode()
    _hero_bg_css = f"url('data:image/png;base64,{_hero_b64}')"

st.markdown(f"""
<style>
/* ── App background – light ─────────────────────────── */
.stApp {{
    background: #f0f4fa !important;
}}
.stApp > header {{
    background: transparent !important;
}}
.block-container {{
    padding-top: 0.5rem !important;
    padding-bottom: 2rem !important;
    background: transparent !important;
}}

/* ── Hero header image ──────────────────────────────── */
.hero-wrap {{
    width: 100%;
    height: 540px;
    background: {_hero_bg_css} center/cover no-repeat;
    border-radius: 0 0 22px 22px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 40px rgba(0,0,0,0.28);
    margin-bottom: 0;
    margin-top: -1rem;
    margin-left: -1rem;
    margin-right: -1rem;
    width: calc(100% + 2rem);
}}
.hero-wrap::after {{
    content: "";
    position: absolute; inset: 0;
    background: linear-gradient(
        180deg,
        rgba(5,10,30,0.08) 0%,
        rgba(5,10,30,0.38) 38%,
        rgba(5,10,30,0.75) 70%,
        rgba(5,10,30,0.92) 100%
    );
}}
.hero-inner {{
    position: absolute;
    bottom: 2.8rem; left: 3rem; right: 3rem;
    z-index: 2;
}}
.hero-inner h1 {{
    color: #ffffff !important;
    font-size: 3.4rem !important;
    margin: 0 0 0.5rem !important;
    text-shadow: 0 4px 22px rgba(0,0,0,0.60);
    font-weight: 900 !important;
    letter-spacing: -0.025em;
    line-height: 1.05;
}}
.hero-inner p {{
    color: rgba(255,255,255,0.94) !important;
    font-size: 1.28rem !important;
    margin: 0 !important;
    text-shadow: 0 2px 10px rgba(0,0,0,0.42);
}}
.hero-inner .hero-sub {{
    font-size: 0.95rem !important;
    color: rgba(255,255,255,0.72) !important;
    margin-top: 0.4rem !important;
}}
.hero-tagline {{
    font-size: 1.05rem !important;
    font-style: italic;
    color: rgba(255,255,255,0.80) !important;
    text-shadow: 0 1px 8px rgba(0,0,0,0.38);
    margin-bottom: 0.3rem !important;
    line-height: 1.5;
}}

/* ── Sidebar – keep dark for contrast ─────────────── */
section[data-testid="stSidebar"] {{
    background: #1e2340 !important;
    border-right: 1px solid rgba(99,102,241,0.22);
}}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {{
    color: rgba(255,255,255,0.82) !important;
}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: rgba(255,255,255,0.96) !important;
}}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {{
    background: rgba(255,255,255,0.09) !important;
    color: white !important;
    border-color: rgba(99,102,241,0.45) !important;
}}

/* ── Metric cards – light glass ────────────────────── */
[data-testid="stMetric"],
[data-testid="stMetricLabel"] {{
    background: rgba(255,255,255,0.75) !important;
    border: 1px solid rgba(99,102,241,0.14) !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
}}
[data-testid="stMetricValue"] {{
    color: #3730a3 !important;
    font-weight: 700 !important;
    font-size: 1.15rem !important;
}}
[data-testid="stMetricLabel"] {{
    color: #6366f1 !important;
    font-size: 0.72rem !important;
}}

/* ── Buttons ─────────────────────────────────────────── */
.stButton > button,
.stDownloadButton > button {{
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.38) !important;
}}
.stButton > button:hover,
.stDownloadButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 7px 22px rgba(99,102,241,0.55) !important;
}}

/* ── File uploader ──────────────────────────────────── */
[data-testid="stFileUploader"] > div {{
    background: rgba(255,255,255,0.82) !important;
    border: 2px dashed rgba(99,102,241,0.32) !important;
    border-radius: 10px !important;
}}

/* ── Tabs ───────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
.stTabs [data-baseweb="tab"] {{
    background: rgba(255,255,255,0.68);
    border: 1px solid rgba(99,102,241,0.16);
    border-radius: 8px 8px 0 0;
    color: #4b5563;
    font-weight: 500;
}}
.stTabs [aria-selected="true"] {{
    background: rgba(99,102,241,0.14) !important;
    color: #4338ca !important;
    border-color: rgba(99,102,241,0.42) !important;
    font-weight: 600;
}}

/* ── Table – white cards on light bg ──────────────── */
[data-testid="stDataFrame"] table {{
    color: #1f2937 !important;
    font-size: 0.85rem;
}}
[data-testid="stDataFrame"] th {{
    background: rgba(99,102,241,0.11) !important;
    color: #3730a3 !important;
    font-weight: 700;
}}
[data-testid="stDataFrame"] td {{
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(0,0,0,0.06) !important;
}}
[data-testid="stDataFrame"] tr:hover td {{
    background: rgba(99,102,241,0.06) !important;
}}

/* ── Section headers ─────────────────────────────────── */
.section-header {{
    font-size: 1.05rem;
    font-weight: 700;
    color: #3730a3;
    border-bottom: 2px solid rgba(99,102,241,0.18);
    padding-bottom: 0.4rem;
    margin: 1.1rem 0 0.5rem;
    letter-spacing: 0.01em;
}}

/* ── Model cards ────────────────────────────────────── */
.model-card-box {{
    background: rgba(255,255,255,0.85);
    border: 1px solid rgba(99,102,241,0.16);
    border-radius: 12px;
    padding: 0.85rem 0.75rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    height: 100%;
}}
.model-card-en {{
    font-size: 1.05rem;
    font-weight: 700;
    color: #4338ca;
    line-height: 1.3;
}}
.model-card-zh {{
    font-size: 0.76rem;
    color: #6366f1;
    margin-top: 3px;
}}
.model-card-unit {{
    font-size: 0.70rem;
    color: #9ca3af;
    margin-top: 3px;
}}

/* ── Credits box ────────────────────────────────────── */
.credits-box {{
    background: rgba(255,255,255,0.90);
    border: 1px solid rgba(99,102,241,0.16);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-top: 2rem;
    font-size: 0.82rem;
    line-height: 1.75;
    color: #374151;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}}
.credits-box .team-title {{
    font-size: 1rem;
    font-weight: 700;
    color: #4338ca;
    margin-bottom: 0.6rem;
    letter-spacing: 0.03em;
}}
.credits-box .mbr-name {{
    font-weight: 600;
    color: #1e40af;
}}
.credits-box .mbr-zh {{
    font-size: 0.78rem;
    color: #6366f1;
    margin-left: 0.3rem;
}}
.credits-box .mbr-uni {{
    font-size: 0.76rem;
    color: #6b7280;
}}

/* ── Divider ─────────────────────────────────────────── */
hr {{ border-color: rgba(99,102,241,0.14) !important; }}

/* ── Expander ───────────────────────────────────────── */
.streamlit-expanderHeader {{
    background: rgba(255,255,255,0.82) !important;
    border-radius: 8px !important;
    border: 1px solid rgba(99,102,241,0.14) !important;
    color: #3730a3 !important;
    font-weight: 600;
}}

/* ── Text inputs ────────────────────────────────────── */
.stTextInput input,
.stTextArea textarea {{
    background: rgba(255,255,255,0.92) !important;
    color: #1f2937 !important;
    border: 1px solid rgba(99,102,241,0.22) !important;
    border-radius: 6px;
}}

/* ── Selectbox ──────────────────────────────────────── */
.stSelectbox > div > div {{
    background: rgba(255,255,255,0.92) !important;
    border-radius: 6px;
}}

/* ── Markdown text ──────────────────────────────────── */
.stMarkdown h2, .stMarkdown h3 {{ color: #3730a3 !important; }}
p, .stText {{ color: #374151 !important; }}

/* ── Alert boxes ─────────────────────────────────────── */
.stAlert {{ border-radius: 8px; }}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌫️ BrC-Triplet-Atlas")
    st.caption("棕碳光敏性质预测平台")
    st.divider()

    st.markdown("**📌 Input / 输入**")
    input_mode = st.radio(
        "Mode / 模式",
        ["✏️ Manual / 手动输入", "📄 Upload CSV / 上传 CSV"],
        label_visibility="collapsed",
    )

    identifiers_text = ""
    uploaded_file = None
    use_abundance = False

    if input_mode == "✏️ Manual / 手动输入":
        identifiers_text = st.text_area(
            "Compound names or CAS numbers (one per line)\n化合物名称或 CAS 号（每行一个）",
            placeholder="ethanol\n64-17-5\nbenzene",
            height=140,
        )
    else:
        uploaded_file = st.file_uploader(
            "CSV file (name/cas/abundance columns)\nCSV 文件（支持 name/cas/abundance 列）",
            type=["csv"],
        )
        if uploaded_file:
            try:
                tmp_df = pd.read_csv(uploaded_file)
                name_col = next((c for c in tmp_df.columns if c.lower() in ("name","compound_name","chemical_name","compound","title")), None)
                cas_col  = next((c for c in tmp_df.columns if c.lower() in ("cas","cas_number","casno","registry_number")), None)
                abun_col = next((c for c in tmp_df.columns if c.lower() in ("abundance","relative_abundance","weight")), None)
                st.session_state["input_df"] = tmp_df
                st.success(f"✅ Loaded / 已加载：{len(tmp_df)} rows / 行  |  name={name_col}  cas={cas_col}  abundance={abun_col}")
            except Exception as e:
                st.error(f"CSV error: {e}")

        use_abundance = st.checkbox(
            "Use abundance weighting / 使用丰度加权",
            value=False,
        )

    st.divider()
    st.markdown("**⚙️ Options / 选项**")
    st.markdown(
        "**Triplet E₀ 计算公式 / Triplet E₀ Formula**\n\n"
        "`Triplet E₀ = E₀(ground) + ET / F`\n\n"
        "where **F = 96.485 kJ mol⁻¹ V⁻¹**",
        help="三线态 E₀ = 基态 E₀ + 三线态能量 / 法拉第常数",
    )
    faraday = st.number_input(
        "Faraday constant (kJ mol⁻¹ V⁻¹) / 法拉第常数",
        value=96.485, step=0.001, format="%.3f",
    )

    st.divider()
    run_clicked = st.button("🚀 Run Prediction / 运行预测", use_container_width=True)

# ─── Main area ────────────────────────────────────────────────────────────────

# ── Hero header image ──
st.markdown("""
<div class="hero-wrap">
  <div class="hero-inner">
    <p class="hero-tagline">"Wanna know if your molecule is photosensitizing?<br>想知道你的分子光不光敏吗？"</p>
    <h1>BrC-Triplet-Atlas</h1>
    <p>棕碳光敏化特性预测平台 &nbsp;·&nbsp; Based on Machine-Learning</p>
    <p class="hero-sub">All from "only the chemical name or CAS" to predict photosensitizing abilities (E₀ &nbsp;·&nbsp; ET &nbsp;·&nbsp; Phi_ISC &nbsp;·&nbsp; E′ &nbsp;·&nbsp; Absorb)（从"仅化学名称或 CAS"出发，预测光敏化特性）</p>
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Model cards ──
st.markdown(
    "<div class='section-header'>📊 Models &amp; Outputs / 模型与输出指标</div>",
    unsafe_allow_html=True,
)
m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
model_cards = [
    (m_col1, "E₀", "基态单电子还原电位", "V vs SHE", "Ground-State One-Electron Reduction Potential"),
    (m_col2, "ET", "三线态能量", "kJ mol⁻¹", "Triplet Energy (T1, the lowest state)"),
    (m_col3, "Phi_ISC", "系间窜越量子产率<br><span style='font-size:0.68rem;color:#9ca3af'>low&lt;0.1 · 0.1–0.4 med · &gt;0.4 high</span>", "Low / Med / High", "Inter-System Crossing Quantum Yield"),
    (m_col4, "E′", "单电子氧化电位", "V vs SHE", "One-Electron Oxidation Potential"),
    (m_col5, "Absorb", "紫外-可见吸光等级<br><span style='font-size:0.68rem;color:#9ca3af'>300–400 nm molar absorptivity (MAE): ≤936 low · 936–4755 med · &gt;4755 high (M⁻¹cm⁻¹)</span>", "Low / Med / High", "UV-Vis Absorption Class"),
]
for col, sym, zh, unit, en_full in model_cards:
    col.markdown(
        f"<div class='model-card-box'>"
        f"<div class='model-card-en'>{sym}</div>"
        f"<div class='model-card-en' style='font-size:0.73rem;font-weight:400;color:#6b7280'>{en_full}</div>"
        f"<div class='model-card-zh'>{zh}</div>"
        f"<div class='model-card-unit'>{unit}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ── Prediction area ──
if run_clicked:
    with st.spinner("🔬 Resolving structures & predicting... / 正在解析结构与预测…"):

        # Build input DataFrame
        if uploaded_file and "input_df" in st.session_state:
            input_df = st.session_state["input_df"]
        elif identifiers_text.strip():
            lines = [l.strip() for l in identifiers_text.splitlines() if l.strip()]
            input_df = pd.DataFrame({"identifier": lines})
        else:
            st.warning("⚠️ Please enter at least one identifier. / 请输入至少一个化合物名称或 CAS 号。")
            st.stop()

        rows: list[dict] = []
        errors: list[dict] = []
        progress_bar = st.progress(0, text="Initializing / 初始化…")

        for i, (_, row) in enumerate(input_df.iterrows()):
            pct = int(((i + 1) / len(input_df)) * 100)
            progress_bar.progress(pct, text=f"Processing row {i+1}/{len(input_df)} / 处理第 {i+1} 行…")

            # choose identifier
            identifier = None
            for col in ["cas","cas_number","identifier","query","name","compound_name","chemical_name"]:
                if col in row.index and pd.notna(row.get(col)) and str(row.get(col)).strip():
                    identifier = str(row.get(col)).strip()
                    break
            if identifier is None:
                errors.append({"row": i+1, "error": "No identifier found / 未找到标识符"})
                continue

            kind = "CAS" if CAS_PATTERN.match(identifier) else "name"
            abundance = 1.0
            if use_abundance:
                for key in ["abundance","Abundance","ABUNDANCE","relative_abundance"]:
                    if key in row.index and pd.notna(row.get(key)):
                        try:
                            abundance = float(row.get(key))
                            break
                        except Exception:
                            pass

            try:
                record = fetch_pubchem(identifier)
                smiles = record.get("CanonicalSMILES") or record.get("ConnectivitySMILES") or ""
                if not smiles:
                    raise ValueError("No SMILES from PubChem")
                feats = compute_features(smiles)
                row_dict = {
                    "No.": i + 1,
                    "Identifier": identifier,
                    "IdentifierType": kind,
                    "PubChemCID": record.get("CID"),
                    "Compound": record.get("Title") or record.get("IUPACName") or identifier,
                    "Chemical name": record.get("IUPACName") or identifier,
                    "SMILES": smiles,
                    "INCHI": record.get("InChI", ""),
                    "Abundance": abundance,
                    **feats,
                }
                rows.append(row_dict)
            except Exception as exc:
                errors.append({
                    "row": i+1,
                    "identifier": identifier,
                    "error": str(exc),
                })

        progress_bar.empty()

        if not rows:
            st.error("❌ No compounds could be resolved. Check identifiers and network. / 无法解析任何化合物，请检查输入和网络。")
            if errors:
                st.dataframe(pd.DataFrame(errors), use_container_width=True)
            st.stop()

        # Run predictions (models already loaded at module level)
        try:
            bar2 = st.progress(0, text="⚙️ Running predictions…")
            # Run predictions using cached models
            all_cols = (
                ["No.", "Identifier", "IdentifierType", "PubChemCID",
                 "Compound", "Chemical name", "SMILES", "INCHI", "Abundance"]
                + E0_FEATURES + ET_FEATURES + OXI_FEATURES + PHI_FEATURES
            )
            df = pd.DataFrame(rows)
            for c in all_cols:
                if c not in df.columns:
                    df[c] = np.nan
            df["E0"] = _model_e0.predict(_scaler_e0.transform(df[E0_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)))
            df["ET"] = _model_et.predict(_scaler_et.transform(df[ET_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)))
            x_phi = df[PHI_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            phi_cls = _model_phi.predict(_scaler_phi.transform(x_phi))
            phi_proba = _model_phi.predict_proba(_scaler_phi.transform(x_phi))
            confidence = np.max(phi_proba, axis=1)
            phi_labels = [PHI_CLASS_LABELS.get(int(c), "unknown") for c in phi_cls]
            df["Phi_ISC_Label"] = phi_labels
            df["Phi_ISC_Confidence"] = confidence
            df["E_ox_SHE"] = _model_oxi.predict(_scaler_oxi.transform(df[OXI_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)))
            df["Absorb_Class"] = "unavailable"
            if G2_PIPELINE.exists():
                classes = []
                for smiles in df["SMILES"]:
                    try:
                        classes.append(_g2_predict(str(smiles)).lower())
                    except Exception:
                        classes.append("unavailable")
                df["Absorb_Class"] = classes
            df["E0_ground"] = df["E0"]
            df["E0_triplet"] = df["E0_ground"] + df["ET"] / FARADAY
            pred_df = df
            bar2.progress(100, text="✅ Done!")
            bar2.empty()
        except Exception as exc:
            bar2.empty()
            st.error(f"❌ Prediction failed / 预测失败: {exc}")
            st.stop()

        # Store in session
        st.session_state["pred_df"] = pred_df
        st.session_state["errors"] = errors

# ── Results ──
if "pred_df" in st.session_state:
    pred_df: pd.DataFrame = st.session_state["pred_df"]
    errors: list = st.session_state.get("errors", [])

    # ── Table filter controls (MUST come before plot so checkbox change triggers rerun first) ──
    st.markdown(
        "<div class='section-header'>🔍 Filter &amp; Sort / 筛选与排序 &nbsp;·&nbsp; ☑️ 勾选后在图上标注 / Check to mark on plot</div>",
        unsafe_allow_html=True,
    )

    f1, f2, f3 = st.columns([2, 2, 1])
    search_str = f1.text_input("Search name / 搜索名称", placeholder="ethanol…")
    phi_filter_sel = f2.selectbox(
        "Phi_ISC class / 系间窜越类别",
        ["All / 全部", "low / 低", "medium / 中", "high / 高"],
    )
    _sort_options = ["No.", "ET", "E0_triplet", "E_ox_SHE", "E0_ground"]
    sort_options  = [c for c in _sort_options if c in pred_df.columns]
    sort_col = f3.selectbox("Sort by / 排序依据", sort_options)

    # Build disp DataFrame with _mark column initialized from session state
    hl_prev: list[str] = st.session_state.get("_hl_cnames", [])
    disp = pred_df.copy()
    disp["_mark"] = disp["Compound"].isin(hl_prev)

    if search_str:
        mask = disp["Compound"].str.contains(search_str, case=False, na=False) | \
               disp["Identifier"].str.contains(search_str, case=False, na=False)
        disp = disp[mask]
    if phi_filter_sel != "All / 全部":
        key = phi_filter_sel.split("/")[0].strip()
        disp = disp[disp["Phi_ISC_Label"] == key]
    if sort_col in disp.columns:
        disp = disp.sort_values(sort_col)
    else:
        disp = disp.sort_values("No.")

    # Build display table
    disp_out = disp[[
        "_mark", "No.", "Compound", "Identifier", "SMILES",
        "ET", "E0_ground", "E0_triplet", "Phi_ISC_Label", "E_ox_SHE", "Absorb_Class",
    ]].copy()
    disp_out = disp_out.rename(columns={
        "Compound": "Name",
        "ET": "ET (kJ mol⁻¹)",
        "E0_ground": "E0 ground (V)",
        "E0_triplet": "E0 triplet (V)",
        "Phi_ISC_Label": "Phi_ISC",
        "E_ox_SHE": "E' (V)",
        "Absorb_Class": "Absorb",
    })

    col_cfg = {
        "_mark": st.column_config.CheckboxColumn("Plot / 标注", default=False, width="small"),
        "No.": st.column_config.NumberColumn("No.", width="small"),
        "Name": st.column_config.TextColumn("Name", width="medium"),
        "Identifier": st.column_config.TextColumn("Identifier", width="small"),
        "SMILES": st.column_config.TextColumn("SMILES", width="medium"),
        "ET (kJ mol⁻¹)": st.column_config.NumberColumn("ET (kJ mol⁻¹)", format="%.2f", width="small"),
        "E0 ground (V)": st.column_config.NumberColumn("E0 ground (V)", format="%.3f", width="small"),
        "E0 triplet (V)": st.column_config.NumberColumn("E0 triplet (V)", format="%.3f", width="small"),
        "Phi_ISC": st.column_config.TextColumn("Phi_ISC", width="small"),
        "E' (V)": st.column_config.NumberColumn("E' (V)", format="%.3f", width="small"),
        "Absorb": st.column_config.TextColumn("Absorb", width="small"),
    }
    edited = st.data_editor(
        disp_out,
        column_config=col_cfg,
        hide_index=True,
        use_container_width=True,
        height=420,
        key="_table_editor",
    )

    # Sync highlighted compounds from checkbox column back to session state
    if "_mark" in edited.columns:
        marked_mask = edited["_mark"].astype(bool)
        # Use Compound column from disp (original index preserved)
        hl_cnames = disp.loc[marked_mask, "Compound"].dropna().tolist()
        if hl_cnames != st.session_state.get("_hl_cnames", []):
            st.session_state["_hl_cnames"] = hl_cnames
            st.rerun()

    st.divider()

    # ── Plot filter controls ──────────────────────────────────────────────────
    st.markdown(
        "<div class='section-header'>📈 Interactive Scatter Plot / 交互散点图</div>",
        unsafe_allow_html=True,
    )

    pf_col1, pf_col2 = st.columns(2)
    with pf_col1:
        all_phi_labels = ["low", "medium", "high", "unknown"]
        all_phi_labels = [l for l in all_phi_labels if l in pred_df["Phi_ISC_Label"].values]
        phi_plot_filter = st.multiselect(
            "Filter by Phi_ISC class / 按 Phi_ISC 筛选",
            options=all_phi_labels,
            default=all_phi_labels,
            format_func=lambda x: x.capitalize(),
        )
    with pf_col2:
        all_abs_labels = sorted(pred_df["Absorb_Class"].dropna().unique().tolist())
        abs_plot_filter = st.multiselect(
            "Filter by Absorb class / 按吸光等级筛选",
            options=all_abs_labels,
            default=all_abs_labels,
            format_func=lambda x: x.capitalize(),
        )

    # Build plot with current highlighted compounds
    hl_cnames = st.session_state.get("_hl_cnames", [])
    fig = build_plot(
        pred_df,
        phi_filter=phi_plot_filter if phi_plot_filter else None,
        abs_filter=abs_plot_filter if abs_plot_filter else None,
        highlighted=hl_cnames if hl_cnames else None,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Download ──────────────────────────────────────────────────────────────
    dl1, dl2 = st.columns(2)
    # Export from original pred_df (without _mark column)
    exp = pred_df.copy()
    exp_out = exp[[
        "No.", "Compound", "Identifier", "SMILES",
        "ET", "E0_ground", "E0_triplet", "Phi_ISC_Label", "E_ox_SHE", "Absorb_Class",
    ]].rename(columns={
        "ET": "ET (kJ mol⁻¹)",
        "E0_ground": "E0 ground (V)",
        "E0_triplet": "E0 triplet (V)",
        "Phi_ISC_Label": "Phi_ISC",
        "E_ox_SHE": "E' (V)",
        "Absorb_Class": "Absorb",
    })
    buf_csv = exp_out.to_csv(index=False).encode("utf-8-sig")
    dl1.download_button(
        "📥 Download CSV / 下载 CSV",
        buf_csv,
        "brc_predictions.csv",
        "text/csv",
        use_container_width=True,
    )

    excel_bytes = _build_excel_bytes(pred_df, errors)
    dl2.download_button(
        "📥 Download Excel / 下载 Excel",
        excel_bytes,
        "brc_predictions.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    if errors:
        st.warning(f"⚠️ {len(errors)} compound(s) failed to resolve. / {len(errors)} 个化合物解析失败。")
        st.dataframe(pd.DataFrame(errors), use_container_width=True)

else:
    # Empty state
    st.info(
        "⬆️ Enter compounds in the sidebar and click **Run Prediction** to start.\n\n"
        "在侧边栏输入化合物，然后点击 **运行预测** 开始。\n\n"
        "___\n"
        "**Model outputs / 模型输出：**\n"
        "- **E₀ (V vs SHE)** — Ground-State One-Electron Reduction Potential / 基态单电子还原电位\n"
        "- **ET (kJ mol⁻¹)** — Triplet Energy (T1, the lowest state) / 三线态能量\n"
        "- **Phi_ISC (Low / Med / High)** — Inter-System Crossing Quantum Yield / 系间窜越量子产率\n"
        "  _(Low: Φ < 0.1 · Medium: 0.1–0.4 · High: Φ > 0.4)_\n"
        "- **E′ (V vs SHE)** — One-Electron Oxidation Potential / 单电子氧化电位\n"
        "- **Absorb (Low / Med / High)** — UV-Vis Absorption Class / 紫外-可见吸光等级\n"
        "  _(300–400 nm molar absorptivity (MAE): ≤936 low · 936–4755 med · >4755 high M⁻¹cm⁻¹)_"
    )


# ─── Footer credits ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="credits-box">
  <div class="team-title">💬 Contact &amp; Feedback / 联系与反馈</div>
  Whether you have encountered problems when using the model or have suggestions and feedback, please feel free to reach out!<br>
  如果您在使用模型时遇到问题或有建议和反馈，欢迎随时联系！<br><br>
  <div style="color:#374151;"><strong>For citation and reference:</strong> <span style="text-decoration:underline;">Profiling the Photosensitizing Properties of Atmospheric Brown Carbon. <em>ACS ES&T Air</em> 2025, 2 (10), 2081–2091. DOI: 10.1021/acsestair.5c00098</span></div><br><br>
  <div class="team-title">🧑‍💻 Platform Development / 平台开发</div>
  <div class="mbr-name">Dr. Zhancong Liang (梁展聪)</div>
  <div class="mbr-uni">&nbsp;&nbsp;&nbsp;University of Toronto / 多伦多大学 &nbsp;|&nbsp;
  <a href="mailto:zhancong.liang@utoronto.ca">zhancong.liang@utoronto.ca</a></div><br>
  <div class="mbr-name">Dr. Liyuan Zhou (周丽缘)</div>
  <div class="mbr-uni">&nbsp;&nbsp;&nbsp;Institute of Urban Environment, CAS / 中国科学院城市环境研究所</div><br>
  <div class="mbr-name">Mr. Yuqing Chang (常宇清)</div>
  <div class="mbr-uni">&nbsp;&nbsp;&nbsp;King Abdullah University of Science and Technology / 沙特阿卜杜拉国王科技大学</div><br>
  <div class="mbr-name">With all intellectual and material supports from Professor Chak K. Chan (陈泽强)</div>
  <div class="mbr-uni">&nbsp;&nbsp;&nbsp;<a href="https://www.kaust.edu.sa/en/study/faculty/chak-k-chan" target="_blank">King Abdullah University of Science and Technology / 沙特阿卜杜拉国王科技大学</a></div>
</div>
""", unsafe_allow_html=True)
