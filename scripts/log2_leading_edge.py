"""Close the gene-level thread: re-derive the report Fig-5 per-cluster top-N
member ranking (mean_abs_c9) on raw vs log2(x+1), using the FIGURE's own
statistic (common.fit_per_protein_t = OLS Group-coef t, NOT the gradient's
EB-moderated t). Members = frozen WASC measured_uniprots (corpus proven stable)."""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "scripts/viz"))
from common import resolve_groups, fit_per_protein_t, CONTRAST_GROUPS, uniprot_to_hgnc_symbol
from scipy.stats import spearmanr

df = pd.read_csv(ROOT/"output/proteomics/all_als.data.csv", index_col=0)
md = pd.read_csv(ROOT/"output/proteomics/all_als.metadata.csv", index_col=0)
groups = resolve_groups(md)
df_log = np.log2(df + 1.0)

def tstats(d):
    return {c: fit_per_protein_t(d, md, groups, CONTRAST_GROUPS[c]) for c in
            ("C9 vs Sporadic","C9 vs Healthy","Sporadic vs Healthy")}
T_raw = tstats(df); T_log = tstats(df_log)

themes = json.load(open(ROOT/"data/wasc/cluster_members_v1.json"))["themes"]
measured = set(df.index)
TOP_N = 10

def rank(members, T):
    recs=[]
    for u in members:
        t1=float(T["C9 vs Sporadic"].get(u,0.0)); t2=float(T["C9 vs Healthy"].get(u,0.0))
        recs.append((u,(abs(t1)+abs(t2))/2.0))
    recs.sort(key=lambda r:r[1], reverse=True)
    return recs

# build symbol lookup once for all members
all_u=sorted({u for info in themes.values() for u in info["measured_uniprots"] if u in measured})
SYM=uniprot_to_hgnc_symbol(all_u)
def sym(u): return SYM.get(u,u)

OUT = ROOT / "output/leading_edge_log2"
OUT.mkdir(parents=True, exist_ok=True)


def rank_by_contrast(members, T, contrast_name):
    """Per-contrast leading edge: rank measured members by |t| for a SINGLE
    contrast (log2 scale), reusing the same T[...] access pattern as rank().
    Stable tie-break on uniprot so reruns are byte-identical; NaN t → 0.0."""
    recs = []
    for u in members:
        v = abs(float(T[contrast_name].get(u, 0.0)))
        recs.append((u, v if np.isfinite(v) else 0.0))
    recs.sort(key=lambda r: (-r[1], r[0]))
    return recs

out={}
for cl,info in themes.items():
    mem=[u for u in info["measured_uniprots"] if u in measured]
    rr=rank(mem,T_raw); lr=rank(mem,T_log)
    raw_top=[u for u,_ in rr[:TOP_N]]; log_top=[u for u,_ in lr[:TOP_N]]
    inter=set(raw_top)&set(log_top)
    # spearman over all members on mean_abs_c9
    rmap=dict(rr); lmap=dict(lr)
    common=list(rmap)
    rho=spearmanr([rmap[u] for u in common],[lmap[u] for u in common])[0]
    print(f"\n===== {cl} ({len(mem)} measured members) =====")
    print(f"  top-{TOP_N} overlap raw∩log2: {len(inter)}/{TOP_N}   Spearman(mean_abs_c9 all members)={rho:.3f}")
    print(f"  {'rank':>4} {'RAW (sym, score)':<28} {'LOG2 (sym, score)':<28} {'stable?'}")
    for i in range(TOP_N):
        ru,rs=rr[i]; lu,ls=lr[i]
        mark = "" if ru==lu else ("→moved" )
        print(f"  {i+1:>4} {sym(ru)+f' ({rs:.2f})':<28} {sym(lu)+f' ({ls:.2f})':<28} {'same' if ru==lu else mark}")
    dropped=[sym(u) for u in raw_top if u not in inter]
    added=[sym(u) for u in log_top if u not in inter]
    print(f"  raw-top-{TOP_N} dropped under log2: {dropped}")
    print(f"  log2-top-{TOP_N} newly entered:    {added}")
    out[cl]={"n_members":len(mem),"top_overlap":len(inter),"spearman":rho,
             "raw_top":[sym(u) for u in raw_top],"log2_top":[sym(u) for u in log_top],
             "dropped":dropped,"added":added,
             "raw_rank_u":[u for u,_ in rr],"log2_rank_u":[u for u,_ in lr]}

# Named-genes-in-report stability check
NAMED={
 "Splicing":["SRSF1","SRSF2","SRSF5","SRSF7","SRSF9","U2AF2","SF3B1","SF3B3","SF3B4","PRPF3","PRPF8","PRPF19"],
 "Chromatin":["MBD3","RBBP4","RBBP7","TRRAP","CBX5","MCM4","MCM5","MCM7","RAD21","MAD2L1","RCC1"],
 "Transport":["NUP35","NUP54","NUP62","NUP85","NUP88","NUP93","NUP107","NUP133","NUP160","NUP188","NUP205","KPNA1","IPO4","XPO7","RAE1","RANBP1"],
}
print("\n\n=== report-NAMED genes: are they in raw-top-10 / log2-top-10? ===")
for cl in themes:
    rt=set(out[cl]["raw_top"]); lt=set(out[cl]["log2_top"])
    named=NAMED.get(cl,[])
    in_raw=[g for g in named if g in rt]; in_log=[g for g in named if g in lt]
    print(f"  {cl}: named in raw-top10={in_raw}  | named in log2-top10={in_log}")
# ---------------------------------------------------------------------------
# WRITE per-contrast log2 leading-edge CSVs (replace the raw-derived lists).
# Each file is ranked by its OWN contrast's log2 |t| via rank_by_contrast, so
# the three files are distinct; spctrl is driven by Sporadic-vs-Healthy only.
# ---------------------------------------------------------------------------
CONTRAST_FILES = {
    "c9spor": "C9 vs Sporadic",
    "c9ctrl": "C9 vs Healthy",
    "spctrl": "Sporadic vs Healthy",
}
for code, name in CONTRAST_FILES.items():
    rows = []
    for cl, info in themes.items():
        mem = [u for u in info["measured_uniprots"] if u in measured]
        for i, (u, score) in enumerate(rank_by_contrast(mem, T_log, name)):
            rows.append({"cluster": cl, "uniprot": u, "symbol": sym(u),
                         "abs_t_log2": score, "rank": i + 1})
    pd.DataFrame(rows, columns=["cluster", "uniprot", "symbol", "abs_t_log2", "rank"]
                 ).to_csv(OUT / f"{code}.csv", index=False)
    print(f"wrote {OUT / (code + '.csv')}  ({len(rows)} rows)")

# ---------------------------------------------------------------------------
# WRITE the CANONICAL report Fig-5 leading edge on log2: members ranked by
# mean_abs_c9 = mean(|t_C9vsSpor|, |t_C9vsHealthy|) — the exact statistic the
# report uses — so there is a genuine log2 replacement for the raw Fig-5 list
# (the per-contrast files above are the finer-grained companion).
# ---------------------------------------------------------------------------
mrows = []
for cl, info in themes.items():
    mem = [u for u in info["measured_uniprots"] if u in measured]
    for i, (u, score) in enumerate(rank(mem, T_log)):
        mrows.append({"cluster": cl, "uniprot": u, "symbol": sym(u),
                      "mean_abs_c9_log2": score, "rank": i + 1})
pd.DataFrame(mrows, columns=["cluster", "uniprot", "symbol", "mean_abs_c9_log2", "rank"]
             ).to_csv(OUT / "mean_abs_c9.csv", index=False)
print(f"wrote {OUT / 'mean_abs_c9.csv'}  ({len(mrows)} rows)")

# ---------------------------------------------------------------------------
# WRITE raw-vs-log2 churn report (canonical mean_abs_c9 Fig-5 leading edge).
# ---------------------------------------------------------------------------
def _ov(a, b, k):
    kk = min(k, len(a))
    return len(set(a[:k]) & set(b[:k])), kk

mdl = ["# Raw vs log2(x+1) leading-edge churn",
       "",
       "Per-cluster member ranking by the Fig-5 statistic "
       "`mean_abs_c9 = mean(|t_C9vsSpor|, |t_C9vsHealthy|)`, computed via "
       "`common.fit_per_protein_t` on **raw** intensities vs **log2(x+1)**.",
       "Cluster membership frozen from `data/wasc/cluster_members_v1.json`; "
       "UniProt-keyed throughout, HGNC symbols display-only.",
       ""]
for cl in themes:
    o = out[cl]; n = o["n_members"]
    ru = o["raw_rank_u"]; lu = o["log2_rank_u"]
    i10, d10 = _ov(ru, lu, 10)
    i50, d50 = _ov(ru, lu, 50)
    iall, dall = _ov(ru, lu, n)
    mdl += [f"## {cl} ({n} measured members)",
            "",
            f"- top-10 overlap raw∩log2: **{i10}/{d10}**",
            f"- top-50 overlap raw∩log2: **{i50}/{d50}**",
            f"- top-{n} (all-member) overlap raw∩log2: **{iall}/{dall}**",
            f"- Spearman(mean_abs_c9, all members) raw vs log2: **{o['spearman']:.3f}**",
            f"- raw-top-10 dropped under log2: {o['dropped'] or '—'}",
            f"- log2-top-10 newly entered: {o['added'] or '—'}",
            "",
            "| rank | raw top-10 (sym) | log2 top-10 (sym) |",
            "|---:|---|---|"]
    for i in range(10):
        mdl.append(f"| {i + 1} | {o['raw_top'][i]} | {o['log2_top'][i]} |")
    mdl.append("")
(OUT / "raw_vs_log2_churn.md").write_text("\n".join(mdl), encoding="utf-8")
print(f"wrote {OUT / 'raw_vs_log2_churn.md'}")

# machine-readable adjunct (relocated from /tmp so nothing lands there)
json.dump(out, open(OUT / "leading_edge_summary.json", "w"), indent=1)
print(f"wrote {OUT / 'leading_edge_summary.json'}")
