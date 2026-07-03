"""log2-vs-raw sensitivity of the per-protein gradient (Direction B).

Reuses the EXACT production gradient loop (run_gradient_test) + a cached
distance matrix, swapping ONLY the |t| transform. First validates that a
RAW recompute reproduces the cached result.json slopes (faithfulness gate),
then recomputes under log2(x+1) and reports drift.

NOTE: the cached output/landscape_* matrices are the OLD with-intermediates
design (meta.path_traversal=None, hop-2 ~= whole proteome). The |t|-transform
comparison is graph-design-agnostic (same |t|, same shells; only transform
changes), so this validly bounds log2-vs-raw sensitivity even though it is not
the current measured-only graph.
"""
import sys, json, warnings, argparse
warnings.filterwarnings("ignore")
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from cliquefinder.panels.seed_runner import load_panel_inputs
from cliquefinder.panels.landscape import FeatureDistanceMatrix
from cliquefinder.stats.rotation import RotationTestEngine
from cliquefinder.stats.perturbation_gradient import run_gradient_test

ROOT = Path(".")

def resolve(metadata):
    c9 = metadata[(metadata["ClinReport_Mutations_Details"]=="C9orf72")|(metadata["C9orf72_repeat_length"]>=30)]
    known = ["C9orf72","SOD1","FUS","TARDBP","TARDBP (TDP43)","SETX","Multiple","Other"]
    spor = metadata[(metadata["phenotype"]=="CASE")&(~metadata["ClinReport_Mutations_Details"].isin(known))&((metadata["C9orf72_repeat_length"]<30)|metadata["C9orf72_repeat_length"].isna())]
    ctrl = metadata[metadata["phenotype"]=="CTRL"]
    return {"C9ORF72":c9.index,"SPORADIC":spor.index,"CONTROL":ctrl.index}

def abs_t(mat, feat, sm, c1, c2):
    eng = RotationTestEngine(mat.copy(), list(feat), sm.copy())
    eng.fit(conditions=[c1,c2], contrast=(c1,c2), condition_column="_condition", covariates=["Sex"])
    ef = eng._effects
    se = np.sqrt(ef.moderated_variances if ef.moderated_variances is not None else ef.sample_variances)
    se = np.where(se>0, se, np.nan)
    t = ef.U[:,0]/se
    return {g: float(abs(t[i])) for i,g in enumerate(ef.gene_ids) if np.isfinite(t[i])}

def slopes(abs_t_per_feature, dm, graph_degrees, max_hops, nperm, rng_base=42):
    measured = sorted(abs_t_per_feature.keys())
    out = {}
    for fid in measured:
        try:
            sd = dm.distances_from(fid)
        except KeyError:
            continue
        shells = {}
        for tgt,d in sd.items():
            if d>=1 and tgt!=fid and (max_hops is None or d<=max_hops):
                shells.setdefault(d,set()).add(tgt)
        if not shells:
            continue
        ats = {k:v for k,v in abs_t_per_feature.items() if k!=fid}
        import hashlib
        srng = rng_base + int.from_bytes(hashlib.md5(fid.encode()).digest()[:4],"big")
        # Mirror production _per_feature_gradient_loop error handling: re-raise
        # resource-exhaustion, bucket all other anchor-level errors as skipped
        # (production records them as "errored"; e.g. the <10-measurable-genes
        # guardrail on sparse measured-only neighborhoods).
        try:
            r = run_gradient_test(adjacency={}, abs_t_stats=ats, seed=fid, max_hops=max_hops,
                                  n_permutations=nperm, rng_seed=srng, precomputed_shells=shells, verbose=False,
                                  graph_degrees=graph_degrees)
        except (MemoryError, RecursionError):
            raise
        except Exception:
            continue
        out[fid] = float(r.slope)
    return out

def emit_result_json(abs_t_per_feature, dm, graph_degrees, max_hops, nperm, design, out_dir, rng_base=42):
    """Run the EXACT production loop and serialize a full result.json (with shells) for GSEA."""
    import hashlib
    from cliquefinder.panels.landscape import LANDSCAPE_FEATURE_STRATUM_LABEL
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    per_feature = []
    for fid in sorted(abs_t_per_feature.keys()):
        try:
            sd = dm.distances_from(fid)
        except KeyError:
            continue
        shells = {}
        for tgt,d in sd.items():
            if d>=1 and tgt!=fid and (max_hops is None or d<=max_hops):
                shells.setdefault(d,set()).add(tgt)
        if not shells:
            continue
        ats = {k:v for k,v in abs_t_per_feature.items() if k!=fid}
        srng = rng_base + int.from_bytes(hashlib.md5(fid.encode()).digest()[:4],"big")
        # Mirror production error handling (see slopes()): skip anchors that
        # raise (e.g. <10-measurable-genes guardrail), re-raise resource errors.
        try:
            r = run_gradient_test(adjacency={}, abs_t_stats=ats, seed=fid, max_hops=max_hops,
                                  n_permutations=nperm, rng_seed=srng, precomputed_shells=shells,
                                  verbose=False, graph_degrees=graph_degrees)
        except (MemoryError, RecursionError):
            raise
        except Exception:
            continue
        per_feature.append({
            "seed": fid, "stratum": LANDSCAPE_FEATURE_STRATUM_LABEL,
            "slope": float(r.slope), "slope_pvalue": float(r.slope_pvalue),
            "spearman_rho": float(r.spearman_rho), "spearman_pvalue": float(r.spearman_pvalue),
            "shells": [{"hop":int(s.hop),"n_genes":int(s.n_genes),
                        "mean_abs_t":float(s.mean_abs_t),"median_abs_t":float(s.median_abs_t)}
                       for s in r.shells],
            "n_genes_total": int(r.n_genes_total)})
    # Make the nested design self-consistent: this emitter recomputes |t|
    # under log2(x+1), so stamp transform="log2" INTO the design block (not
    # just top-level), else LandscapeDesign.from_dict(file["design"]) would
    # default the copied raw design to "raw" and mislabel a log2 file.
    design = {**design, "transform": "log2"}
    json.dump({"design": design, "per_feature": per_feature, "degenerate_features": [],
               "error_features": [], "transform": "log2(x+1)", "n_permutations": nperm},
              open(out_dir/"result.json","w"))
    print(f"  emitted {out_dir}/result.json  ({len(per_feature)} per_feature, nperm={nperm})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landscape-dir", required=True)
    ap.add_argument("--c1", required=True); ap.add_argument("--c2", required=True)
    ap.add_argument("--nperm", type=int, default=0)
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--emit-log2-dir", help="write a full log2 result.json here for GSEA")
    ap.add_argument("--out")
    a = ap.parse_args()
    d = ROOT/a.landscape_dir
    meta = json.load(open(d/"distances.meta.json"))
    max_hops = meta["max_hops"]; gdeg = meta.get("graph_degrees",{})
    dm = FeatureDistanceMatrix.load_npz(d/"distances.npz")
    cached = {r["seed"]: r["slope"] for r in json.load(open(d/"result.json"))["per_feature"]}

    data, feat, mdata, groups = load_panel_inputs(
        data_path=ROOT/"output/proteomics/all_als.data.csv",
        metadata_path=ROOT/"output/proteomics/all_als.metadata.csv",
        group_resolver=resolve)
    c1,c2 = a.c1,a.c2
    keep = groups[c1].union(groups[c2])
    sm = mdata.loc[mdata.index.intersection(keep)].copy(); sm["_condition"]=None
    sm.loc[sm.index.isin(groups[c1]),"_condition"]=c1; sm.loc[sm.index.isin(groups[c2]),"_condition"]=c2
    sm = sm.dropna(subset=["_condition"])
    idx={s:i for i,s in enumerate(mdata.index)}; ai=[idx[s] for s in sm.index]; sub=data[:,ai]
    print(f"contrast {c1} vs {c2}: n={sub.shape[1]}, max_hops={max_hops}, cached anchors={len(cached)}")

    at_raw = abs_t(sub, feat, sm, c1, c2)
    sl_raw = slopes(at_raw, dm, gdeg, max_hops, a.nperm)
    common = [g for g in sl_raw if g in cached]
    diffs = np.array([abs(sl_raw[g]-cached[g]) for g in common])
    print(f"\n=== FAITHFULNESS GATE (raw recompute vs cached result.json) ===")
    print(f"anchors compared: {len(common)}  max|Δslope|={diffs.max():.3e}  mean|Δ|={diffs.mean():.3e}")
    print(f"  r(raw_recompute, cached) = {pearsonr([sl_raw[g] for g in common],[cached[g] for g in common])[0]:.6f}")
    GATE = diffs.max() < 1e-6
    print(f"  GATE {'PASS' if GATE else 'FAIL'} (threshold max|Δ|<1e-6)")
    if a.gate_only or not GATE:
        sys.exit(0 if GATE else 1)

    sub_log = np.log2(sub+1.0)
    at_log = abs_t(sub_log, feat, sm, c1, c2)
    sl_log = slopes(at_log, dm, gdeg, max_hops, a.nperm)
    c2c = [g for g in sl_raw if g in sl_log]
    ar=np.array([sl_raw[g] for g in c2c]); al=np.array([sl_log[g] for g in c2c])
    print(f"\n=== RAW vs LOG2 slopes ({len(c2c)} anchors) ===")
    print(f"  Pearson  = {pearsonr(ar,al)[0]:.3f}   Spearman = {spearmanr(ar,al)[0]:.3f}")
    for k in (50,100,200):
        sr=set(np.array(c2c)[np.argsort(-ar)[:k]]); slg=set(np.array(c2c)[np.argsort(-al)[:k]])
        print(f"  top-{k} steepest-anchor overlap: {len(sr&slg)}/{k}  Jaccard {len(sr&slg)/len(sr|slg):.3f}")
    if a.out:
        json.dump({"raw":sl_raw,"log2":sl_log}, open(a.out,"w"))
        print(f"  wrote {a.out}")
    if a.emit_log2_dir:
        design = json.load(open(d/"result.json"))["design"]
        # GSEA consumes only `slope` (score=-slope), so nperm=0 is fine here (slope_pvalue unused).
        emit_result_json(at_log, dm, gdeg, max_hops, a.nperm, design, a.emit_log2_dir)

if __name__=="__main__":
    main()
