#!/usr/bin/env python
"""Q1 summary: LoRAS-T at text positions vs the same corrector at all positions.

Reads the ladder report (every rung vs FP16), the pairwise report (all vs text), the
object analysis and the two calibration diagnostics; prints one table and writes
q1_summary.json plus a trimmed ladder (only the four rungs, with captions). CPU only.
"""
import argparse
import json
from pathlib import Path

METRICS = [("coverage", "dCov"), ("chair_i", "dCHAIR_i"), ("pope_f1", "dF1"),
           ("pope_yes", "dyes"), ("pope_auroc", "dAUROC"), ("pope_recall", "drecall"),
           ("length", "dlen")]


def cell(c):
    if not c:
        return "-"
    s = f"{c['delta']:+.3f}{'*' if c.get('significant') else ' '}"
    if "lo" in c and "hi" in c:
        s += f" [{c['lo']:+.3f},{c['hi']:+.3f}]"
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True, help="ladder dir (copy of Study 2 + Q1 rungs)")
    p.add_argument("--base", default="w3a16")
    p.add_argument("--text-rung", required=True)
    p.add_argument("--all-rung", required=True)
    p.add_argument("--text-pt", required=True)
    p.add_argument("--all-pt", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    W, O = Path(a.work), Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    rep = json.loads((W / "ladder_report.json").read_text())["summary"]
    pw_path = W / "pairwise_all_vs_text" / "ladder_report.json"
    pw = json.loads(pw_path.read_text())["summary"] if pw_path.exists() else {}
    pw3_path = W / "pairwise_vs_base" / "ladder_report.json"
    pw3 = json.loads(pw3_path.read_text())["summary"] if pw3_path.exists() else {}
    obj = json.loads((W / "object_analysis.json").read_text()) if (W / "object_analysis.json").exists() else {}
    rows = {"W3 RTN": a.base, "+ LoRAS-T text (Q1 ref)": a.text_rung,
            "+ same, ALL positions": a.all_rung}
    lines, out = [], {"vs_fp16": {}, "all_minus_text": {}, "vs_base": {}, "object_drop": {},
                      "calibration": {}}
    lines.append("== change vs FP16 (paired, image-clustered 95% CI; * = excludes 0) ==")
    for lab, r in rows.items():
        cells = rep.get(r, {}).get("vs_reference", {})
        out["vs_fp16"][lab] = {m: cells.get(m) for m, _ in METRICS}
        lines.append(f"{lab:28s} " + "  ".join(f"{n}={cell(cells.get(m))}" for m, n in METRICS[:5]))
    lines.append("\n== ALL positions minus TEXT positions (same rank, same calibration data) ==")
    cells = pw.get(a.all_rung, {}).get("vs_reference", {})
    out["all_minus_text"] = {m: cells.get(m) for m, _ in METRICS}
    lines.append("  " + "  ".join(f"{n}={cell(cells.get(m))}" for m, n in METRICS))
    lines.append("\n== each correction minus W3 RTN ==")
    for lab, r in list(rows.items())[1:]:
        cells = pw3.get(r, {}).get("vs_reference", {})
        out["vs_base"][lab] = {m: cells.get(m) for m, _ in METRICS}
        lines.append(f"{lab:28s} " + "  ".join(f"{n}={cell(cells.get(m))}" for m, n in METRICS[:3]))
    if obj:
        lines.append("\n== share of FP16-mentioned objects dropped (small <1% / large >20% / all) ==")
        for lab, r in rows.items():
            d = obj.get("rungs", {}).get(r, {}).get("drop_rate", {})
            out["object_drop"][lab] = d
            lines.append(f"{lab:28s} " + " / ".join(f"{d.get(k, float('nan')):.3f}"
                                                     for k in ("<1%", ">20%", "all")))
    lines.append("\n== calibration (held-out, from the .json next to each .pt) ==")
    for lab, pt in (("text", a.text_pt), ("all", a.all_pt)):
        j = Path(pt).with_suffix(".json")
        if not j.exists():
            lines.append(f"{lab}: {j} missing")
            continue
        d = json.loads(j.read_text())
        m, L = d["meta"], d["layers"]
        vr = [v.get("val_red") for v in L.values() if v.get("val_red") is not None]
        info = {"positions": m.get("positions", "text"), "params": m["params"], "rank": m["rank"],
                "qa_per_image": m.get("qa_per_image"), "n_calib": m.get("n_calib"),
                "kl_before": m["kl_before"].get("kl"), "kl_after": m["kl_after"].get("kl"),
                "top1_before": m["kl_before"].get("top1_agree"),
                "top1_after": m["kl_after"].get("top1_agree"),
                "pyes_before": m["kl_before"].get("qa", {}).get("abs"),
                "pyes_after": m["kl_after"].get("qa", {}).get("abs"),
                "mean_val_red": sum(vr) / len(vr) if vr else None,
                "skipped_layers": [k for k, v in L.items() if v.get("skipped")]}
        out["calibration"][lab] = info
        lines.append(f"{lab:5s} positions={info['positions']} params={info['params']/1e6:.1f}M "
                     f"KL {info['kl_before']:.4f}->{info['kl_after']:.4f} "
                     f"top1 {info['top1_before']:.3f}->{info['top1_after']:.3f} "
                     f"mean val_red {info['mean_val_red'] if info['mean_val_red'] is None else round(info['mean_val_red'], 3)} "
                     f"skipped {info['skipped_layers']}")
    txt = "\n".join(lines)
    print(txt)
    (O / "q1_summary.txt").write_text(txt + "\n")
    (O / "q1_summary.json").write_text(json.dumps(out, indent=2))
    lad = json.loads((W / "ladder.json").read_text())
    keep = ["fp16"] + list(rows.values())
    trimmed = {"config": lad["config"], "table": {r: lad["table"][r] for r in keep if r in lad["table"]}}
    (O / "ladder_q1_rungs.json").write_text(json.dumps(trimmed))
    print(f"\n[q1] written {O}/q1_summary.txt, q1_summary.json, ladder_q1_rungs.json")


if __name__ == "__main__":
    main()
