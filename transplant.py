#%% Setup: data, model, tokenizer

import json
import math
import random

import torch as t
from transformers import FineGrainedFP8Config

from mechtools import *

from utils import MODEL_ID, tiny_bridge, donors, readout_numbers, valid_donors, prompt_ids, regions, answer_ids, prefill, transplant

t.set_grad_enabled(False)
items = [json.loads(l) for l in open("data/eval.jsonl")]
shots = [json.loads(l) for l in open("data/fewshot.jsonl")]

TAG = "tiny"
model = tiny_bridge()
# TAG = "v4flash"
# model = load_bridge(MODEL_ID)  # packed FP8 + FP4 experts, ~160 GB; needs `uv add kernels`
# model = load_bridge(MODEL_ID, quantization_config=FineGrainedFP8Config(dequantize=True))  # bf16, ~569 GB
tok = model.tokenizer
n_layers = model.cfg.n_layers

# Source rows of every transplant batch: row 0 is the clean target, then one donor per edit kind.
DONOR_KINDS = ["xk2", "x", "k1", "k2"]
# Readout outcomes by where y and k2 come from: per donor kind, the readout number computed from that combination (None when the donor
# cannot produce it). The x and k1 donors' own answers are (donor y, target k2); the k2 donor's is (target y, donor k2).
OUTCOMES = {
    "target y, target k2": {"xk2": "a_T", "x": "a_T", "k1": "a_T", "k2": "a_T"},
    "donor y, target k2": {"xk2": "ty", "x": "ty", "k1": "a_k1", "k2": None},
    "donor y, donor k2": {"xk2": "a_xk2", "x": None, "k1": None, "k2": None},
    "target y, donor k2": {"xk2": "a_k2", "x": None, "k1": None, "k2": "a_k2"},
}

#%% Pick transplant targets from the benchmark (high P(correct) at 100 dots; flip vs always-right by P at 0 dots) and one valid donor draw (x', k1', k2') each; saves results/families_{TAG}.json

pick_families = True
if pick_families:
    k = 100
    min_p_filler = 0.6
    # min_p_filler = 0.0  # tiny model
    flip_max_p0 = 0.3
    always_min_p0 = 0.7
    n_per_group = 10
    # n_per_group = 100
    seed = 0
    rng = random.Random(seed)
    bench = {(r["idx"], r["k"]): math.exp(r["logp"]) for r in json.load(open(f"results/bench_{TAG}.json"))}
    pool = {"flip": [], "always": []}
    draw = {}
    for it in items:
        i = it["idx"]
        if (i, k) not in bench or bench[(i, k)] < min_p_filler or it["answer"] >= 1000:
            continue
        candidates = ((rng.randint(10, 99), rng.randint(1, 50), rng.randint(1, 50)) for _ in range(10_000))
        draw[i] = next((d for d in candidates if valid_donors(it, donors(it, *d))), None)
        if draw[i] is None:  # e.g. item 253: no k2' keeps a_k2 non-negative and 5 away from a_T
            continue
        if bench[(i, 0)] < flip_max_p0:
            pool["flip"].append(i)
        if bench[(i, 0)] > always_min_p0:
            pool["always"].append(i)
    print(f"{yellow}{sum(d is None for d in draw.values())} items have no valid donor draw{endc}")
    families = []
    for group, idxs in pool.items():
        for i in sorted(rng.sample(idxs, min(n_per_group, len(idxs)))):
            x, k1, k2 = draw[i]
            families.append({"idx": i, "group": group, "x": x, "k1": k1, "k2": k2})
        print(f"{cyan}{group}: {len(idxs)} eligible, {sum(f['group'] == group for f in families)} picked{endc}")
    json.dump(families, open(f"results/families_{TAG}.json", "w"))
    print(f"{green}saved results/families_{TAG}.json{endc}")

#%% Gates on the first family, within one batch: no-op rows reproduce the clean target, and a transplant of every position but the last reproduces the donor

check_gates = True
if check_gates:
    k = 100
    tol = 1e-3
    families = json.load(open(f"results/families_{TAG}.json"))
    fam = families[0]
    it = items[fam["idx"]]
    dons = donors(it, fam["x"], fam["k1"], fam["k2"])
    prefix, tail = prompt_ids(tok, shots, it, k)
    src_tails = [tail] + [prompt_ids(tok, shots, dons[kind], k)[1] for kind in DONOR_KINDS]
    reg = regions(tok, tail, k)
    F = reg["filler"]
    everything = list(range(len(tail) - 1))
    scopes = [
        {"src": 1, "pos": [], "layers": [], "frozen": []},
        {"src": 1, "pos": [], "layers": [], "frozen": F},
        {"src": 1, "pos": everything, "layers": range(n_layers), "frozen": []},
        {"src": 1, "pos": F, "layers": range(n_layers), "frozen": F},
    ]
    cache = prefill(model, prefix)
    logp = transplant(model, cache, src_tails, scopes)
    n_src = len(src_tails)
    for name, row, ref in [("none", n_src, 0), ("filler frozen to target", n_src + 1, 0), ("all but last from donor", n_src + 2, 1)]:
        d = (logp[row] - logp[ref]).abs().max().item()
        print(f"{cyan}{name} vs source row {ref}: max |dlogp| {d:.1e} (clean target vs donor {(logp[0] - logp[1]).abs().max():.2f}){endc}")
        assert d < tol
    alone = transplant(model, cache, src_tails, scopes[3:])[n_src]
    print(f"{cyan}filler scope alone vs in a batch of {n_src + len(scopes)}: max |dlogp| {(alone - logp[n_src + 3]).abs().max():.1e} (cross-batch noise, not gated){endc}")

#%% Transplant sweep: for each family, scopes of regions, filler prefixes/suffixes and layer ranges, sourced from the x+k2 donor (all scopes) and the x-, k1- and k2-only donors (region scopes); saves results/transplant_{TAG}.json

run_transplant = True
if run_transplant:
    k = 100
    batch_size = 32
    ns = [10, 25, 50, 75]
    layer_cuts = sorted({round(n_layers * f / 6) for f in range(1, 6)})
    control_scopes = ["filler", "filler_post_frozen", "post", "label", "q_defs", "q_line", "first_50", "last_50"]
    families = json.load(open(f"results/families_{TAG}.json"))
    # families = families[:4]
    prefix = prompt_ids(tok, shots, items[0], k)[0]
    cache = prefill(model, prefix)
    records = []
    for fam in pbar(families, desc="families"):
        it = items[fam["idx"]]
        dons = donors(it, fam["x"], fam["k1"], fam["k2"])
        pre, tail = prompt_ids(tok, shots, it, k)
        assert pre == prefix
        src_tails = [tail] + [prompt_ids(tok, shots, dons[kind], k)[1] for kind in DONOR_KINDS]
        assert all(len(s) == len(tail) for s in src_tails)
        reg = regions(tok, tail, k)
        F, post, label, q_defs, q_line = reg["filler"], reg["post"], reg["label"], reg["q_defs"], reg["q_line"]
        all_layers = range(n_layers)
        by_name = {  # name: (positions from the donor, layers, positions frozen to the target)
            "none": ([], [], []),
            "frozen_target": ([], [], F),
            "all_but_last": (list(range(len(tail) - 1)), all_layers, []),
            "filler": (F, all_layers, F),
            "filler_post_frozen": (F, all_layers, F + post),  # Brauer et al.'s cache-row swap: only the answer position recomputes
            "post": (post, all_layers, []),
            "label": (label, all_layers, F),
            "q_defs": (q_defs, all_layers, q_line + label + F),
            "q_line": (q_line, all_layers, label + F),
            "question": (q_defs + q_line, all_layers, label + F),
            "filler_question": (q_defs + q_line + label + F, all_layers, []),
            **{f"first_{n}": (F[:n], all_layers, F) for n in ns},
            **{f"last_{n}": (F[-n:], all_layers, F) for n in ns},
            **{f"first_{n}_free": (F[:n], all_layers, []) for n in ns},
            **{f"layers_lt_{c}": (F, range(c), F) for c in layer_cuts},
            **{f"layers_ge_{c}": (F, range(c, n_layers), F) for c in layer_cuts},
        }
        scopes = [{"name": name, "kind": "xk2", "src": 1, "pos": pos, "layers": layers, "frozen": frozen} for name, (pos, layers, frozen) in by_name.items()]
        scopes += [{"name": name, "kind": kind, "src": 1 + DONOR_KINDS.index(kind), "pos": by_name[name][0], "layers": by_name[name][1], "frozen": by_name[name][2]} for kind in DONOR_KINDS[1:] for name in control_scopes]
        read = {name: answer_ids(tok, n)[0] for name, n in readout_numbers(it, dons).items()}
        read_name = {tid: name for name, tid in read.items()}
        read_ids = list(read.values())
        n_src = len(src_tails)
        chunk = batch_size - n_src
        for b, i in enumerate(range(0, len(scopes), chunk)):
            batch = scopes[i:i + chunk]
            logp = transplant(model, cache, src_tails, batch)
            clean = [dict(zip(read, row.tolist())) for row in logp[:n_src, read_ids]]
            for j, sc in enumerate(batch):
                top = logp[n_src + j].argmax().item()
                records.append({
                    "idx": fam["idx"],
                    "group": fam["group"],
                    "scope": sc["name"],
                    "kind": sc["kind"],
                    "logp": dict(zip(read, logp[n_src + j, read_ids].tolist())),
                    "top": read_name.get(top, "other"),
                    "clean": clean,
                    "batch": b,
                })
    json.dump(records, open(f"results/transplant_{TAG}.json", "w"))
    print(f"{green}saved {len(records)} scope rows for {len(families)} families to results/transplant_{TAG}.json{endc}")

#%% Load transplant records, keeping families whose clean donors give their own answers (source row d + 1 is DONOR_KINDS[d])

min_donor_p = 0.6
# min_donor_p = 0.0  # tiny model
records = json.load(open(f"results/transplant_{TAG}.json"))
own = {"xk2": "a_xk2", "x": "ty", "k1": "a_k1", "k2": "a_k2"}
ok = {r["idx"] for r in records if r["scope"] == "none" and all(math.exp(r["clean"][1 + d][own[kind]]) >= min_donor_p for d, kind in enumerate(DONOR_KINDS))}
print(f"{cyan}{len(ok)} of {len({r['idx'] for r in records})} families have all four donors giving their own answer with P >= {min_donor_p}{endc}")
records = [r for r in records if r["idx"] in ok]

#%% Mean probability at the answer position by transplanted region and donor kind, stacked by where the answer's y and k2 come from; one figure per target group

plot_regions = True
if plot_regions:
    bars = [("xk2", s) for s in ["none", "filler", "filler_post_frozen", "post", "label", "q_defs", "q_line", "question", "filler_question"]]
    bars += [(kind, s) for kind in DONOR_KINDS[1:] for s in ["filler", "q_defs", "q_line"]]
    for group in ["flip", "always"]:
        rows = {b: [r for r in records if r["group"] == group and (r["kind"], r["scope"]) == b] for b in bars}
        if not rows[bars[0]]:
            continue
        mean_p = lambda rs, name: sum(math.exp(r["logp"][name]) for r in rs) / len(rs) if name else 0.0
        series = [[mean_p(rows[b], OUTCOMES[outcome][b[0]]) for b in bars] for outcome in OUTCOMES]
        rest = [1 - sum(col) for col in zip(*series)]
        bar(
            series + [rest],
            x=[f"{kind}: {scope}" for kind, scope in bars],
            names=list(OUTCOMES) + ["rest"],
            labels={"x": "donor: transplanted scope", "y": "mean probability at the answer position"},
            title=f"Filler transplant, {group} targets (n={len(rows[bars[0]])})",
            barmode="stack",
            legend_title_text="",
        )

#%% Where in the filler: for the x+k2 donor, P(donor y, target k2) and P(donor y, donor k2) vs the number of donor filler positions (first N frozen, last N frozen, first N free)

plot_positions = True
if plot_positions:
    ns = sorted({int(r["scope"].split("_")[1]) for r in records if r["scope"].startswith("last_")})
    for group in ["flip", "always"]:
        rows = [r for r in records if r["group"] == group and r["kind"] == "xk2"]
        if not rows:
            continue
        mean_p = lambda scope, name: sum(math.exp(r["logp"][name]) for r in rows if r["scope"] == scope) / sum(r["scope"] == scope for r in rows)
        series, names = [], []
        for outcome in ["donor y, target k2", "donor y, donor k2"]:
            for fmt, how in [("first_{}", "first N frozen"), ("last_{}", "last N frozen"), ("first_{}_free", "first N free")]:
                name = OUTCOMES[outcome]["xk2"]
                series.append([mean_p("none", name)] + [mean_p(fmt.format(n), name) for n in ns] + [mean_p("filler", name)])
                names.append(f"P({outcome}), {how}")
        line(
            series,
            x=[0] + ns + [100],
            names=names,
            labels={"x": "donor filler positions N", "y": "mean probability at the answer position"},
            title=f"Filler transplant by position, {group} targets (n={len({r['idx'] for r in rows})})",
            markers=True,
            legend_title_text="",
        )

#%% When: for the x+k2 donor, P(donor y, target k2) and P(donor y, donor k2) when the whole filler is transplanted only at layers < L or only at layers >= L

plot_layers = True
if plot_layers:
    cuts = sorted({int(r["scope"].split("_")[-1]) for r in records if r["scope"].startswith("layers_ge_")})
    for group in ["flip", "always"]:
        rows = [r for r in records if r["group"] == group and r["kind"] == "xk2"]
        if not rows:
            continue
        mean_p = lambda scope, name: sum(math.exp(r["logp"][name]) for r in rows if r["scope"] == scope) / sum(r["scope"] == scope for r in rows)
        outcomes = ["donor y, target k2", "donor y, donor k2"]
        line(
            [[mean_p(f"layers_{side}_{c}", OUTCOMES[outcome]["xk2"]) for c in cuts] for outcome in outcomes for side in ["lt", "ge"]],
            x=cuts,
            names=[f"P({outcome}), layers {side} L" for outcome in outcomes for side in ["<", ">="]],
            labels={"x": "layer cut L", "y": "mean probability at the answer position"},
            title=f"Filler transplant by layer, {group} targets (n={len({r['idx'] for r in rows})})",
            markers=True,
            legend_title_text="",
        )
