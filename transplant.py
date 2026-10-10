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
    per_item = f"results/dose_{TAG}.json"  # P(correct) per item and k from the filler-length sweep in bench.py
    # per_item = f"results/bench_{TAG}.json"
    rng = random.Random(seed)
    bench = {(r["idx"], r["k"]): math.exp(r["logp"]) for r in json.load(open(per_item))}
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
    blocks = [(25, 50), (50, 75), (25, 75)]  # inner dot ranges, the rest of the filler frozen to the target
    layer_cuts = sorted({round(n_layers * f) for f in [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 0.75, 0.8, 0.88, 0.93]})  # 7, 14, 22, 29, 32, 34, 36, 38, 40 of 43
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
            **{f"dots_{a}_{b}": (F[a:b], all_layers, F) for a, b in blocks},
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

#%% Load transplant records; a scope row counts only if its donor, run clean in the same batch, gives its own answer with P >= min_donor_p

load_records = True
if load_records:
    min_donor_p = 0.6
    # min_donor_p = 0.0  # tiny model
    n_boot = 2000
    own = {"xk2": "a_xk2", "x": "ty", "k1": "a_k1", "k2": "a_k2"}  # each donor's own answer among the readout numbers
    records = json.load(open(f"results/transplant_{TAG}.json"))
    n_fam = {kind: len({r["idx"] for r in records if r["kind"] == kind}) for kind in DONOR_KINDS}
    records = [r for r in records if math.exp(r["clean"][1 + DONOR_KINDS.index(r["kind"])][own[r["kind"]]]) >= min_donor_p]
    for kind in DONOR_KINDS:
        print(f"{cyan}{kind} donor kept for {len({r['idx'] for r in records if r['kind'] == kind})} of {n_fam[kind]} families (clean donor P(own answer) >= {min_donor_p}){endc}")
    GROUPS = {"always": "Always-right questions (correct with and without filler)", "flip": "Flipped questions (wrong without filler, right with 100 dots)"}
    KIND_NAMES = {"xk2": "donor differs in x and k₂", "x": "donor differs in x", "k1": "donor differs in k₁", "k2": "donor differs in k₂"}
    rows_of = lambda group, kind, scope: [r for r in records if r["group"] == group and r["kind"] == kind and r["scope"] == scope]
    def stat(rows: list[dict], name: str) -> tuple[float, float, float]:  # mean P(readout `name`) over families (one row each) with a 95% bootstrap interval
        p = t.tensor([math.exp(r["logp"][name]) for r in rows])
        boot = p[t.randint(0, len(p), (n_boot, len(p)), generator=t.Generator().manual_seed(0))].mean(1)
        return p.mean().item(), boot.quantile(0.025).item(), boot.quantile(0.975).item()
    INTRO = (
        "<br><span style='font-size:13px'>Task: two-step arithmetic in words. A number x is given, y = c₁·x ± k₁ is defined, and the question asks for c₂·y ± k₂. 100 filler dots sit between the question and 'Answer:'.</span>"
        "<br><span style='font-size:13px'>Transplant: the model runs on a target question; at the chosen token positions and layers its internal state is replaced by the state from a run on a donor question,</span>"
        "<br><span style='font-size:13px'>which differs from the target in one or two numbers, so the donor's y and answer differ. Read at the answer position: the probability of the target's own answer, of the donor's answer,</span>"
        "<br><span style='font-size:13px'>and (x + k₂ donor) of the two mixed values: the donor's y finished with the target's k₂, and the target's y with the donor's k₂. None of these numbers is written in either prompt.</span>"
    )

#%% Figure: which prompt region carries the answer. Mean probability at the answer position by transplanted region and donor kind, stacked by outcome; one panel per target group

plot_regions = True
if plot_regions:
    scope_names = {"none": "nothing<br>(control)", "filler": "all 100 dots", "filler_post_frozen": "all 100 dots,<br>'Answer:' tokens<br>kept from target", "post": "'Answer:'<br>tokens only", "label": "'Filler:'<br>label only", "q_defs": "definition<br>lines", "q_line": "question<br>line", "question": "definitions +<br>question", "filler_question": "question +<br>dots", "all_but_last": "every token<br>but the last"}
    bars = [("xk2", s) for s in scope_names] + [(kind, s) for kind in DONOR_KINDS[1:] for s in ["filler", "q_defs", "q_line"]]
    outcomes = {  # stack component: readout name per donor kind; the donor's answer goes first so its error bar sits at the bar's own height
        "donor's answer": {"xk2": "a_xk2", "x": "ty", "k1": "a_k1", "k2": "a_k2"},
        "target's answer": {k: "a_T" for k in DONOR_KINDS},
        "donor's y, finished with the target's k₂": {"xk2": "ty"},
        "target's y with the donor's k₂": {"xk2": "a_k2"},
    }
    colors = [SERIES[3], SERIES[0], SERIES[1], SERIES[2], "#555555"]
    fig = make_subplots(rows=2, cols=1, subplot_titles=list(GROUPS.values()), vertical_spacing=0.2)
    for col, group in enumerate(GROUPS, 1):
        n = {kind: len(rows_of(group, kind, "filler")) for kind in DONOR_KINDS}
        x = [[f"{KIND_NAMES[kind]} (n={n[kind]})" for kind, _ in bars], [scope_names[s] for _, s in bars]]
        stacks = []
        for (name, by_kind), color in zip(outcomes.items(), colors):
            st = [stat(rows_of(group, kind, scope), by_kind[kind]) if kind in by_kind else (0.0, 0.0, 0.0) for kind, scope in bars]
            m, lo, hi = zip(*st)
            err = {"type": "data", "symmetric": False, "array": [h - v for v, h in zip(m, hi)], "arrayminus": [v - l for v, l in zip(m, lo)]} if name == "donor's answer" else None
            fig.add_trace(go.Bar(x=x, y=m, name=name, marker_color=color, error_y=err, showlegend=col == 1, legendgroup=name), row=col, col=1)
            stacks.append(m)
        fig.add_trace(go.Bar(x=x, y=[1 - sum(v) for v in zip(*stacks)], name="other", marker_color=colors[-1], showlegend=col == 1, legendgroup="other"), row=col, col=1)
    pa = {g: stat(rows_of(g, "xk2", "filler"), "a_xk2")[0] for g in GROUPS}
    pt = {g: stat(rows_of(g, "xk2", "none"), "a_T")[0] for g in GROUPS}
    fig.update_layout(
        title={"y": 0.975, "yanchor": "top", "text": (
            f"<b>Swapping in a donor question's filler-dot states makes the model give the donor's answer: probability {pa['always']:.2f} on always-right questions, {pa['flip']:.2f} on flipped ones</b>"
            f"<br>Untouched, the target's own answer has probability {pt['always']:.2f} and {pt['flip']:.2f}.  (DeepSeek V4 Flash, 100 filler dots)"
            + INTRO
            + "<br><span style='font-size:13px'>Bars: which tokens come from the donor (at every layer). Dots not taken from the donor keep the target's own states; the 'Answer:' tokens and the answer position are recomputed unless stated.</span>"
            + "<br><span style='font-size:13px'>n = questions whose donor, run on its own, gives its own answer with probability ≥ 0.6. Error bars: 95% intervals from resampling questions.</span>"
        )},
        barmode="stack",
        height=1300,
        width=1900,
        margin={"t": 300, "b": 120},
        legend={"title": "the model's answer is"},
        **DARK,
    )
    fig.update_yaxes(title_text="mean probability at the answer position", range=[0, 1.02])
    fig.update_xaxes(tickangle=0, tickfont={"size": 11})
    write_dark_html(fig, "figs/transplant_regions.html")
    print(f"{green}wrote figs/transplant_regions.html{endc}")

#%% Figure: which dots and which layers carry the answer. P(donor's answer) when only a range of the dots comes from the x+k2 donor (the other dots kept from the target or, first-N free, recomputed), and when all 100 dots come from it only at layers < L or only at layers >= L

plot_where = True
if plot_where:
    kind = "xk2"
    ranges = {}  # scope: (first dot, last dot, later dots recomputed)
    for s in {r["scope"] for r in records if r["kind"] == kind}:
        if s == "filler":
            ranges[s] = (0, 100, False)
        elif s.startswith("first_"):
            ranges[s] = (0, int(s.split("_")[1]), s.endswith("_free"))
        elif s.startswith("last_"):
            ranges[s] = (100 - int(s.split("_")[1]), 100, False)
        elif s.startswith("dots_"):
            ranges[s] = (*map(int, s.split("_")[1:]), False)
    order = sorted(ranges, key=lambda s: (ranges[s][2], ranges[s][0], -ranges[s][1]))
    labels = [f"dots {a}–{b}" + (", later dots recomputed" if free else "") for a, b, free in (ranges[s] for s in order)]
    cuts = sorted({int(r["scope"].split("_")[-1]) for r in records if r["scope"].startswith("layers_ge_")})
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{g}<br>{what}" for what in ["donor dots by range, at every layer", "all 100 donor dots, by layer cut"] for g in GROUPS.values()], shared_yaxes=True, horizontal_spacing=0.03, vertical_spacing=0.1, row_heights=[0.55, 0.45])
    for col, group in enumerate(GROUPS, 1):
        st = [stat(rows_of(group, kind, s), "a_xk2") for s in order]
        m, lo, hi = zip(*st)
        fig.add_trace(go.Bar(
            y=labels, x=[ranges[s][1] - ranges[s][0] for s in order], base=[ranges[s][0] for s in order], orientation="h",
            marker={"color": m, "colorscale": "Viridis", "cmin": 0, "cmax": 1, "showscale": col == 1, "colorbar": {"title": "P(donor's<br>answer)", "x": 1.01, "y": 0.8, "len": 0.4}},
            text=[f"{v:.2f} [{l:.2f}, {h:.2f}]" for v, l, h in st], textposition="outside", cliponaxis=False, showlegend=False,
            customdata=[len(rows_of(group, kind, s)) for s in order], hovertemplate="%{y}<br>P(donor's answer) = %{text}<br>n = %{customdata}<extra></extra>",
        ), row=1, col=col)
        for side, color, side_name in [("lt", SERIES[0], "layers below L only"), ("ge", SERIES[3], "layer L and above only")]:
            for name, dash, label in [("a_xk2", "solid", "donor's answer"), ("a_T", "dot", "target's answer")]:
                m, lo, hi = zip(*[stat(rows_of(group, kind, f"layers_{side}_{c}"), name) for c in cuts])
                fig.add_trace(go.Scatter(x=cuts + cuts[::-1], y=list(hi) + list(lo)[::-1], fill="toself", fillcolor=color, opacity=0.12, line={"width": 0}, showlegend=False, hoverinfo="skip"), row=2, col=col)
                fig.add_trace(go.Scatter(x=cuts, y=m, mode="lines+markers", line={"color": color, "dash": dash}, name=f"{label}: donor dots at {side_name}", showlegend=col == 1, legendgroup=f"{side}{name}"), row=2, col=col)
    p = lambda s: stat(rows_of("always", kind, s), "a_xk2")[0]
    c = cuts[len(cuts) // 2]
    fig.update_layout(
        title={"y": 0.975, "yanchor": "top", "text": (
            f"<b>Which dots and which layers carry the donor's answer: the first 50 dots alone give probability {p('first_50'):.2f}, the last 50 alone {p('last_50'):.2f}, all 100 {p('filler'):.2f};</b>"
            f"<br><b>all 100 dots only at layers ≥ {c} give {p(f'layers_ge_{c}'):.2f}, only at layers < {c} give {p(f'layers_lt_{c}'):.2f}</b>  (always-right questions; DeepSeek V4 Flash, 43 layers, 100 filler dots)"
            + INTRO
            + "<br><span style='font-size:13px'>Top: each bar spans the dots taken from the x + k₂ donor at every layer; the other dots keep the target's own states or, in the last rows, are recomputed on top of the donor's dots. Color and text: mean probability of the donor's answer [95% interval from resampling questions].</span>"
            + "<br><span style='font-size:13px'>Bottom: all 100 dots come from the donor, but only at the layers below the cut L (blue) or only at L and above (orange); at the other layers the dots keep the target's own states. Bands: 95% intervals.</span>"
        )},
        height=1500,
        width=1700,
        margin={"t": 310, "l": 250},
        legend={"title": "the model's answer is", "y": 0.3},
        **DARK,
    )
    fig.update_yaxes(autorange="reversed", row=1)
    fig.update_xaxes(title_text="filler dot index (0 = first dot after 'Filler:')", range=[0, 128], tickvals=[0, 25, 50, 75, 100], row=1)
    fig.update_xaxes(title_text="layer cut L", tickvals=cuts, row=2)
    fig.update_yaxes(title_text="mean probability at the answer position", range=[0, 1.02], row=2, col=1)
    write_dark_html(fig, "figs/transplant_where.html")
    print(f"{green}wrote figs/transplant_where.html{endc}")
