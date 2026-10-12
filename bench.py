#%% Setup: data, model, tokenizer

import csv
import html
import json
import math
from collections import Counter

import torch as t
from transformers import FineGrainedFP8Config
from transformers.cache_utils import DynamicCache

from mechtools import *

from prompts import parse_answer
from utils import MODEL_ID, EOS, tiny_bridge, fp32_routers, prompt_ids, regions, answer_ids, prefill, tail_logits, decode, answer_logprob

UNSURE = "#c678dd"  # plot color for the questions the model is unsure about with no filler (P(correct) < 0.8 at 0 tokens)

def summarize(pk: Tensor, p0: Tensor, metric: str, n_boot: int = 5000) -> tuple[Tensor, Tensor, Tensor]:
    """Per filler length: the mean P(correct) under filler (`metric` 'p'; `pk` is [items] or [items, ks]) or the fraction of the no-filler error it removes ('frac': (mean pk - mean p0) / (1 - mean p0), `p0` [items], the same questions), with a 95% bootstrap interval over questions, paired. Returns (value, lo, hi), each [ks] or a scalar. NaN entries (lengths not run) are skipped."""
    pk2 = pk[:, None] if pk.dim() == 1 else pk
    f = (lambda a, b: a.nanmean(-2)) if metric == "p" else (lambda a, b: (a.nanmean(-2) - b.nanmean(-1)[..., None]) / (1 - b.nanmean(-1)[..., None]))
    ix = t.randint(0, len(p0), (n_boot, len(p0)), generator=t.Generator().manual_seed(0))
    v, boot = f(pk2, p0), f(pk2[ix], p0[ix])
    v, lo, hi = v, boot.quantile(0.025, 0), boot.quantile(0.975, 0)
    return (v.squeeze(-1), lo.squeeze(-1), hi.squeeze(-1)) if pk.dim() == 1 else (v, lo, hi)
METRIC = {  # metric -> (file suffix, axis title, hover label, y range, explanation)
    "p": ("", "mean probability of the correct answer", "mean P(correct)", [0, 1], ["Probability of the correct answer: the model's probability of writing exactly the right number and stopping (its expected accuracy when sampling). Bands or error bars: 95% intervals from resampling questions."]),
    "frac": ("_err", "fraction of the no-filler error removed", "fraction of error removed", [-0.35, 1], ["Fraction of error removed: (mean P(correct) with the filler − mean without) / (1 − mean without), each mean over the questions in the group, where P(correct) is the model's probability of writing exactly the right number and stopping.",
        "Questions the model already gets right add little to either mean, and groups with different baselines share one scale. Bands or error bars: 95% intervals from resampling questions."]),
}
def explain_lines(explain: list[str]) -> str:
    return "".join(f"<br><span style='font-size:13px'>{e}</span>" for e in explain)

t.set_grad_enabled(False)
items = [json.loads(l) for l in open("data/eval.jsonl")]
shots = [json.loads(l) for l in open("data/fewshot.jsonl")]
api = [json.loads(l) for l in open("data/phase2_full.jsonl")]
assert all(it["idx"] == i for i, it in enumerate(items))

TAG = "tiny"
model = tiny_bridge()
# TAG = "v4flash"
# model = load_bridge(MODEL_ID)  # packed FP8 + FP4 experts, ~160 GB; needs `uv add kernels`
# model = load_bridge(MODEL_ID, quantization_config=FineGrainedFP8Config(dequantize=True))  # bf16, ~569 GB
# model = load_bridge(MODEL_ID, quantization_config=FineGrainedFP8Config(dequantize=True), max_memory={i: "76GiB" for i in range(8)})  # 8x H100: a layer is 12.3 GiB and unsplittable, so 72GiB spills to disk
# fp32_routers(model)  # route in fp32 like DeepSeek's reference inference (HF routes in bf16)
tok = model.tokenizer

#%% Smoke test: parameter and stream dtypes, GPU memory, and the prefix-cache path against a plain full forward on a few items

smoke_test = True
if smoke_test:
    k = 100
    n_items = 8
    hf = model.original_model
    print(f"{cyan}param dtypes {Counter(p.dtype for p in hf.parameters())}, lm_head {hf.lm_head.weight.dtype}{endc}")
    for i in range(t.cuda.device_count()):
        print(f"{gray}cuda:{i} {t.cuda.memory_allocated(i) / 1e9:.1f} GB allocated{endc}")
    stream_dtypes = {}
    def grab_dtype(act, hook):
        stream_dtypes[hook.name] = act.dtype
    prefix, _ = prompt_ids(tok, shots, items[0], k)
    cache = prefill(model, prefix)
    for i in range(n_items):
        prefix, tail = prompt_ids(tok, shots, items[i], k)
        cached = tail_logits(model, cache, [tail], [(f"blocks.{l}.hook_in", grab_dtype) for l in [0, 1, model.cfg.n_layers - 1]])[0, -1].log_softmax(-1)
        full = tail_logits(model, DynamicCache(config=hf.config), [prefix + tail])[0, -1].log_softmax(-1)
        top = full.topk(5).indices
        print(f"{cyan}item {i}: argmax agree {cached.argmax() == full.argmax()}, max |dlogp| over full top-5 {(cached[top] - full[top]).abs().max():.3f}, top-1 {tok.decode([top[0]])!r} p={full[top[0]].exp():.3f}{endc}")
    print(f"{cyan}decoder-layer input dtypes {stream_dtypes}{endc}")

#%% Benchmark: greedy answer and exact P(correct answer + EOS) for every item at each filler length, one prefix cache per length; saves results/bench_{TAG}.json

run_bench = True
if run_bench:
    ks = [0, 100]
    # ks = [0, 10, 25, 50, 100]
    n_items = 600
    # n_items = 32
    batch_size = 16
    n_samples = 0
    # n_samples = 8
    temp = 1.0
    records = []
    for k in ks:
        pts = [prompt_ids(tok, shots, it, k) for it in items[:n_items]]
        assert all(prefix == pts[0][0] for prefix, _ in pts)
        cache = prefill(model, pts[0][0])
        for i in pbar(range(0, n_items, batch_size), desc=f"k={k}"):
            batch = items[i:min(i + batch_size, n_items)]
            tails = [tail for _, tail in pts[i:i + batch_size]]
            gens, first = decode(model, cache, tails)
            logp = answer_logprob(model, cache, tails, [answer_ids(tok, it["answer"]) for it in batch])
            top = first.topk(5)
            samples = [s for tail in tails for s in decode(model, cache, [tail] * n_samples, temp=temp)[0]] if n_samples else []
            for j, it in enumerate(batch):
                text = tok.decode([x for x in gens[j] if x != EOS])
                records.append({
                    "idx": it["idx"],
                    "k": k,
                    "gen": gens[j],
                    "text": text,
                    "parsed": parse_answer(text),
                    "correct": parse_answer(text) == it["answer"],
                    "eos": EOS in gens[j],
                    "logp": logp[j].item(),
                    "top_ids": top.indices[j].tolist(),
                    "top_logp": top.values[j].tolist(),
                    "samples": [parse_answer(tok.decode([x for x in s if x != EOS])) for s in samples[j * n_samples:(j + 1) * n_samples]],
                })
        done = [r for r in records if r["k"] == k]
        n_correct = sum(r["correct"] for r in done)
        ci = wilson(n_correct, len(done))
        print(f"{cyan}k={k}: greedy {n_correct}/{len(done)} = {n_correct / len(done):.3f} [{ci[0]:.3f}, {ci[1]:.3f}], mean P(correct) {sum(math.exp(r['logp']) for r in done) / len(done):.3f}, no EOS {sum(not r['eos'] for r in done)}{endc}")
    json.dump(records, open(f"results/bench_{TAG}.json", "w"))
    print(f"{green}saved results/bench_{TAG}.json{endc}")

#%% Compare with Olivia's API greedy run (phase2_full): accuracy with Wilson CIs, per-item answer agreement by local confidence margin, and overlap of the 0 -> 100 dots flip sets

compare_api = True
if compare_api:
    ours = {(r["idx"], r["k"]): r for r in json.load(open(f"results/bench_{TAG}.json"))}
    theirs = {(r["idx"], int(r["cond"].split("_")[1])): r for r in api if r["cond"].startswith("dots")}
    for k in [0, 100]:
        keys = [key for key in ours if key[1] == k and key in theirs]
        n = len(keys)
        n_ours = sum(ours[key]["correct"] for key in keys)
        n_api = sum(theirs[key]["correct"] for key in keys)
        mean_p = sum(math.exp(ours[key]["logp"]) for key in keys) / n
        agree = [ours[key]["parsed"] == theirs[key]["parsed"] for key in keys]
        margin = [ours[key]["top_logp"][0] - ours[key]["top_logp"][1] for key in keys]
        print(f"{purple}k={k}, n={n}{endc}")
        ci_ours, ci_api = wilson(n_ours, n), wilson(n_api, n)
        print(f"{cyan}local greedy {n_ours / n:.3f} [{ci_ours[0]:.3f}, {ci_ours[1]:.3f}], API greedy {n_api / n:.3f} [{ci_api[0]:.3f}, {ci_api[1]:.3f}], local mean P {mean_p:.3f}, answer agreement {sum(agree) / n:.3f}{endc}")
        for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, math.inf)]:
            sel = [a for a, m in zip(agree, margin) if lo <= m < hi]
            print(f"{gray}  top-1 minus top-2 log-prob in [{lo}, {hi}): n={len(sel)}, agreement {sum(sel) / max(len(sel), 1):.3f}{endc}")
    both = [i for i in range(len(items)) if all((i, k) in ours and (i, k) in theirs for k in [0, 100]) and not ours[(i, 0)]["correct"] and not theirs[(i, 0)]["correct"]]
    flip_ours = {i for i in both if not ours[(i, 0)]["correct"] and ours[(i, 100)]["correct"]}
    flip_api = {i for i in both if not theirs[(i, 0)]["correct"] and theirs[(i, 100)]["correct"]}
    a, b, c = len(flip_ours & flip_api), len(flip_ours - flip_api), len(flip_api - flip_ours)
    d = len(both) - a - b - c
    print(f"{cyan}flips 0 -> 100 dots among the {len(both)} items wrong at 0 dots in both runs: local {len(flip_ours)}, API {len(flip_api)}, both {a}, odds ratio {(a + 0.5) * (d + 0.5) / ((b + 0.5) * (c + 0.5)):.1f} (Haldane-corrected){endc}")

#%% Local P(correct) against Olivia's T=1 pass rates on items 0-299 (16 samples at 100 dots; the 8 held-out samples at 0 dots)

compare_sampled = True
if compare_sampled:
    ours = {(r["idx"], r["k"]): r for r in json.load(open(f"results/bench_{TAG}.json"))}
    rates = list(csv.DictReader(open("data/phase3_M300_items.csv")))
    for k, col in [(0, "p0"), (100, "pf")]:
        rows = [r for r in rates if (int(r["idx"]), k) in ours]
        p_local = t.tensor([math.exp(ours[(int(r["idx"]), k)]["logp"]) for r in rows])
        p_api = t.tensor([float(r[col]) for r in rows])
        print(f"{cyan}k={k}, n={len(rows)}: Pearson r {pearson(p_local, p_api):.3f}, mean local P {p_local.mean():.3f}, mean API pass rate {p_api.mean():.3f}{endc}")
        scatter(
            p_api,
            p_local,
            labels={"x": f"API T=1 pass rate ({col})", "y": "local P(correct)"},
            title=f"Local P(correct) vs API sampling, {k} dots",
            add_line="y=x",
            opacity=0.5,
        )

#%% Per-item P(correct) at 0 vs 100 dots, colored by the (c1, c2) multiplier cell, and the distribution of the change

plot_flips = True
if plot_flips:
    ours = {(r["idx"], r["k"]): r for r in json.load(open(f"results/bench_{TAG}.json"))}
    idxs = [i for i in range(len(items)) if (i, 0) in ours and (i, 100) in ours]
    p0 = t.tensor([math.exp(ours[(i, 0)]["logp"]) for i in idxs])
    p100 = t.tensor([math.exp(ours[(i, 100)]["logp"]) for i in idxs])
    cell = [f"c1={items[i]['chain']['c1x'] // items[i]['chain']['x']}, c2={items[i]['coefficient']}" for i in idxs]
    scatter(
        p0,
        p100,
        color=cell,
        labels={"x": "P(correct), 0 dots", "y": "P(correct), 100 dots", "color": "multipliers"},
        title="Per-item filler uplift",
        add_line="y=x",
        opacity=0.6,
    )
    hist(
        p100 - p0,
        nbins=50,
        labels={"x": "P(correct) at 100 dots minus at 0 dots"},
        title="Per-item change in P(correct)",
    )

#%% Reproduction for readers without context: accuracy with no filler and with 100 dots in Olivia's API runs and on the local model, with 95% bootstrap intervals over questions (left), and per question the local P(correct) against her T=1 pass rate (right); written to figs/reproduction.html (run with TAG = "v4flash_fp32r_b16", the bf16 benchmark)

plot_reproduction = True
if plot_reproduction:
    ours = {(r["idx"], r["k"]): r for r in json.load(open(f"results/bench_{TAG}.json"))}
    theirs = {(r["idx"], int(r["cond"].split("_")[1])): r for r in api if r["cond"].startswith("dots")}
    rates = list(csv.DictReader(open("data/phase3_M300_items.csv")))
    bars = {  # label -> {k: per-question values}
        "API greedy<br>(600 questions)": {k: [float(theirs[(i, k)]["correct"]) for i in range(len(items))] for k in [0, 100]},
        "API sampled at T=1<br>(300 questions, 8 / 16 samples)": {0: [float(r["p0"]) for r in rates], 100: [float(r["pf"]) for r in rates]},
        "local greedy<br>(600 questions)": {k: [float(ours[(i, k)]["correct"]) for i in range(len(items))] for k in [0, 100]},
        "local P(correct)<br>(600 questions)": {k: [math.exp(ours[(i, k)]["logp"]) for i in range(len(items))] for k in [0, 100]},
    }
    fig = make_subplots(rows=1, cols=2, column_widths=[0.56, 0.44], horizontal_spacing=0.08, subplot_titles=["Accuracy with no filler and with 100 filler dots", "Per question: local P(correct) vs the API's T=1 pass rate"])
    for k, color in [(0, SERIES[0]), (100, SERIES[1])]:
        stats = []
        for vals in bars.values():
            v = t.tensor(vals[k])
            boot = v[t.randint(0, len(v), (5000, len(v)), generator=t.Generator().manual_seed(0))].mean(1)
            stats.append((v.mean().item(), boot.quantile(0.025).item(), boot.quantile(0.975).item()))
        fig.add_trace(go.Bar(x=list(bars), y=[m for m, _, _ in stats], marker_color=color, name=f"{k} dots", text=[f"{m:.2f}" for m, _, _ in stats], textposition="inside", insidetextanchor="start",
            error_y={"type": "data", "symmetric": False, "array": [hi - m for m, _, hi in stats], "arrayminus": [m - lo for m, lo, _ in stats], "color": "#e8e6dc"}), row=1, col=1)
    r_text = []
    for k, col, color in [(0, "p0", SERIES[0]), (100, "pf", SERIES[1])]:
        x = t.tensor([float(r[col]) for r in rates])
        y = t.tensor([math.exp(ours[(int(r["idx"]), k)]["logp"]) for r in rates])
        r_text.append(f"r = {pearson(x, y):.2f} at {k} dots")
        fig.add_trace(go.Scatter(x=x.tolist(), y=y.tolist(), mode="markers", marker={"color": color, "size": 6, "opacity": 0.6}, name=f"{k} dots, one point per question", showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"color": "#e8e6dc", "dash": "dot", "width": 1}, showlegend=False, hoverinfo="skip"), row=1, col=2)
    fig.update_yaxes(title_text="share correct / mean P(correct)", range=[0, 0.8], row=1, col=1)
    fig.update_xaxes(title_text="API pass rate at T=1 (share of 8 samples at 0 dots, 16 at 100)", range=[-0.03, 1.03], row=1, col=2)
    fig.update_yaxes(title_text="local P(correct)", range=[-0.03, 1.03], row=1, col=2)
    fig.update_layout(
        title=(
            "<b>The open DeepSeek V4 Flash weights, run locally, show the same filler uplift as the API</b>  600 two-step arithmetic questions, no filler vs 100 filler dots"
            "<br><span style='font-size:13px'>Each question is asked with k dots ('Filler: . . .') between the question and 'Answer:'; the prompt's 10 worked examples carry the same filler. API: Olivia Velte's runs through OpenRouter. Local: the open weights, dequantized to bf16.</span>"
            "<br><span style='font-size:13px'>P(correct): the local model's probability of writing exactly the right number and stopping, its expected accuracy when sampling at T=1. Error bars: 95% intervals from resampling questions.</span>"
            f"<br><span style='font-size:13px'>Right: one point per question (the 300 with API samples), local P(correct) against the share of the API's samples that were right; {', '.join(r_text)}. Dotted: equal values.</span>"
        ),
        barmode="group",
        height=720,
        width=1700,
        margin={"t": 190},
        legend={"title": "filler"},
        **DARK,
    )
    write_dark_html(fig, "figs/reproduction.html")
    print(f"{green}wrote figs/reproduction.html; {', '.join(r_text)}{endc}")

#%% Dose sweep: exact P(correct answer + EOS) for every item at filler lengths 0, 5, ..., 100 (no decoding), one prefix cache per length; saves results/dose_{TAG}.json (the commented `runs` lines are the long, text-filler and placement variants)

run_dose = True
if run_dose:
    runs = [("dots", False, list(range(0, 101, 5)), f"results/dose_{TAG}.json")]  # (filler kind, filler before the question, ks, output)
    # runs = [("dots", False, [200, 500, 1000], f"results/dose_long_{TAG}.json")]  # the long-filler extension: the whole question leaves the exact 128-token window past about 120 dots
    # runs = [("wiki", False, [0, 10, 20, 50, 100, 200, 500, 1000, 2000], f"results/dose_wiki_{TAG}.json")]  # k tokens of Wikipedia text instead of k dots (utils.TEXT_FILLER)
    # runs = [("count", False, [0, 10, 20, 50, 100, 200, 500, 1000, 2000], f"results/dose_count_{TAG}.json")]  # k tokens of counting from a random start, a different sequence per question and shot
    # runs = [("rand", False, [0, 10, 100], f"results/dose_rand_{TAG}.json"), ("rand", "k", [10, 100], f"results/dose_rand_before_{TAG}.json"), ("dots", "k", [10, 100], f"results/dose_before_{TAG}.json")]  # random integers 1..1000 after the question; then random integers and dots before the question ("k": the swept length goes above the definitions, nothing after the question; at k = 0 the placements share a prompt)
    # runs = [("wiki", "k", [10, 100], f"results/dose_wiki_before_{TAG}.json"), ("count", "k", [10, 100], f"results/dose_count_before_{TAG}.json")]  # Wikipedia text and counting before the question
    # runs = [("dots", (b, n), [0, 10, 100], f"results/dose_{b}_before{n}_dots_{TAG}.json") for n in [10, 100] for b in ["wiki", "count", "rand"]]  # n tokens of Wikipedia text, counting or random integers above the definitions plus 0, 10 or 100 dots after the question
    n_items = 600
    # n_items = 32
    for kind, before, ks, out in runs:
        records = []
        for k in ks:
            batch_size = 16 if k <= 100 else 8 if k <= 500 else 2 if k <= 1000 else 1  # the CSA branch scores every pooled entry before its top-512 cut: about k keys per query
            pts = [prompt_ids(tok, shots, it, 0, kind, (kind, k)) if before == "k" else prompt_ids(tok, shots, it, k, kind, before) for it in items[:n_items]]
            assert all(prefix == pts[0][0] for prefix, _ in pts)
            cache = prefill(model, pts[0][0], chunk=1024)
            for i in pbar(range(0, n_items, batch_size), desc=f"{kind} {before} k={k}"):
                batch = items[i:min(i + batch_size, n_items)]
                logp = answer_logprob(model, cache, [tail for _, tail in pts[i:i + batch_size]], [answer_ids(tok, it["answer"]) for it in batch])
                records += [{"idx": it["idx"], "k": k, "logp": logp[j].item()} for j, it in enumerate(batch)]
            done = [math.exp(r["logp"]) for r in records if r["k"] == k]
            print(f"{cyan}{kind} {before} k={k}: mean P(correct) {sum(done) / len(done):.3f}{endc}")
            json.dump(records, open(out, "w"))
        print(f"{green}saved {out}{endc}")

#%% Dose curve for readers without context: mean P(correct) against the number of filler dots, for all questions and by the two multipliers, with 95% bootstrap intervals over questions; written to figs/dose_curve.html

plot_dose = True
if plot_dose:
    n_boot = 5000
    recs = json.load(open(f"results/dose_{TAG}.json"))
    ks = sorted({r["k"] for r in recs})
    idxs = sorted({r["idx"] for r in recs})
    lp = {(r["idx"], r["k"]): r["logp"] for r in recs}
    p = t.tensor([[math.exp(lp[(i, k)]) for k in ks] for i in idxs])  # [items, ks]
    cc = t.tensor([[items[i]["chain"]["c1x"] // items[i]["chain"]["x"], items[i]["coefficient"]] for i in idxs])
    lines = {"all questions": t.ones(len(idxs), dtype=t.bool), "×2 then ×2": (cc == 2).all(1), "×2 then ×3": (cc[:, 0] == 2) & (cc[:, 1] == 3), "×3 then ×2": (cc[:, 0] == 3) & (cc[:, 1] == 2), "×3 then ×3": (cc == 3).all(1)}
    fig = go.Figure()
    for (name, m), color in zip(lines.items(), ["#e8e6dc"] + SERIES):
        pm = p[m]
        boot = pm[t.randint(0, len(pm), (n_boot, len(pm)), generator=t.Generator().manual_seed(0))].mean(1)
        lo, hi = boot.quantile(0.025, 0), boot.quantile(0.975, 0)
        fig.add_trace(go.Scatter(x=ks + ks[::-1], y=hi.tolist() + lo.flip(0).tolist(), fill="toself", fillcolor=color, opacity=0.15, line={"width": 0}, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=ks, y=pm.mean(0).tolist(), mode="lines+markers", line={"color": color, "width": 4 if name == "all questions" else 2}, name=f"{name} (n={m.sum().item()})"))
    allp = p.mean(0)
    fig.update_layout(
        title=(
            f"<b>How the chance of a correct answer grows with the number of filler dots</b>  DeepSeek V4 Flash, {len(idxs)} two-step arithmetic questions"
            f"<br><span style='font-size:13px'>All questions: {allp[0]:.2f} with no filler, {allp[-1]:.2f} with {ks[-1]} dots. Each question is asked with k dots ('Filler: . . .') between the question and 'Answer:'; the prompt's 10 worked examples carry the same filler.</span>"
            f"<br><span style='font-size:13px'>Example question (×3 then ×2): <i>{items[0]['x_name']} = {items[0]['chain']['x']} · {items[0]['queried_term']} = {dict(items[0]['definitions'])[items[0]['queried_term']]} · (3 more definitions) · {items[0]['question']}</i>  →  {items[0]['queried_term']} = {items[0]['chain']['y']}, answer = {items[0]['answer']}</span>"
            "<br><span style='font-size:13px'>Probability of the correct answer: the model's probability of writing exactly the right number and stopping (its expected accuracy when sampling). Bands: 95% intervals from resampling questions.</span>"
        ),
        xaxis_title="number of filler dots",
        yaxis_title="mean probability of the correct answer",
        xaxis={"tickvals": ks},
        yaxis_range=[0, 1],
        height=700,
        width=1400,
        margin={"t": 170},
        legend={"title": "questions (multipliers in the two steps)"},
        **DARK,
    )
    write_dark_html(fig, "figs/dose_curve.html")
    print(f"{green}wrote figs/dose_curve.html{endc}")
    for name, m in lines.items():
        print(f"{name:14s} " + " ".join(f"{k}:{v:.2f}" for k, v in zip(ks, p[m].mean(0).tolist())))

#%% Histograms of the per-question uplift: P(correct) at 10 and at 100 dots minus at 0 dots (top), and the fraction of the no-filler error removed for questions with P0 < 0.5 (bottom), from the dose sweep; written to figs/uplift_hist.html

plot_uplift_hist = True
if plot_uplift_hist:
    recs = json.load(open(f"results/dose_{TAG}.json"))
    lp = {(r["idx"], r["k"]): r["logp"] for r in recs}
    idxs = sorted({r["idx"] for r in recs})
    p = {k: t.tensor([math.exp(lp[(i, k)]) for i in idxs]) for k in [0, 10, 100]}
    room = p[0] < 0.5  # the fraction of error removed is unstable when there is almost no error left
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"10 dots: mean change {(p[10] - p[0]).mean():+.3f}", f"100 dots: mean change {(p[100] - p[0]).mean():+.3f}", f"10 dots: fraction of error removed, {room.sum().item()} questions with P0 < 0.5", f"100 dots: fraction of error removed, {room.sum().item()} questions with P0 < 0.5"], shared_yaxes=True, vertical_spacing=0.18)
    for col, k in enumerate([10, 100], 1):
        d = p[k] - p[0]
        f = ((p[k] - p[0]) / (1 - p[0]))[room].clamp(min=-1)
        fig.add_trace(go.Histogram(x=d.tolist(), xbins={"start": -1, "end": 1, "size": 0.05}, marker_color=SERIES[col - 1], name=f"{k} dots", showlegend=False), row=1, col=col)
        fig.add_trace(go.Histogram(x=f.tolist(), xbins={"start": -1, "end": 1, "size": 0.05}, marker_color=SERIES[col - 1], name=f"{k} dots", showlegend=False), row=2, col=col)
        for row in [1, 2]:
            fig.add_vline(x=0, line={"color": "#e8e6dc", "width": 1, "dash": "dot"}, row=row, col=col)
        print(f"{k} dots: mean {d.mean():+.3f}, median {d.median():+.3f}, |change| > 0.1: {(d.abs() > 0.1).float().mean():.2f} of items, up > 0.5: {(d > 0.5).sum().item()}, down < -0.5: {(d < -0.5).sum().item()}; fraction of error removed (P0 < 0.5): mean {f.mean():+.3f}, median {f.median():+.3f}, > 0.5: {(f > 0.5).sum().item()}, > 0.9: {(f > 0.9).sum().item()}, clipped at -1: {(((p[k] - p[0]) / (1 - p[0]))[room] < -1).sum().item()}")
    fig.update_layout(
        title=(
            f"<b>Per-question change in the chance of a correct answer with filler dots</b>  DeepSeek V4 Flash, {len(idxs)} two-step arithmetic questions"
            "<br><span style='font-size:13px'>Each count is one question. Top: its probability of the correct answer with k dots ('Filler: . . .' between the question and 'Answer:') minus with no filler. Probability of the correct answer: the model's probability of writing exactly the right number and stopping.</span>"
            "<br><span style='font-size:13px'>Bottom: the fraction of the no-filler error removed, (P(correct) with filler − without) / (1 − without), for questions with P(correct) below 0.5 without filler; values below −1 are shown at −1.</span>"
        ),
        height=950,
        width=1400,
        margin={"t": 150},
        **DARK,
    )
    fig.update_xaxes(title_text="P(correct) with filler minus P(correct) without", range=[-1, 1], row=1)
    fig.update_xaxes(title_text="fraction of the no-filler error removed", range=[-1, 1], row=2)
    fig.update_yaxes(title_text="questions", col=1)
    write_dark_html(fig, "figs/uplift_hist.html")
    print(f"{green}wrote figs/uplift_hist.html{endc}")

#%% Which questions get the 100-dot uplift: Gaussian mixtures on the per-question change (1D) and on (P0, P100) (2D), component count by BIC; the uplift component's members go to results/uplift_items_{TAG}.csv and figs/uplift_mixture.html overlays the 1D fit on the histogram

find_uplift_subset = True
if find_uplift_subset:
    from sklearn.mixture import GaussianMixture
    recs = json.load(open(f"results/dose_{TAG}.json"))
    lp = {(r["idx"], r["k"]): r["logp"] for r in recs}
    idxs = sorted({r["idx"] for r in recs})
    p0 = t.tensor([math.exp(lp[(i, 0)]) for i in idxs]); p100 = t.tensor([math.exp(lp[(i, 100)]) for i in idxs]); d = p100 - p0
    fits = {n: GaussianMixture(n, n_init=10, random_state=0).fit(d[:, None].numpy()) for n in range(1, 6)}
    print("1D BIC by components: " + " ".join(f"{n}:{g.bic(d[:, None].numpy()):.0f}" for n, g in fits.items()))
    g1 = fits[min(fits, key=lambda n: fits[n].bic(d[:, None].numpy()))]
    order = g1.means_[:, 0].argsort()
    for j in order:
        print(f"  component mean {g1.means_[j, 0]:+.3f} sd {g1.covariances_[j, 0, 0] ** 0.5:.3f} weight {g1.weights_[j]:.2f} (~{g1.weights_[j] * len(idxs):.0f} questions)")
    up = order[-1]  # the component with the largest mean change
    resp = t.tensor(g1.predict_proba(d[:, None].numpy())[:, up])
    fits2 = {n: GaussianMixture(n, n_init=10, random_state=0).fit(t.stack([p0, p100], 1).numpy()) for n in range(2, 8)}
    print("2D BIC by components: " + " ".join(f"{n}:{g.bic(t.stack([p0, p100], 1).numpy()):.0f}" for n, g in fits2.items()))
    g2 = fits2[min(fits2, key=lambda n: fits2[n].bic(t.stack([p0, p100], 1).numpy()))]
    lab2 = t.tensor(g2.predict(t.stack([p0, p100], 1).numpy()))
    for j in g2.means_[:, 1].argsort():
        m = lab2 == j
        print(f"  2D component P0 {g2.means_[j, 0]:.2f} P100 {g2.means_[j, 1]:.2f} n={m.sum().item()} mean change {d[m].mean():+.3f}; in the 1D uplift component (resp > 0.5): {(resp[m] > 0.5).float().mean():.2f}")
    noise = t.cat([p100 - t.tensor([math.exp(lp[(i, k)]) for i in idxs]) for k in [90, 95]])  # within-run change between neighbouring filler lengths: the scale of a move that means nothing
    thr = noise.abs().quantile(0.975).item()
    print(f"noise proxy (P100 - P95, P100 - P90): sd {noise.std():.3f}, |change| > 0.1 on {(noise.abs() > 0.1).float().mean():.2f} of items, 97.5% of |change| below {thr:.3f}")
    mover = resp > 0.5
    sel = mover & (d > 0)
    cc = t.tensor([[items[i]["chain"]["c1x"] // items[i]["chain"]["x"], items[i]["coefficient"]] for i in idxs])
    print(f"\nbroad component: {mover.sum().item()} questions, of which {sel.sum().item()} move up (change {d[sel].min():+.2f} to {d[sel].max():+.2f}, mean {d[sel].mean():+.2f}, {(d[sel] > thr).sum().item()} above the noise threshold; P0 median {p0[sel].median():.2f}, P100 median {p100[sel].median():.2f}) and {(mover & (d < 0)).sum().item()} move down (mean {d[mover & (d < 0)].mean():+.2f}, {(d[mover & (d < 0)] < -thr).sum().item()} below -threshold); the {(~mover).sum().item()} others: mean change {d[~mover].mean():+.3f}, sd {d[~mover].std():.3f}")
    for c1, c2 in [(2, 2), (2, 3), (3, 2), (3, 3)]:
        m = (cc[:, 0] == c1) & (cc[:, 1] == c2)
        print(f"  ×{c1} then ×{c2}: {(m & sel).sum().item()} of {m.sum().item()} move up ({(m & sel).sum().item() / m.sum().item():.2f}), {(m & mover & (d < 0)).sum().item()} down; P0 of the risers median {p0[m & sel].median():.2f}, of the rest {p0[m & ~sel].median():.2f}")
    rows = sorted([(idxs[j], cc[j, 0].item(), cc[j, 1].item(), items[idxs[j]]["chain"]["x"], items[idxs[j]]["chain"]["y"], items[idxs[j]]["answer"], p0[j].item(), p100[j].item(), d[j].item(), resp[j].item()) for j in range(len(idxs)) if sel[j]], key=lambda r: -r[8])
    with open(f"results/uplift_items_{TAG}.csv", "w") as f:
        w = csv.writer(f); w.writerow(["idx", "c1", "c2", "x", "y", "answer", "p0", "p100", "change", "resp_mover", "above_noise"]); w.writerows([[*r[:6], f"{r[6]:.4f}", f"{r[7]:.4f}", f"{r[8]:.4f}", f"{r[9]:.3f}", int(r[8] > thr)] for r in rows])
    print(f"{green}wrote results/uplift_items_{TAG}.csv{endc} ({len(rows)} risers, sorted by change; idx, multipliers, P0 -> P100, first 40):")
    print("  " + "  ".join(f"{r[0]}(×{r[1]}×{r[2]} {r[6]:.2f}->{r[7]:.2f})" for r in rows[:40]))
    xs = t.linspace(-1, 1, 400)
    fig = go.Figure(go.Histogram(x=d.tolist(), xbins={"start": -1, "end": 1, "size": 0.05}, marker_color="#555", name="questions"))
    for j, color in zip(order, SERIES):
        pdf = g1.weights_[j] * t.exp(-0.5 * ((xs - g1.means_[j, 0]) / g1.covariances_[j, 0, 0] ** 0.5) ** 2) / (g1.covariances_[j, 0, 0] ** 0.5 * (2 * math.pi) ** 0.5)
        fig.add_trace(go.Scatter(x=xs.tolist(), y=(pdf * len(idxs) * 0.05).tolist(), line={"color": color, "width": 2}, name=f"component: mean {g1.means_[j, 0]:+.2f}, sd {g1.covariances_[j, 0, 0] ** 0.5:.2f}, {g1.weights_[j] * len(idxs):.0f} questions"))
    fig.update_layout(title=f"<b>Mixture fit to the per-question change in P(correct) with 100 dots</b>  {len(idxs)} questions, {g1.n_components} Gaussian components chosen by BIC", xaxis_title="P(correct) with 100 dots minus without", yaxis_title="questions per 0.05 bin", height=550, width=1100, **DARK)
    write_dark_html(fig, "figs/uplift_mixture.html")
    print(f"{green}wrote figs/uplift_mixture.html{endc}")

#%% Long-filler dose curve for readers without context: mean P(correct) at 0 to 1000 dots (the 0..100 sweep plus results/dose_long_{TAG}.json), evenly spaced ticks, all questions and by multipliers, with 95% bootstrap intervals; written to figs/dose_curve_long.html

plot_dose_long = True
if plot_dose_long:
    n_boot = 5000
    show_ks = [0, 10, 25, 50, 100, 200, 500, 1000]
    recs = json.load(open(f"results/dose_{TAG}.json")) + json.load(open(f"results/dose_long_{TAG}.json"))
    idxs = sorted({r["idx"] for r in recs})
    lp = {(r["idx"], r["k"]): r["logp"] for r in recs}
    p = t.tensor([[math.exp(lp[(i, k)]) for k in show_ks] for i in idxs])  # [items, ks]
    cc = t.tensor([[items[i]["chain"]["c1x"] // items[i]["chain"]["x"], items[i]["coefficient"]] for i in idxs])
    lines = {"all questions": t.ones(len(idxs), dtype=t.bool), "×2 then ×2": (cc == 2).all(1), "×2 then ×3": (cc[:, 0] == 2) & (cc[:, 1] == 3), "×3 then ×2": (cc[:, 0] == 3) & (cc[:, 1] == 2), "×3 then ×3": (cc == 3).all(1)}
    xs = [str(k) for k in show_ks]
    fig = go.Figure()
    for (name, m), color in zip(lines.items(), ["#e8e6dc"] + SERIES):
        pm = p[m]
        boot = pm[t.randint(0, len(pm), (n_boot, len(pm)), generator=t.Generator().manual_seed(0))].mean(1)
        lo, hi = boot.quantile(0.025, 0), boot.quantile(0.975, 0)
        fig.add_trace(go.Scatter(x=xs + xs[::-1], y=hi.tolist() + lo.flip(0).tolist(), fill="toself", fillcolor=color, opacity=0.15, line={"width": 0}, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=xs, y=pm.mean(0).tolist(), mode="lines+markers", line={"color": color, "width": 4 if name == "all questions" else 2}, name=f"{name} (n={m.sum().item()})"))
    allp = p.mean(0)
    fig.update_layout(
        title=(
            f"<b>Does the chance of a correct answer keep growing with more filler dots?</b>  DeepSeek V4 Flash, {len(idxs)} two-step arithmetic questions, 0 to 1000 dots"
            f"<br><span style='font-size:13px'>All questions: {allp[0]:.2f} with no filler, {allp[show_ks.index(100)]:.2f} with 100 dots, {allp[-1]:.2f} with 1000. Each question is asked with k dots ('Filler: . . .') between the question and 'Answer:'; the prompt's 10 worked examples carry the same filler, so the prompt is about 11,000 tokens at 1000 dots.</span>"
            "<br><span style='font-size:13px'>The model attends exactly over a 128-token window and through pooled summaries beyond it, so past about 120 dots the question is reachable from the answer only through the summaries.</span>"
            "<br><span style='font-size:13px'>Probability of the correct answer: the model's probability of writing exactly the right number and stopping (its expected accuracy when sampling). Bands: 95% intervals from resampling questions. The x axis is evenly spaced, not to scale.</span>"
        ),
        xaxis_title="number of filler dots",
        yaxis_title="mean probability of the correct answer",
        xaxis={"type": "category"},
        yaxis_range=[0, 1],
        height=700,
        width=1400,
        margin={"t": 170},
        legend={"title": "questions (multipliers in the two steps)"},
        **DARK,
    )
    write_dark_html(fig, "figs/dose_curve_long.html")
    print(f"{green}wrote figs/dose_curve_long.html{endc}")
    for name, m in lines.items():
        print(f"{name:14s} " + " ".join(f"{k}:{v:.2f}" for k, v in zip(show_ks, p[m].mean(0).tolist())))

#%% Text-filler dose curves for readers without context: for each text filler kind (Wikipedia text, counting), mean P(correct) at 0 to 2000 filler tokens (results/dose_{kind}_{TAG}.json), evenly spaced ticks, all questions, by multipliers and the questions the model is unsure about without filler, with 95% bootstrap intervals, plus the all-questions and unsure curves of the other filler kinds for comparison; written to figs/dose_curve_{kind}.html, and as the fraction of the no-filler error removed to figs/dose_curve_{kind}_err.html

plot_dose_text = True
if plot_dose_text:
    kinds = {"wiki": "Wikipedia text", "count": "counting"}
    describe = {
        "wiki": "the first k tokens of the Wikipedia article 'Tree' ('Filler: In botany, a tree is a perennial plant ...'); the prompt's 10 worked examples carry the same text",
        "count": "k tokens of integers counted up or down from a random start below 10,000 ('Filler: 6311 6310 6309 ...'), a different sequence for every question and for each of the prompt's 10 worked examples",
    }
    recs = {kind: json.load(open(f"results/dose_{kind}_{TAG}.json")) for kind in kinds}
    recs["dots"] = json.load(open(f"results/dose_{TAG}.json")) + json.load(open(f"results/dose_long_{TAG}.json"))
    lp = {kind: {(r["idx"], r["k"]): r["logp"] for r in rs} for kind, rs in recs.items()}
    for kind, label in kinds.items():
        show_ks = sorted({k for _, k in lp[kind]})
        idxs = sorted({i for i, _ in lp[kind]})
        p = t.tensor([[math.exp(lp[kind][(i, k)]) for k in show_ks] for i in idxs])  # [items, ks]
        others = {o: t.tensor([[math.exp(lp[o][(i, k)]) if (i, k) in lp[o] else math.nan for k in show_ks] for i in idxs]) for o in lp if o != kind}
        cc = t.tensor([[items[i]["chain"]["c1x"] // items[i]["chain"]["x"], items[i]["coefficient"]] for i in idxs])
        unsure = t.tensor([math.exp(lp["dots"][(i, 0)]) < 0.8 for i in idxs])  # questions the model is unsure about with no filler (from the dots sweep's k=0 run)
        every = t.ones(len(idxs), dtype=t.bool)
        lines = {"all questions": every, "×2 then ×2": (cc == 2).all(1), "×2 then ×3": (cc[:, 0] == 2) & (cc[:, 1] == 3), "×3 then ×2": (cc[:, 0] == 3) & (cc[:, 1] == 2), "×3 then ×3": (cc == 3).all(1), "unsure without filler: P(correct) < 0.8 at 0 tokens": unsure}
        xs = [str(k) for k in show_ks]
        i100, last = show_ks.index(100), show_ks[-1]
        for metric, (suffix, ytitle, ylabel, yrange, explain) in METRIC.items():
            fig = go.Figure()
            for (name, m), color in zip(lines.items(), ["#e8e6dc"] + SERIES[:4] + [UNSURE]):
                v, lo, hi = summarize(p[m], p[m, 0], metric)
                fig.add_trace(go.Scatter(x=xs + xs[::-1], y=hi.tolist() + lo.flip(0).tolist(), fill="toself", fillcolor=color, opacity=0.15, line={"width": 0}, hoverinfo="skip", showlegend=False))
                fig.add_trace(go.Scatter(x=xs, y=v.tolist(), mode="lines+markers", line={"color": color, "width": 4 if name == "all questions" or color == UNSURE else 2}, name=f"{name} (n={m.sum().item()})", hovertemplate=f"{ylabel} %{{y:.3f}}"))
            for (other, op), dash in zip(others.items(), ["dash", "dot"]):
                for m, color, who in [(every, "#e8e6dc", "all questions"), (unsure, UNSURE, "unsure questions")]:
                    fig.add_trace(go.Scatter(x=xs, y=summarize(op[m], op[m, 0], metric)[0].tolist(), mode="lines+markers", line={"color": color, "width": 2, "dash": dash}, name=f"{who}, {kinds.get(other, 'dots')} as filler instead", hovertemplate=f"{ylabel} %{{y:.3f}}"))
            allv, unv = summarize(p, p[:, 0], metric)[0], summarize(p[unsure], p[unsure, 0], metric)[0]
            summary = {
                "p": f"All questions: {allv[0]:.2f} with no filler, {allv[i100]:.2f} with 100 tokens, {allv[-1]:.2f} with {last}; the {unsure.sum().item()} questions the model is unsure about without filler (P(correct) < 0.8 at 0 tokens): {unv[0]:.2f}, {unv[i100]:.2f}, {unv[-1]:.2f}.",
                "frac": f"With no filler the error (1 − P(correct)) is {1 - p[:, 0].mean():.2f} over all questions and {1 - p[unsure, 0].mean():.2f} over the {unsure.sum().item()} the model is unsure about (P(correct) < 0.8); {label} removes {allv[i100]:.0%} / {allv[-1]:.0%} of it at 100 / {last} tokens over all questions, {unv[i100]:.0%} / {unv[-1]:.0%} over the unsure ones.",
            }[metric]
            fig.update_layout(
                title=(
                    f"<b>Does {label} work as filler the way dots do?</b>  DeepSeek V4 Flash, {len(idxs)} two-step arithmetic questions, 0 to {last} tokens of {label}"
                    f"<br><span style='font-size:13px'>{summary}</span>"
                    f"<br><span style='font-size:13px'>Each question is asked with {describe[kind]} between the question and 'Answer:'.</span>"
                    "<br><span style='font-size:13px'>The dashed and dotted lines are the same measurement with other fillers; the purple lines are over the unsure questions only. The x axis is evenly spaced, not to scale.</span>"
                    "<br><span style='font-size:13px'>The model attends exactly over a 128-token window and through pooled summaries beyond it, so past about 120 filler tokens the question is reachable from the answer only through the summaries.</span>"
                    + explain_lines(explain)
                ),
                xaxis_title="number of filler tokens",
                yaxis_title=ytitle,
                xaxis={"type": "category"},
                yaxis_range=yrange,
                height=820,
                width=1900,
                margin={"t": 270},
                legend={"title": "questions (multipliers in the two steps)"},
                **DARK,
            )
            write_dark_html(fig, f"figs/dose_curve_{kind}{suffix}.html")
            print(f"{green}wrote figs/dose_curve_{kind}{suffix}.html{endc}")
        for name, m in lines.items():
            print(f"{name:14s} " + " ".join(f"{k}:{v:.2f}" for k, v in zip(show_ks, p[m].mean(0).tolist())))
        for other, op in others.items():
            print(f"{kinds.get(other, 'dots'):14s} " + " ".join(f"{k}:{v:.2f}" for k, v in zip(show_ks, op.nanmean(0).tolist())))

#%% Filler placement and random integers for readers without context: mean P(correct) over all questions at 10 and 100 filler tokens for each filler kind (dots, Wikipedia text, counting, random integers), after and before the question, plus 10 or 100 tokens of Wikipedia text, counting or random integers before the question combined with 0, 10 or 100 dots after it; the questions the model is unsure about without filler as a hatched bar inside each bar; 95% bootstrap intervals and the no-filler levels as lines; written to figs/filler_position.html, as the fraction of the no-filler error removed to figs/filler_position_err.html, and with the combined groups' bars measured against the same prompt without the dots (what the dots add on top of the before-filler) to figs/filler_position_marg.html
plot_position = True
if plot_position:
    files = {  # (kind label, before) -> results file
        ("dots", False): f"results/dose_{TAG}.json", ("Wikipedia text", False): f"results/dose_wiki_{TAG}.json", ("counting", False): f"results/dose_count_{TAG}.json",
        ("random integers", False): f"results/dose_rand_{TAG}.json", ("dots", True): f"results/dose_before_{TAG}.json", ("Wikipedia text", True): f"results/dose_wiki_before_{TAG}.json",
        ("counting", True): f"results/dose_count_before_{TAG}.json", ("random integers", True): f"results/dose_rand_before_{TAG}.json",
    }
    combined = {f"{n} {name} before the question,<br>dots after the question": (f"results/dose_{kind}_before{n}_dots_{TAG}.json", n) for n in [10, 100] for kind, name in [("wiki", "Wikipedia tokens"), ("count", "counting tokens"), ("rand", "random integers")]}  # the k=0 run is the before-filler alone, drawn in the 'n before' slot
    lp = {key: {(r["idx"], r["k"]): r["logp"] for r in json.load(open(f))} for key, f in files.items()}
    lp.update({(label, False): {(r["idx"], r["k"]): r["logp"] for r in json.load(open(f)) if r["k"] > 0} for label, (f, n) in combined.items()})
    lp.update({(label, True): {(r["idx"], n): r["logp"] for r in json.load(open(f)) if r["k"] == 0} for label, (f, n) in combined.items()})
    idxs = sorted({i for i, _ in lp[("dots", False)]})
    p = {(key, k): t.tensor([math.exp(lp[key][(i, k)]) for i in idxs]) for key in lp for k in [10, 100] if (idxs[0], k) in lp[key]}
    P0 = t.tensor([math.exp(lp[("dots", False)][(i, 0)]) for i in idxs])
    unsure = P0 < 0.8  # questions the model is unsure about with no filler
    subsets = [(t.ones(len(idxs), dtype=t.bool), "all questions"), (unsure, f"the {unsure.sum().item()} unsure questions")]
    kinds = ["dots", "Wikipedia text", "counting", "random integers", *combined]
    conds = {"10 tokens after the question": (10, False), "100 tokens after the question": (100, False), "10 tokens before the question": (10, True), "100 tokens before the question": (100, True)}
    unit = {"dots": "dots", "Wikipedia text": "tokens of Wikipedia text", "counting": "tokens of counting", "random integers": "tokens of random integers"}
    def describe(kind: str, k: int, before: bool) -> str:
        if kind in combined:
            return kind.split(",")[0] + (", no dots after the question" if before else f", {k} dots after the question")
        return f"{k} {unit[kind]} {'before' if before else 'after'} the question, nothing {'after' if before else 'before'} it"
    marg = {**METRIC, "marg": ("_marg", "fraction of the remaining error removed by the dots", "fraction of error removed", [-0.35, 1], ["Fraction of error removed: (mean P(correct) with the filler − mean without) / (1 − mean without), each mean over the questions in the group, where P(correct) is the model's probability of writing exactly the right number and stopping.",
        "For the right-hand groups 'without' is the same prompt with the before-question filler and no dots (its own bar in the other versions of this figure, omitted here), so those bars show what the dots add on top of the before-filler's own effect.",
        "For the left-hand groups 'without' is the plain prompt. Bands or error bars: 95% intervals from resampling questions."])}
    for metric, (suffix, ytitle, ylabel, yrange, explain) in marg.items():
        fig = go.Figure()
        for (name, (k, before)), color in zip(conds.items(), SERIES):
            for sub, who in subsets:  # the unsure subset as a hatched bar over the lower part of each full bar
                base = lambda kind: p[((kind, True), combined[kind][1])] if metric == "marg" and kind in combined else P0  # the before-filler-only run as the baseline in the marginal version
                stats = [summarize(p[((kind, before), k)][sub], base(kind)[sub], "p" if metric == "p" else "frac") if ((kind, before), k) in p and not (metric == "marg" and before and kind in combined) else (t.tensor(math.nan),) * 3 for kind in kinds]
                marker = {"color": color} if who == "all questions" else {"color": color, "pattern": {"shape": "/", "fgcolor": "#1a1a19", "size": 6, "solidity": 0.4}}
                fig.add_trace(go.Bar(x=kinds, y=[round(v.item(), 4) for v, _, _ in stats], offsetgroup=name, error_y={"type": "data", "symmetric": False, "array": [(hi - v).item() for v, _, hi in stats], "arrayminus": [(v - lo).item() for v, lo, _ in stats], "color": "#e8e6dc", "thickness": 1},
                    marker=marker, name=name if who == "all questions" else "hatched: the same over the unsure questions only", showlegend=who == "all questions" or name == next(iter(conds)),
                    hovertext=[f"{describe(kind, k, before)}<br>{who}" for kind in kinds], hovertemplate="%{hovertext}<br>" + ylabel + " %{y:.4f}<extra></extra>"))
        if metric == "p":
            for (sub, who), pos in zip(subsets, ["top left", "bottom left"]):
                fig.add_hline(y=P0[sub].mean().item(), line={"color": "#e8e6dc", "dash": "dot", "width": 1.5}, annotation_text=f"no filler, {who} {P0[sub].mean():.2f}", annotation_position=pos)
        else:
            fig.add_hline(y=0, line={"color": "#e8e6dc", "width": 1})
        fig.update_layout(
            title=(
                f"<b>Which fillers help, and do they help only after the question?</b>  DeepSeek V4 Flash, {len(idxs)} two-step arithmetic questions"
                "<br><span style='font-size:13px'>Each question is asked with k filler tokens on a 'Filler:' line, either between the question and 'Answer:' (after) or above the variable definitions (before, with an empty 'Filler:' line kept before 'Answer:'); the prompt's 10 worked examples carry the same filler kind and placement.</span>"
                "<br><span style='font-size:13px'>Fillers: dots ('. . .'), the Wikipedia article 'Tree', integers counted up or down from a random start below 10,000, independent random integers from 1 to 1000 (counting and random integers differ for every question and example). Text fillers are cut to k tokens (±1).</span>"
                "<br><span style='font-size:13px'>The right-hand groups keep 10 or 100 tokens of Wikipedia text, counting or random integers above the definitions in every prompt and add 0 (the 'before the question' bar of that length), 10 or 100 dots after the question.</span>"
                f"<br><span style='font-size:13px'>The hatched part of each bar is the same measurement over the {unsure.sum().item()} questions the model is unsure about without filler (P(correct) < 0.8 at 0 tokens); the other {len(idxs) - unsure.sum().item()} questions are near the ceiling and add little to the uplift. Missing bars were not run.</span>"
                + explain_lines(explain)
            ),
            barmode="group",
            xaxis_title="filler kind",
            yaxis_title=ytitle,
            yaxis_range=[0, 1.02] if metric == "p" else [-0.3, 0.5],
            height=840,
            width=1900,
            margin={"t": 270},
            legend={"title": "filler length and placement"},
            **DARK,
        )
        write_dark_html(fig, f"figs/filler_position{suffix}.html")
        print(f"{green}wrote figs/filler_position{suffix}.html{endc}")
    print(f"no filler {P0.mean():.2f}, unsure questions {P0[unsure].mean():.2f}")
    for (key, k), pk in p.items():
        print(f"  {key[0]:16s} {'before' if key[1] else 'after ':6s} k={k:3d}: {pk.mean().item():.2f} ({(pk.mean() - P0.mean()) / (1 - P0.mean()):+.2f} of the error), unsure {pk[unsure].mean().item():.2f} ({(pk[unsure].mean() - P0[unsure].mean()) / (1 - P0[unsure].mean()):+.2f})")

#%% Alphabet filler: greedy answer and exact P(correct answer + EOS) for every item with no filler, 27 and 100 dots, and the capital alphabet 'A B ... Z' once (26 letters + the line's closing newline = 27 positions, where 27 dots also take 27) or repeated to 100 positions, the same filler in the shots; saves results/alpha_{TAG}.json

run_alpha = True
if run_alpha:
    conds = [("dots", 0), ("dots", 27), ("alpha", 27), ("dots", 100), ("alpha", 100)]  # (filler kind, filler positions)
    n_items = 600
    # n_items = 32
    batch_size = 16
    records = []
    for kind, k in conds:
        pts = [prompt_ids(tok, shots, it, k, kind) for it in items[:n_items]]
        assert all(prefix == pts[0][0] for prefix, _ in pts)
        cache = prefill(model, pts[0][0])
        for i in pbar(range(0, n_items, batch_size), desc=f"{kind} k={k}"):
            batch = items[i:min(i + batch_size, n_items)]
            tails = [tail for _, tail in pts[i:i + batch_size]]
            gens, first = decode(model, cache, tails)
            logp = answer_logprob(model, cache, tails, [answer_ids(tok, it["answer"]) for it in batch])
            for j, it in enumerate(batch):
                text = tok.decode([x for x in gens[j] if x != EOS])
                records.append({"idx": it["idx"], "kind": kind, "k": k, "text": text, "parsed": parse_answer(text), "correct": parse_answer(text) == it["answer"], "logp": logp[j].item()})
        done = [r for r in records if r["kind"] == kind and r["k"] == k]
        print(f"{cyan}{kind} k={k}: greedy {sum(r['correct'] for r in done) / len(done):.3f}, mean P(correct) {sum(math.exp(r['logp']) for r in done) / len(done):.3f}{endc}")
    json.dump(records, open(f"results/alpha_{TAG}.json", "w"))
    print(f"{green}saved results/alpha_{TAG}.json{endc}")

#%% Alphabet uplift for readers without context: mean P(correct) over all questions with 95% bootstrap intervals (bars) and greedy accuracy (hollow markers) for no filler, 27 and 100 dots, and the alphabet once (27 positions) or repeated to 100; written to figs/uplift_alpha.html

plot_alpha = True
if plot_alpha:
    n_boot = 2000
    conds = [("dots", 0, "no filler"), ("dots", 27, "27 dots"), ("alpha", 27, "alphabet once<br>(27 positions)"), ("dots", 100, "100 dots"), ("alpha", 100, "alphabet repeated<br>(100 positions)")]
    records = json.load(open(f"results/alpha_{TAG}.json"))
    p = t.tensor([[math.exp(r["logp"]) for r in records if (r["kind"], r["k"]) == (kind, k)] for kind, k, _ in conds])  # [condition, question]
    g = t.tensor([[float(r["correct"]) for r in records if (r["kind"], r["k"]) == (kind, k)] for kind, k, _ in conds])
    draws = t.randint(p.shape[1], (n_boot, p.shape[1]), generator=t.Generator().manual_seed(0))
    boot = p[:, draws].mean(-1)  # [condition, draw]: the same redrawn questions in every condition
    lo, hi = boot.quantile(0.025, dim=1), boot.quantile(0.975, dim=1)
    dlo, dhi = (boot - boot[0]).quantile(0.025, dim=1), (boot - boot[0]).quantile(0.975, dim=1)
    names = [name for _, _, name in conds]
    colors = ["#e8e6dc" if k == 0 else SERIES[0] if kind == "dots" else SERIES[1] for kind, k, _ in conds]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=p.mean(1).tolist(), error_y={"type": "data", "symmetric": False, "array": (hi - p.mean(1)).tolist(), "arrayminus": (p.mean(1) - lo).tolist()}, marker_color=colors, text=[f"{v:.2f}" for v in p.mean(1).tolist()], textposition="inside", insidetextanchor="middle", textfont={"size": 15, "color": "#1a1a19"}, name="mean P(correct), T=1"))
    fig.add_trace(go.Scatter(x=names, y=g.mean(1).tolist(), mode="markers+text", marker={"symbol": "circle-open", "size": 16, "color": "#e8e6dc", "line": {"width": 2.5}}, text=[f"{v:.2f}" for v in g.mean(1).tolist()], textposition="top right", textfont={"size": 12}, name="greedy accuracy"))
    fig.update_layout(
        title=(
            "<b>Does the alphabet work as filler the way dots do?</b>  DeepSeek V4 Flash, 600 variable-binding questions, 10-shot"
            "<br><span style='font-size:13px'>Bars: mean probability of the correct answer (teacher-forced P(answer, then end of turn), the expected accuracy when sampling at temperature 1),</span>"
            "<br><span style='font-size:13px'>with 95% bootstrap intervals over questions. Hollow markers: greedy accuracy.</span>"
            "<br><span style='font-size:13px'>The filler sits between the question and 'Answer:' on a 'Filler:' line, the same in the 10 worked examples. The alphabet is 'A B C … Z'</span>"
            "<br><span style='font-size:13px'>(one token per letter plus the line's closing newline = 27 positions); repeated, it continues 'A B …' to 100 positions.</span>"
        ),
        yaxis_title="P(correct) / accuracy",
        yaxis_range=[0, 1],
        height=700,
        width=1100,
        margin={"t": 190},
        legend={"orientation": "h", "y": -0.16, "x": 0.5, "xanchor": "center"},
        **DARK,
    )
    write_dark_html(fig, "figs/uplift_alpha.html")
    for j, name in enumerate(names):
        print(f"{cyan}{name.replace('<br>', ' '):34s} P(correct) {p[j].mean():.3f} [{lo[j]:.3f}, {hi[j]:.3f}]  vs no filler {p[j].mean() - p[0].mean():+.3f} [{dlo[j]:+.3f}, {dhi[j]:+.3f}]  greedy {g[j].mean():.3f}{endc}")
    print(f"{green}wrote figs/uplift_alpha.html{endc}")

#%% Filler types in one figure, for readers without context: mean P(correct) against the number of filler tokens for dots, Wikipedia text, counting, random integers and the alphabet, with 95% bootstrap intervals (left); of the questions each filler fixes at 100 tokens (P(correct) < 0.3 without filler, >= 0.6 with it), the share that 100 dots also fix, against chance and against 95 dots (right); written to figs/filler_types.html

plot_filler_types = True
if plot_filler_types:
    load = lambda path, kind=None: {(r["idx"], r["k"]): math.exp(r["logp"]) for r in json.load(open(path)) if kind is None or r["kind"] == kind}
    dots = load(f"results/dose_{TAG}.json") | load(f"results/dose_long_{TAG}.json")
    alpha_run = {kind: load(f"results/alpha_{TAG}.json", kind) for kind in ["dots", "alpha"]}  # the alphabet run also has 27 and 100 dots, for a matched comparison
    fillers = {
        "dots": dots,
        "Wikipedia text": load(f"results/dose_wiki_{TAG}.json"),
        "counting": load(f"results/dose_count_{TAG}.json"),
        "random integers": load(f"results/dose_rand_{TAG}.json"),
        "alphabet 'A B ... Z' (repeated)": alpha_run["alpha"] | {key: p for key, p in dots.items() if key[1] == 0},
    }
    ticks = [0, 10, 20, 50, 100, 200, 500, 1000, 2000]
    idxs = list(range(len(items)))
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.09, subplot_titles=["Mean P(correct) by number of filler tokens", "Of the questions a filler fixes at 100 tokens, the share 100 dots also fix"])
    for (name, f), color in zip(fillers.items(), ["#e8e6dc"] + SERIES):
        stats = {}
        for k in sorted({k for _, k in f}):
            v = t.tensor([f[(i, k)] for i in idxs])
            boot = v[t.randint(0, len(v), (5000, len(v)), generator=t.Generator().manual_seed(0))].mean(1)
            stats[k] = (v.mean().item(), boot.quantile(0.025).item(), boot.quantile(0.975).item())
        x = [k + 1 for k in stats]  # log axis: k + 1 so that 0 fits
        fig.add_trace(go.Scatter(x=x, y=[m for m, _, _ in stats.values()], mode="lines+markers", line={"color": color, "width": 4 if name == "dots" else 2.5}, name=name,
            error_y={"type": "data", "symmetric": False, "array": [hi - m for m, _, hi in stats.values()], "arrayminus": [m - lo for m, lo, _ in stats.values()], "color": color, "thickness": 1}), row=1, col=1)
        print(f"{cyan}{name:32s} " + " ".join(f"{k}:{m:.2f}" for k, (m, _, _) in stats.items()) + endc)
    p0 = {i: dots[(i, 0)] for i in idxs}
    hard = [i for i in idxs if p0[i] < 0.3]
    fixed = lambda f, k: {i for i in hard if f[(i, k)] >= 0.6}
    ref = fixed(dots, 100)
    chance = len(ref) / len(hard)
    rows = {"dots, 95 instead of 100<br>(reference: re-measuring dots)": fixed(dots, 95)} | {name.replace(" (repeated)", ""): fixed(f, 100) for name, f in fillers.items() if name != "dots"}
    shares = {name: len(s & ref) / len(s) for name, s in rows.items()}
    fig.add_trace(go.Bar(x=list(rows), y=list(shares.values()), marker_color=["#8a8a85"] + SERIES[:4], text=[f"{len(s & ref)} of {len(s)}" for s in rows.values()], textposition="inside", showlegend=False), row=1, col=2)
    fig.add_hline(y=chance, line={"color": "#e8e6dc", "dash": "dot", "width": 1.5}, annotation_text=f"chance {chance:.2f}", annotation_position="bottom right", row=1, col=2)
    print(f"{cyan}questions wrong without filler (P0 < 0.3): {len(hard)}; fixed by 100 dots: {len(ref)}; " + ", ".join(f"{name.split('<br>')[0]} {len(rows[name] & ref)} of {len(rows[name])}" for name in rows) + endc)
    fig.update_xaxes(title_text="number of filler tokens (log scale)", type="log", tickvals=[k + 1 for k in ticks], ticktext=[str(k) for k in ticks], row=1, col=1)
    fig.update_yaxes(title_text="mean probability of the correct answer", range=[0.3, 0.7], row=1, col=1)
    fig.update_yaxes(title_text="share also fixed by 100 dots", range=[0, 1.02], row=1, col=2)
    fig.update_layout(
        title=(
            f"<b>Filler type doesn't matter much</b>  DeepSeek V4 Flash, {len(idxs)} two-step arithmetic questions, filler tokens between the question and 'Answer:'"
            "<br><span style='font-size:13px'>Fillers: dots ('Filler: . . .'), the Wikipedia article 'Tree', integers counted up or down from a random start below 10,000, independent random integers from 1 to 1000, and the capital alphabet 'A B ... Z' (one alphabet is 27 tokens; 100 repeats it).</span>"
            "<br><span style='font-size:13px'>The prompt's 10 worked examples carry the same kind of filler. P(correct): the model's probability of writing exactly the right number and stopping. Error bars: 95% intervals from resampling questions.</span>"
            f"<br><span style='font-size:13px'>In the alphabet's own run, 27 dots give {sum(alpha_run['dots'][(i, 27)] for i in idxs) / len(idxs):.2f} and 100 dots {sum(alpha_run['dots'][(i, 100)] for i in idxs) / len(idxs):.2f}, against {sum(alpha_run['alpha'][(i, 27)] for i in idxs) / len(idxs):.2f} and {sum(alpha_run['alpha'][(i, 100)] for i in idxs) / len(idxs):.2f} for the alphabet: one alphabet gives about two thirds of the gain of 27 dots, and repeating it adds nothing.</span>"
            f"<br><span style='font-size:13px'>Right: a question counts as fixed when P(correct) is below 0.3 without filler and at least 0.6 with it. Of the {len(hard)} questions below 0.3 without filler, 100 dots fix {len(ref)} (chance line: {chance:.2f});</span>"
            "<br><span style='font-size:13px'>each bar is the share of another filler's fixed questions that dots also fix. The grey bar re-measures dots at 95 instead of 100, a reference for how much agreement to expect from the same filler.</span>"
        ),
        height=820,
        width=1800,
        margin={"t": 300},
        legend={"title": "filler", "x": 0.0, "y": 1.0, "xanchor": "left", "bgcolor": "rgba(0,0,0,0)"},
        **DARK,
    )
    write_dark_html(fig, "figs/filler_types.html")
    print(f"{green}wrote figs/filler_types.html{endc}")

#%% Selector check: greedy answers for the same question under 9 differently seeded prompts (no filler after the question; nothing, or 10 / 100 tokens of dots, Wikipedia text, counting or random integers before it). Could a selector over these runs beat one run? Majority vote, most confident (highest P of its own greedy answer) and the oracle (any run right) against the single clean run; saves results/selector_{TAG}.json

run_selector = True
if run_selector:
    n_items, batch_size = 600, 16
    # n_items = 32
    conds = [("none", 0)] + [(kind, n) for kind in ["dots", "wiki", "count", "rand"] for n in [10, 100]]
    out = {}
    for kind, n in conds:
        pts = [prompt_ids(tok, shots, it, 0, "dots", (kind, n) if n else None) for it in items[:n_items]]
        cache = prefill(model, pts[0][0])
        ans, conf = [], []
        for i in pbar(range(0, n_items, batch_size), desc=f"{kind} {n}"):
            gens, first = decode(model, cache, [tail for _, tail in pts[i:i + batch_size]])
            ans += [parse_answer(tok.decode([x for x in g if x != EOS])) for g in gens]
            conf += [first[j, g[0]].exp().item() for j, g in enumerate(gens)]  # P of the first answer token, the greedy run's own confidence
        out[f"{kind}_{n}"] = {"answer": ans, "conf": conf}
        print(f"{cyan}{kind} {n}: greedy accuracy {sum(a == it['answer'] for a, it in zip(ans, items)) / n_items:.3f}{endc}")
    json.dump(out, open(f"results/selector_{TAG}.json", "w"))
    right = t.tensor([[out[c]["answer"][i] == items[i]["answer"] for c in out] for i in range(n_items)])  # [items, conds]
    answers = [[out[c]["answer"][i] for c in out] for i in range(n_items)]
    confs = t.tensor([[out[c]["conf"][i] for c in out] for i in range(n_items)])
    vote = [Counter(a).most_common(1)[0][0] == items[i]["answer"] for i, a in enumerate(answers)]
    most_conf = [answers[i][confs[i].argmax()] == items[i]["answer"] for i in range(n_items)]
    print(f"{green}single clean run {right[:, 0].float().mean():.3f}; mean over the 9 runs {right.float().mean():.3f}; majority vote {sum(vote) / n_items:.3f}; most confident {sum(most_conf) / n_items:.3f}; oracle (any run right) {right.any(1).float().mean():.3f}; all 9 agree {sum(len(set(a)) == 1 for a in answers) / n_items:.3f}{endc}")

#%% Figure for readers without context: greedy accuracy of the single plain run, of each of the 8 runs with a filler before the question, and of three ways of combining the 9 runs (majority vote, most confident, any run right), against one run with 100 dots after the question (results/knockout_{TAG}.json); Wilson intervals; written to figs/selector.html

plot_selector = True
if plot_selector:
    sel = json.load(open(f"results/selector_{TAG}.json"))
    ko = {(r["idx"], r["k"], r["cond"]): r["top1"] for r in json.load(open(f"results/knockout_{TAG}.json"))}
    n = len(sel["none_0"]["answer"])
    right = {c: [sel[c]["answer"][i] == items[i]["answer"] for i in range(n)] for c in sel}
    answers = [[sel[c]["answer"][i] for c in sel] for i in range(n)]
    confs = [[sel[c]["conf"][i] for c in sel] for i in range(n)]
    names = {"none_0": "plain prompt", "dots": "dots", "wiki": "Wikipedia text", "count": "counting", "rand": "random integers"}
    bars = [("plain prompt,<br>one run", right["none_0"], "#888888")]
    bars += [(f"{names[c.split('_')[0]]} before<br>the question,<br>{c.split('_')[1]} tokens", right[c], SERIES[0]) for c in sel if c != "none_0"]
    bars += [
        ("majority vote<br>over the 9 runs", [Counter(a).most_common(1)[0][0] == items[i]["answer"] for i, a in enumerate(answers)], SERIES[1]),
        ("most confident<br>of the 9 runs", [answers[i][max(range(9), key=lambda j: confs[i][j])] == items[i]["answer"] for i in range(n)], SERIES[1]),
        ("any of the 9<br>runs right", [any(right[c][i] for c in sel) for i in range(n)], SERIES[1]),
        ("100 dots after<br>the question,<br>one run", [ko[(i, 100, "full")] for i in range(n)], SERIES[2]),
    ]
    acc = [sum(v) / n for _, v, _ in bars]
    ci = [wilson(sum(v), n) for _, v, _ in bars]
    fig = go.Figure(go.Bar(x=[b[0] for b in bars], y=[round(a, 4) for a in acc], marker_color=[b[2] for b in bars], text=[f"{a:.3f}" for a in acc], textposition="outside",
        error_y={"type": "data", "symmetric": False, "array": [h - a for a, (_, h) in zip(acc, ci)], "arrayminus": [a - l for a, (l, _) in zip(acc, ci)]},
        hovertext=[f"{b[0].replace('<br>', ' ')}<br>greedy accuracy {a:.4f} [{l:.3f}, {h:.3f}]" for b, a, (l, h) in zip(bars, acc, ci)], hovertemplate="%{hovertext}<extra></extra>"))
    agree = sum(len(set(a)) == 1 for a in answers) / n
    fig.update_layout(
        title={"y": 0.95, "yanchor": "top", "text": (
            f"<b>Selecting among 9 differently-prefixed runs gives at most {max(acc[9:11]):.2f}, against {acc[0]:.2f} for one plain run and {acc[-1]:.2f} for one run with 100 filler dots after the question</b>"
            f"<br><span style='font-size:13px'>DeepSeek V4 Flash, {n} two-step arithmetic questions, greedy answers. The idea under test: the filler helps because many filler positions re-run the same computation from different starting points and the answer selects among them.</span>"
            f"<br><span style='font-size:13px'>If so, running the whole model 9 times with different text before the question (nothing, or 10 / 100 tokens of dots, Wikipedia text, counting or random integers; no filler after the question) and selecting should help too. The runs do disagree (all 9 agree on {agree:.0%} of questions),</span>"
            "<br><span style='font-size:13px'>but selecting among them recovers little. 'Most confident' picks the run with the highest probability on its own first answer token. 'Any run right' is the ceiling for any selector. Error bars: 95% Wilson intervals.</span>"
        )},
        height=820, width=1900, margin={"t": 180, "b": 110}, **DARK,
    )
    fig.update_yaxes(title_text="greedy accuracy", range=[0, 1.0])
    fig.update_xaxes(tickangle=0, tickfont={"size": 12})
    write_dark_html(fig, "figs/selector.html")
    print(f"{green}wrote figs/selector.html{endc}")

#%% The full prompt as the model sees it, one box per token (hover for position, id and repr), the token where the answer is read outlined; written to figs/prompt_k{k}.html. Also the question's own turn alone, each token tinted by its prompt region with a legend, written to figs/prompt_k{k}_turn.html

save_prompt_html = True
if save_prompt_html:
    k = 10
    # k = 0
    # k = 100
    item = items[0]
    prefix, tail = prompt_ids(tok, shots, item, k)
    ids = prefix + tail
    head = f"Prompt for question {item['idx']} with {k} filler dots: {len(ids)} tokens ({len(prefix)} shared by every question: system prompt and 10 worked examples; {len(tail)} for this question). The outlined last token is where the model's answer is read; the correct answer is {item['answer']}."
    strip = toks_html([tok.decode([i]) for i in ids], ids, len(ids) - 1)
    open(f"figs/prompt_k{k}.html", "w").write(f"<!doctype html><html><head><meta charset='utf-8'><title>Prompt, {k} dots</title><style>html, body {{ background: #111; margin: 0; }}</style></head><body><div style='background:#111;color:#ddd;font:12px monospace;padding:8px'><div style='margin-bottom:6px;font-weight:bold'>{html.escape(head)}</div>{strip}</div></body></html>")
    print(f"{green}wrote figs/prompt_k{k}.html ({len(ids)} tokens){endc}")
    tint = {"q_defs": ("variable definitions", "57,135,229"), "q_line": ("question", "25,158,112"), "label": ("'Filler:' label", "201,133,0"), "filler": ("filler dots", "150,150,140"), "post": ("'Answer:'", "217,89,38"), "last": ("answer is read here", "217,89,38")}
    region_of = {p: name for name, pos in regions(tok, tail, k).items() for p in pos}
    spans = "".join(f"<span data-h='pos {i} &middot; id {tid} &middot; {html.escape(repr(tok.decode([tid])))}' style='{f'--v:rgba({tint[region_of[i]][1]},0.38);' if i in region_of else ''}{'outline:1px solid #fc6' if region_of.get(i) == 'last' else ''}'>{html.escape(tok.decode([tid])).replace(chr(10), '↵' + chr(10))}</span>" for i, tid in enumerate(tail))
    legend = "".join(f"<span style='display:inline-block;margin:0 14px 4px 0'><span style='display:inline-block;width:11px;height:11px;margin-right:5px;vertical-align:-1px;background:rgba({rgb},0.6)'></span>{label}</span>" for name, (label, rgb) in tint.items() if name != "last")
    turn = f"<div style='background:#1c1c1c;color:#ddd;font:13px monospace;padding:10px 12px'><div style='margin-bottom:6px'>{legend}</div>{TOKS_CSS}<style>.tk span{{box-shadow:inset -1px 0 0 #1c1c1c}}</style><div class='tk' style='white-space:pre-wrap;line-height:1.8'>{spans}</div></div>"
    open(f"figs/prompt_k{k}_turn.html", "w").write(f"<!doctype html><html><head><meta charset='utf-8'><title>Question turn, {k} dots</title><style>html, body {{ background: #1c1c1c; margin: 0; }}</style></head><body>{turn}</body></html>")
    print(f"{green}wrote figs/prompt_k{k}_turn.html ({len(tail)} tokens; answer {item['answer']}){endc}")
