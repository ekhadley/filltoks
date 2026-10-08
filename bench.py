#%% Setup: data, model, tokenizer

import csv
import json
import math
from collections import Counter

import torch as t
from transformers import FineGrainedFP8Config
from transformers.cache_utils import DynamicCache

from mechtools import *

from prompts import parse_answer
from utils import MODEL_ID, EOS, tiny_bridge, prompt_ids, answer_ids, prefill, tail_logits, decode, answer_logprob

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
