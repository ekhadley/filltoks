#%% Setup: Olivia's equation task on Qwen3.6 35B-A3B (bf16, one GPU, plain HF), her prompt rendered with Qwen's chat template in non-thinking mode

import json
import math

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from mechtools import *

from prompts import build_messages

QWEN_ID = "Qwen/Qwen3.6-35B-A3B"
TAG = "qwen36_35b"
NAME = "Qwen3.6 35B-A3B"

t.set_grad_enabled(False)
items = [json.loads(l) for l in open("data/eval.jsonl")]
shots = [json.loads(l) for l in open("data/fewshot.jsonl")]
tok = AutoTokenizer.from_pretrained(QWEN_ID)
EOT = tok.convert_tokens_to_ids("<|im_end|>")

def qwen_ids(item: dict, k: int) -> list[int]:
    """Olivia's 10-shot k-dot prompt; the shots' assistant turns are bare numbers, the last turn ends with the template's empty think block."""
    text = tok.apply_chat_template(build_messages(shots, item, k=k), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), text[-80:]
    return tok(text, add_special_tokens=False).input_ids

def score(rows: list[list[int]], answers: list[int]) -> tuple[list[float], list[bool]]:
    """Per row: teacher-forced log P(answer digits, then <|im_end|>), and whether greedy decoding writes exactly that (argmax at every step). Rows are right-padded; fp32 unembedding of the final normed state."""
    tgts = [tok(str(a), add_special_tokens=False).input_ids + [EOT] for a in answers]
    seqs = [r + tg[:-1] for r, tg in zip(rows, tgts)]
    T = max(map(len, seqs))
    ids = t.tensor([s + [EOT] * (T - len(s)) for s in seqs], device=DEV)
    mask = t.tensor([[1] * len(s) + [0] * (T - len(s)) for s in seqs], device=DEV)
    h = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
    logp, greedy = [], []
    for i, (r, tg) in enumerate(zip(rows, tgts)):
        lp = (h[i, len(r) - 1:len(r) - 1 + len(tg)].float() @ W.T).log_softmax(-1)
        logp.append(lp[range(len(tg)), tg].sum().item())
        greedy.append((lp.argmax(-1).tolist() == tg))
    return logp, greedy

#%% Load the model (72 GB in bf16) on one GPU

load_model = True
if load_model:
    DEV = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(QWEN_ID, dtype=t.bfloat16, device_map=DEV).eval()
    W = model.lm_head.weight.float()
    print(f"{cyan}{type(model).__name__}, {t.cuda.memory_allocated(0) / 1e9:.1f} GB on {DEV}{endc}")

#%% Smoke test: greedy generations on a few items at 0 and 100 dots, against the teacher-forced score

smoke_test = True
if smoke_test:
    for k in [0, 100]:
        rows = [qwen_ids(it, k) for it in items[:8]]
        logp, greedy = score(rows, [it["answer"] for it in items[:8]])
        for it, r, lp, g in zip(items[:8], rows, logp, greedy):
            out = model.generate(t.tensor([r], device=DEV), max_new_tokens=8, do_sample=False)[0, len(r):]
            print(f"{cyan}k={k} item {it['idx']} ({len(r)} tokens): generated {tok.decode(out)!r}, answer {it['answer']}, greedy-correct by score {g}, P(correct) {math.exp(lp):.3f}{endc}")

#%% Dose sweep: exact P(correct answer + <|im_end|>) and greedy correctness for every item at 0, 5, ..., 100 dots; saves results/dose_{TAG}.json (the same records as the V4 dose sweep, plus 'greedy')

run_dose = True
if run_dose:
    ks = list(range(0, 101, 5))
    n_items = 600
    batch_size = 16
    records = []
    for k in ks:
        rows = [qwen_ids(it, k) for it in items[:n_items]]
        for i in pbar(range(0, n_items, batch_size), desc=f"k={k}"):
            batch = items[i:i + batch_size]
            logp, greedy = score(rows[i:i + batch_size], [it["answer"] for it in batch])
            records += [{"idx": it["idx"], "k": k, "logp": lp, "greedy": g} for it, lp, g in zip(batch, logp, greedy)]
        done = [r for r in records if r["k"] == k]
        print(f"{cyan}k={k}: mean P(correct) {sum(math.exp(r['logp']) for r in done) / len(done):.3f}, greedy {sum(r['greedy'] for r in done) / len(done):.3f}{endc}")
        json.dump(records, open(f"results/dose_{TAG}.json", "w"))
    print(f"{green}saved results/dose_{TAG}.json{endc}")

#%% Uplift metrics and per-question histograms for readers without context: P(correct) at 10 and 100 dots minus at 0 (top) and the fraction of the no-filler error removed for questions with P0 < 0.5 (bottom), each against the same quantity from 95 to 100 dots as a per-question noise reference (outline); log counts; written to figs/uplift_hist_{TAG}.html. Prints mean / greedy accuracy with paired bootstrap intervals, tail counts against the reference, by multipliers, and the overlap of the risers with V4 Flash's

plot_uplift_hist = True
if plot_uplift_hist:
    recs = json.load(open(f"results/dose_{TAG}.json"))
    lp = {(r["idx"], r["k"]): r["logp"] for r in recs}
    gr = {(r["idx"], r["k"]): r["greedy"] for r in recs}
    idxs = sorted({r["idx"] for r in recs})
    p = {k: t.tensor([math.exp(lp[(i, k)]) for i in idxs]) for k in [0, 5, 10, 90, 95, 100]}
    g = {k: t.tensor([float(gr[(i, k)]) for i in idxs]) for k in [0, 10, 100]}
    boot = t.randint(0, len(idxs), (5000, len(idxs)), generator=t.Generator().manual_seed(0))
    ci = lambda d: f"{d.mean():+.3f} [{d[boot].mean(1).quantile(0.025):+.3f}, {d[boot].mean(1).quantile(0.975):+.3f}]"
    print(f"{cyan}{NAME}, {len(idxs)} questions: mean P(correct) " + " / ".join(f"{p[k].mean():.3f}" for k in [0, 10, 100]) + ", greedy " + " / ".join(f"{g[k].mean():.3f}" for k in [0, 10, 100]) + f" at 0 / 10 / 100 dots; uplift P 10: {ci(p[10] - p[0])}, 100: {ci(p[100] - p[0])}; greedy 10: {ci(g[10] - g[0])}, 100: {ci(g[100] - g[0])}{endc}")
    cc = t.tensor([[items[i]["chain"]["c1x"] // items[i]["chain"]["x"], items[i]["coefficient"]] for i in idxs])
    for c1 in [2, 3]:
        for c2 in [2, 3]:
            m = (cc[:, 0] == c1) & (cc[:, 1] == c2)
            print(f"  ×{c1} then ×{c2} (n={m.sum().item()}): mean P(correct) " + " / ".join(f"{p[k][m].mean():.2f}" for k in [0, 10, 100]))
    frac = lambda a, b: ((b - a) / (1 - a))[a < 0.5]  # fraction of a's error removed by b, for questions with a < 0.5
    noise_d, noise_f = p[100] - p[95], frac(p[95], p[100])
    room = p[0] < 0.5
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"10 dots: mean change {(p[10] - p[0]).mean():+.3f}", f"100 dots: mean change {(p[100] - p[0]).mean():+.3f}", f"10 dots: fraction of error removed, {room.sum().item()} questions with P0 < 0.5", f"100 dots: fraction of error removed, {room.sum().item()} questions with P0 < 0.5"], vertical_spacing=0.18)
    edges = t.arange(-1, 1.0001, 0.05)
    step = lambda x, legend: go.Scatter(x=edges.tolist(), y=t.histogram(x.clamp(-1, 1), bins=edges).hist.tolist() + [0], mode="lines", line={"color": "#e8e6dc", "width": 1.5, "shape": "hv"}, name="95 → 100 dots (noise reference)", showlegend=legend)
    for col, k in enumerate([10, 100], 1):
        d, f = p[k] - p[0], frac(p[0], p[k])
        fig.add_trace(go.Histogram(x=d.tolist(), xbins={"start": -1, "end": 1, "size": 0.05}, marker_color=SERIES[col - 1], name=f"0 → {k} dots"), row=1, col=col)
        fig.add_trace(go.Histogram(x=f.clamp(min=-1).tolist(), xbins={"start": -1, "end": 1, "size": 0.05}, marker_color=SERIES[col - 1], showlegend=False), row=2, col=col)
        fig.add_trace(step(noise_d, col == 1), row=1, col=col)
        fig.add_trace(step(noise_f, False), row=2, col=col)
        for row in [1, 2]:
            fig.add_vline(x=0, line={"color": "#e8e6dc", "width": 1, "dash": "dot"}, row=row, col=col)
        tails = lambda d: "up > " + ", ".join(f"{c}: {(d > c).sum().item()}" for c in [0.1, 0.3, 0.5]) + "; down < " + ", ".join(f"-{c}: {(d < -c).sum().item()}" for c in [0.1, 0.3, 0.5])
        print(f"0 -> {k} dots: median {d.median():+.3f}, sd {d.std():.3f}, |change| > 0.1: {(d.abs() > 0.1).float().mean():.2f} of items, {tails(d)}; fraction of error removed (P0 < 0.5, n={room.sum().item()}): mean {f.mean():+.3f}, median {f.median():+.3f}, > 0.5: {(f > 0.5).sum().item()}, > 0.9: {(f > 0.9).sum().item()}, < -0.5: {(f < -0.5).sum().item()}")
    print(f"95 -> 100 dots (reference): median {noise_d.median():+.3f}, sd {noise_d.std():.3f}, |change| > 0.1: {(noise_d.abs() > 0.1).float().mean():.2f} of items, {tails(noise_d)}; 90 -> 100: sd {(p[100] - p[90]).std():.3f}, {tails(p[100] - p[90])}; 5 -> 10: sd {(p[10] - p[5]).std():.3f}, {tails(p[10] - p[5])}; fraction removed 95 -> 100 (P95 < 0.5, n={len(noise_f)}): > 0.5: {(noise_f > 0.5).sum().item()}, < -0.5: {(noise_f < -0.5).sum().item()}")
    v4 = json.load(open("results/dose_v4flash_packed.json"))
    v4p = {(r["idx"], r["k"]): math.exp(r["logp"]) for r in v4}
    up_v4 = t.tensor([v4p[(i, 100)] - v4p[(i, 0)] > 0.5 for i in idxs])
    for k in [10, 100]:
        up = p[k] - p[0] > 0.3
        a, b, c, d_ = (up & up_v4).sum().item(), (up & ~up_v4).sum().item(), (~up & up_v4).sum().item(), (~up & ~up_v4).sum().item()
        print(f"risers 0 -> {k} dots (> 0.3, n={up.sum().item()}) that are V4 Flash 0 -> 100 risers (> 0.5, n={up_v4.sum().item()}): {a}; odds ratio {(a * d_) / max(b * c, 1):.2f}")
    fig.update_layout(
        title=(
            f"<b>Per-question change in the chance of a correct answer with filler dots</b>  {NAME}, {len(idxs)} two-step arithmetic questions"
            f"<br><span style='font-size:13px'>Mean probability of the correct answer {p[0].mean():.2f} with no filler, {p[10].mean():.2f} with 10 dots, {p[100].mean():.2f} with 100 (DeepSeek V4 Flash: 0.47, 0.48, 0.62). Probability of the correct answer: the model's probability of writing exactly the right number and stopping.</span>"
            "<br><span style='font-size:13px'>Top: each question's probability with k dots ('Filler: . . .' between the question and 'Answer:', the 10 worked examples carrying the same filler) minus with no filler. Bottom: the fraction of the no-filler error removed, (with − without) / (1 − without), for questions below 0.5 without filler; values below −1 shown at −1.</span>"
            "<br><span style='font-size:13px'>Outline: the same quantity going from 95 to 100 dots, where filler length should no longer matter: how far a question moves when the prompt changes slightly. Counts on a log scale.</span>"
        ),
        height=950,
        width=1400,
        margin={"t": 170},
        barmode="overlay",
        **DARK,
    )
    fig.update_xaxes(title_text="P(correct) with filler minus P(correct) without", range=[-1, 1], row=1)
    fig.update_xaxes(title_text="fraction of the no-filler error removed", range=[-1, 1], row=2)
    fig.update_yaxes(title_text="questions", type="log", col=1)
    fig.update_yaxes(type="log", col=2)
    write_dark_html(fig, f"figs/uplift_hist_{TAG}.html")
    print(f"{green}wrote figs/uplift_hist_{TAG}.html{endc}")
