#%% Setup: Brauer et al.'s logit-lens decode of the chain x -> c1x -> y -> c2y -> answer at every (layer, position) from the question end through the filler to the answer position

import json

import torch as t
from transformers import FineGrainedFP8Config

from mechtools import *

from utils import MODEL_ID, tiny_bridge, fp32_routers, prompt_ids, regions, answer_ids, prefill, tail_logits, with_coefs

t.set_grad_enabled(False)
items = [json.loads(l) for l in open("data/eval.jsonl")]
shots = [json.loads(l) for l in open("data/fewshot.jsonl")]

TAG = "tiny"
model = tiny_bridge()
# TAG = "v4flash"
# model = load_bridge(MODEL_ID, quantization_config=FineGrainedFP8Config(dequantize=True))
# fp32_routers(model)
tok = model.tokenizer
hf = model.original_model
L = model.cfg.n_layers
TARGETS = ["x", "c1x", "y", "c2y", "answer"]
N_Q = 24  # question-side positions kept before the 'Filler:' label

def space_rows(fig, gap_px: float) -> list[tuple[float, float]]:
    """Re-space a faceted figure's rows `gap_px` pixels apart, so one row's tick labels clear the next row's titles; moves the facet titles with their rows. Returns the row domains, top row first."""
    gap = gap_px / (fig.layout.height - fig.layout.margin.t - fig.layout.margin.b)
    old = sorted({tuple(ax["domain"]) for name, ax in fig.layout.to_plotly_json().items() if name.startswith("yaxis")}, key=lambda d: -d[0])
    h = (1 - gap * (len(old) - 1)) / len(old)
    new = [(1 - (j + 1) * h - j * gap, 1 - j * h - j * gap) for j in range(len(old))]
    for name, ax in fig.layout.to_plotly_json().items():
        if name.startswith("yaxis"):
            fig.layout[name].domain = new[old.index(tuple(ax["domain"]))]
    for a in fig.layout.annotations:  # facet titles sit on the top edge of their row
        rows = [j for j, (_, top) in enumerate(old) if a.yref == "paper" and abs(a.y - top) < 1e-6]
        if rows:
            a.y = new[rows[0]][1]
    return new

score_text = {  # dropdown label, colorbar title, description
    "top1": ("top-1 share", "share of<br>questions", "Color: share of questions where the model's internal state at that layer and token, read through its own output layer (the 'logit lens'), ranks that value first among the numbers 0–999."),
    "top5": ("top-5 share", "share with value<br>in the top 5", "Color: share of questions where that value is among the 5 numbers (of 0–999) that the model's internal state at that layer and token ranks highest, read through its own output layer (the 'logit lens'). Unlike first place only, this counts near-misses."),
    "rank": ("mean 1/(1+rank)", "average<br>1/(1 + rank)", "Color: average over questions of 1/(1 + rank), where rank is the value's place among the numbers 0–999 as ranked by the model's internal state at that layer and token, read through its own output layer (the 'logit lens').<br>1 = always first, 0.5 = typically second, near 0 = far down the list."),
    "within5": ("top number within ±5", "share with top<br>number within ±5", "Color: share of questions where the number that the model's internal state at that layer and token ranks first (read through its own output layer, the 'logit lens'; numbers 0–999)<br>is within 5 of the value, so near-misses like 226 for 228 count."),
    "resid": ("residualized probability", "extra probability<br>vs other questions", "Color: the probability the model's state at that layer and token gives this question's value (logit lens over the numbers 0–999), minus the average probability it gives the same number<br>on the other questions, averaged over questions. 0 = no more than on any other question, 1 = all of it. This removes numbers that are likely on every question."),
}

def scored(r: dict, jt: t.Tensor, score: str) -> t.Tensor:
    """[items, layers, positions, targets]: how strongly each cell's readout points to each chain value under `score`."""
    if score == "top1":
        return (r["rank"] == 0).float()
    if score == "top5":
        return (r["rank"] < 5).float()
    if score == "rank":
        return 1 / (1 + r["rank"].float())
    if score == "within5":
        return ((r["top1"].long()[..., None] - jt[:, None, None, :]).abs() <= 5).float()
    own = r["logp"].float().exp()
    n = own.shape[0]
    return own - (r["num_p_mean"][:, :, jt].permute(2, 0, 1, 3) * n - own) / (n - 1)  # resid: minus the same number's mean probability at this cell over the other questions

def score_dropdown(fig, heats: dict, titles: dict):
    """A dropdown at the top right that swaps every facet's data for another scoring (the keys of `heats`, in order) and changes the title and colorbar label with it."""
    first = next(iter(heats.values()))
    assert len(fig.data) == len(first) and all(t.allclose(t.tensor(tr.z, dtype=t.float32), first[i]) for i, tr in enumerate(fig.data))  # a button restyles trace i with facet i
    fig.update_layout(title={"y": 0.99, "yanchor": "top"}, updatemenus=[{  # the title starts at the top of the margin so the dropdown fits below it
        "buttons": [{"label": score_text[s][0], "method": "update", "args": [{"z": heats[s].round(decimals=3).tolist()}, {"title.text": titles[s], "coloraxis.colorbar.title.text": score_text[s][1]}]} for s in heats],
        "x": 1, "xanchor": "right", "y": 1.025, "yanchor": "bottom", "bgcolor": "#333333", "bordercolor": "#888888", "font": {"color": "#dddddd"},
    }])

hc_head, norm, W = hf.model.hc_head.original_component, hf.model.norm.original_component, hf.lm_head.weight
assert all(len(answer_ids(tok, n)) == 1 for n in range(1000))
num_ids = t.tensor([answer_ids(tok, n)[0] for n in range(1000)], device=W.device)
W32 = W.float()
W_num = W32[num_ids]

def collapse(stack: Tensor) -> Tensor:
    """The model's own readout of a [B, P, 4, D] stream stack up to the unembedding: hc_head's input-dependent collapse, then the final RMSNorm. fp32 [B, P, D] on the unembedding's device."""
    h = hc_head(stack.to(hc_head.hc_fn.device))
    return norm(h.to(norm.weight.device)).to(W.device).float()

def position_names(k: int) -> list[str]:
    return [f"q{j}" for j in range(-N_Q, 0)] + ["F", "iller", ":"] + [f".{j}" for j in range(1, k + 1)] + ["Answer", ":", "Asst", "</think>"]

def run_lens(cache, tails: list[list[int]], chains: list[dict], n_keep: int, targets: list[str] = TARGETS) -> dict:
    """One forward of right-padded tails. At every decoder layer's output and each row's last n_keep positions: the number lens's top-1 value, and each chain value's rank and log-prob among the numbers 0..999; also every number's probability summed over the batch's rows ([1, layer, position, 1000], for per-cell averages over items). At the answer position: the full-vocab lens's log-prob and rank of the answer token per layer. Also the model's own answer-position log-prob of the answer and whether it is the argmax."""
    B, lens_ = len(tails), [len(tl) for tl in tails]
    tgt = t.tensor([[ch[name] for name in targets] for ch in chains], device=W.device)
    ans = num_ids[tgt[:, -1]]
    out = {key: [] for key in ["top1", "rank", "logp", "ans_logp", "ans_rank"]}
    num_sum = []
    def hook(act, hook):
        assert act.ndim == 4 and act.shape[0] == B, act.shape
        pos = t.tensor([[n - n_keep + j for j in range(n_keep)] for n in lens_], device=act.device)
        h = collapse(act[t.arange(B, device=act.device)[:, None], pos])
        logp = (h @ W_num.T).log_softmax(-1)
        tl = logp.gather(-1, tgt[:, None, :].expand(-1, n_keep, -1))
        out["top1"].append(logp.argmax(-1).short().cpu())
        out["rank"].append((logp[:, :, None, :] > tl[..., None]).sum(-1).short().cpu())
        out["logp"].append(tl.half().cpu())
        num_sum.append(logp.exp().sum(0).cpu())
        full = (h[:, -1] @ W32.T).log_softmax(-1)
        a = full.gather(-1, ans[:, None])[:, 0]
        out["ans_logp"].append(a.cpu())
        out["ans_rank"].append((full > a[:, None]).sum(-1).cpu())
        if hook.layer() == L - 1:
            out["last_full"] = full
    logp = tail_logits(model, cache, tails, [(f"blocks.{l}.hook_out", hook) for l in range(L)])[:, -1].log_softmax(-1).to(W.device)
    top = logp.topk(10, dim=-1).indices
    gap = (out.pop("last_full").gather(-1, top) - logp.gather(-1, top)).abs().max().item()
    assert gap < 0.2, f"last-layer lens differs from the model's logits by {gap:.3f} nats on its top-10"
    res = {key: t.stack(v, 1) for key, v in out.items()}
    res["model_logp"] = logp.gather(-1, ans[:, None])[:, 0].cpu()
    res["correct"] = (logp.argmax(-1) == ans).cpu()
    res["gap"] = t.full((B,), gap)
    res["num_p_sum"] = t.stack(num_sum)[None]
    return res

#%% Run the lens over every item with all chain values single-token (0..999) at each filler length; saves results/lens_{TAG}.pt

run_lens_all = True
if run_lens_all:
    ks = [0, 25, 100]
    n_items = 600
    # n_items = 32
    batch_size = 16
    use = [it for it in items[:n_items] if all(0 <= v < 1000 for v in it["chain"].values())]
    res = {}
    for k in ks:
        pts = [prompt_ids(tok, shots, it, k) for it in use]
        assert all(prefix == pts[0][0] for prefix, _ in pts)
        n_keep = k + 7 + N_Q
        assert regions(tok, pts[0][1], k)["label"][0] == len(pts[0][1]) - n_keep + N_Q and len(position_names(k)) == n_keep
        cache = prefill(model, pts[0][0])
        chunks = [run_lens(cache, [tl for _, tl in pts[i:i + batch_size]], [it["chain"] for it in use[i:i + batch_size]], n_keep) for i in pbar(range(0, len(use), batch_size), desc=f"k={k}")]
        res[k] = {key: t.cat([c[key] for c in chunks]) for key in chunks[0]}
        res[k]["num_p_mean"] = res[k].pop("num_p_sum").sum(0) / len(use)
        res[k]["idx"] = t.tensor([it["idx"] for it in use])
        print(f"{cyan}k={k}: {len(use)} items, greedy correct {res[k]['correct'].float().mean():.3f}, mean P(answer) {res[k]['model_logp'].exp().mean():.3f}, max last-layer lens gap {res[k]['gap'].max():.4f} nats{endc}")
    t.save(res, f"results/lens_{TAG}.pt")
    print(f"{green}saved results/lens_{TAG}.pt{endc}")

#%% Coefficient variants: base items whose every (c1, c2) combination within each coefficient set keeps all chain values in 0..999, each rewritten with both hop coefficients set (x, k1, k2 and the other definitions unchanged; the shots keep their 2s and 3s); saves results/lens_coef_{TAG}.pt

run_lens_coef = True
if run_lens_coef:
    ks = [0, 5, 10, 15]
    coef_sets = [(2, 3), (2, 4), (3, 5)]
    max_base = 1000
    # max_base = 4
    batch_size = 16
    base = [it for it in items if all(0 <= v < 1000 for s in coef_sets for c1 in s for c2 in s for v in with_coefs(it, c1, c2)["chain"].values())][:max_base]
    meta = t.tensor([[si, c1, c2, it["idx"]] for si, s in enumerate(coef_sets) for c1 in s for c2 in s for it in base])
    variants = [with_coefs(items[i], c1, c2) for _, c1, c2, i in meta.tolist()]
    print(f"{cyan}{len(base)} base items, {len(variants)} variants{endc}")
    res = {}
    for k in ks:
        pts = [prompt_ids(tok, shots, it, k) for it in variants]
        assert all(prefix == pts[0][0] for prefix, _ in pts)
        n_keep = k + 7 + N_Q
        assert regions(tok, pts[0][1], k)["label"][0] == len(pts[0][1]) - n_keep + N_Q and len(position_names(k)) == n_keep
        cache = prefill(model, pts[0][0])
        chunks = [run_lens(cache, [tl for _, tl in pts[i:i + batch_size]], [it["chain"] for it in variants[i:i + batch_size]], n_keep) for i in pbar(range(0, len(variants), batch_size), desc=f"k={k}")]
        res[k] = {key: t.cat([c[key] for c in chunks]) for key in chunks[0]}
        res[k]["num_p_mean"] = res[k].pop("num_p_sum").sum(0) / len(variants)
        res[k]["meta"] = meta
        print(f"{cyan}k={k}: greedy correct " + ", ".join(f"c1={c1} c2={c2} {res[k]['correct'][(meta[:, 1] == c1) & (meta[:, 2] == c2)].float().mean():.3f}" for c1, c2 in sorted({(a, b) for s in coef_sets for a in s for b in s})) + f"; max last-layer lens gap {res[k]['gap'].max():.4f} nats{endc}")
    t.save({"coef_sets": coef_sets, "res": res}, f"results/lens_coef_{TAG}.pt")
    print(f"{green}saved results/lens_coef_{TAG}.pt{endc}")

#%% 2-fact addition (Brauer et al.'s released set, data/2fact_addition.json, their 5 shots): the lens for fact 1, fact 2 and their sum at 0 and 10 dots (10 is this task's uplift peak), over the question line, label, dots and answer tokens; saves results/lens2_{TAG}.pt

TARGETS2 = ["a1", "a2", "answer"]
as_item2 = lambda e, i: {"idx": i, "type": "2fact_addition", "question": f"What is {e['fact_phrase_1']} plus {e['fact_phrase_2']}?", "answer": e["answer"], "chain": {"a1": e["fact_value_1"], "a2": e["fact_value_2"], "answer": e["answer"]}}
d2 = json.load(open("data/2fact_addition.json"))
shots2 = [as_item2(e, i) for i, e in enumerate(d2["few_shot_facts"])]
items2 = [as_item2(e, i) for i, e in enumerate(d2["examples"])]

run_lens_2fact = True
if run_lens_2fact:
    ks2 = [0, 10]
    n_items2 = 1500
    # n_items2 = 32
    batch_size = 16
    use2 = items2[:n_items2]
    res = {}
    for k in ks2:
        pts = [prompt_ids(tok, shots2, it, k) for it in use2]
        assert all(prefix == pts[0][0] for prefix, _ in pts)
        n_q = min(regions(tok, tail, k)["label"][0] for _, tail in pts)  # question-side positions kept per row (counted back from the label): the shortest question line, so every row has them
        n_keep = k + 7 + n_q
        cache = prefill(model, pts[0][0])
        chunks = [run_lens(cache, [tl for _, tl in pts[i:i + batch_size]], [it["chain"] for it in use2[i:i + batch_size]], n_keep, TARGETS2) for i in pbar(range(0, len(use2), batch_size), desc=f"k={k}")]
        res[k] = {key: t.cat([c[key] for c in chunks]) for key in chunks[0]}
        res[k]["num_p_mean"] = res[k].pop("num_p_sum").sum(0) / len(use2)
        res[k]["idx"] = t.tensor([it["idx"] for it in use2])
        res[k]["n_q"] = n_q
        print(f"{cyan}k={k}: {len(use2)} items, {n_q} question positions kept, greedy correct {res[k]['correct'].float().mean():.3f}, mean P(answer) {res[k]['model_logp'].exp().mean():.3f}, max last-layer lens gap {res[k]['gap'].max():.4f} nats{endc}")
    t.save(res, f"results/lens2_{TAG}.pt")
    print(f"{green}saved results/lens2_{TAG}.pt{endc}")

#%% Heatmaps for readers without context, written to figs/lens_k{k}.html: at each filler length, how strongly the model's state at each (layer, token) points to x, y, c2*y or the answer; a dropdown switches the scoring.
# Rows: by whether the model is right at 0 and 100 dots (same questions at every length), then by the two multipliers on the original questions. With coefswap, extra figures with rows by multiplier on the coefficient-swapped questions.
# Scorings: top1 (share with the value ranked first), top5, rank (mean 1/(1+rank)), within5 (top number within 5 of the value), resid (Brauer et al.'s residualized readout: this question's probability of its value minus the same number's mean probability at that cell on the other questions; only where the results carry num_p_mean)

plot_lens_heatmaps = True
if plot_lens_heatmaps:
    min_layer = 20
    n_show_q = 8
    ks = [0, 10, 100]
    # ks = [0, 5, 10, 15, 25, 100]
    coefswap = False
    res = {**t.load(f"results/lens_{TAG}.pt"), **t.load(f"results/lens_{TAG}_small.pt"), **t.load(f"results/lens_{TAG}_numdist.pt")}
    coef_sets = t.load(f"results/lens_coef_{TAG}.pt")["coef_sets"]
    coef_res = {**t.load(f"results/lens_coef_{TAG}.pt")["res"], **t.load(f"results/lens_coef_{TAG}_k100.pt")["res"]}
    assert all(t.equal(r["idx"], res[0]["idx"]) for r in res.values())
    ex = items[0]
    ch = ex["chain"]
    example = f"Example question: <i>{ex['definitions'][0][0]} = {ch['x']} · {ex['queried_term']} = {dict(ex['definitions'])[ex['queried_term']]} · (3 more definitions) · {ex['question']}</i>  →  x = {ch['x']} (written), y = {ex['queried_term']} = {ch['y']} (never written), c2·y = {ch['c2y']}, answer = {ch['answer']}"
    cols = [("x", "x: the given number"), ("y", "y: the hidden first result"), ("c2y", "c2·y: the second product"), ("answer", "the final answer")]
    c0, c100 = res[0]["correct"], res[100]["correct"]
    groups = {
        f"<b>Always right</b><br>right with 0 dots<br>and with 100<br>n={(c0 & c100).sum().item()}": c0 & c100,
        f"<b>Flipped by filler</b><br>wrong with 0 dots,<br>right with 100<br>n={(~c0 & c100).sum().item()}": ~c0 & c100,
        f"<b>Always wrong</b><br>wrong with 0 dots<br>and with 100<br>n={(~c0 & ~c100).sum().item()}": ~c0 & ~c100,
    }
    tgt = t.tensor([[items[i]["chain"][name] for name in TARGETS] for i in res[0]["idx"].tolist()])
    cc = tgt[:, [1, 3]] // tgt[:, [0, 2]]
    jobs = []
    for k in ks:
        r = res[k]
        coef_rows = {f"<b>{name}</b><br>n={m.sum().item()}<br>accuracy {r['correct'][m].float().mean():.0%}" + (f"<br>(0 dots: {c0[m].float().mean():.0%})" if k else ""): m for name, m in [("×2 then ×2", (cc == 2).all(1)), ("One ×2, one ×3<br>(either order)", cc[:, 0] != cc[:, 1]), ("×3 then ×3", (cc == 3).all(1))]}
        about = (
            f"Top three rows: the same questions at every filler length, grouped by whether the model's answer is right with 0 and with 100 dots ({(c0 & ~c100).sum().item()} questions right with 0 but wrong with 100 are not shown)."
            "<br>Bottom three rows: the same questions grouped by their two multipliers, c1 in y = c1·x ± a and c2 in answer = c2·y ± b ('twice' = ×2, 'three times' = ×3). The example is ×3 then ×2."
        )
        jobs.append((f"figs/lens_k{k}", k, r, tgt, {**groups, **coef_rows}, about, example))
    for si, (lo, hi) in enumerate(coef_sets) if coefswap else []:
        for k in sorted(coef_res):
            r = coef_res[k]
            meta = r["meta"]
            ins = meta[:, 0] == si
            swap_tgt = t.tensor([[with_coefs(items[i], c1, c2)["chain"][name] for name in TARGETS] for _, c1, c2, i in meta.tolist()])
            swap_rows = {f"<b>{name}</b><br>n={m.sum().item()}<br>accuracy {r['correct'][m].float().mean():.0%}" + (f"<br>(0 dots: {coef_res[0]['correct'][m].float().mean():.0%})" if k else ""): m for name, m in [(f"Both multipliers {hi}", ins & (meta[:, 1] == hi) & (meta[:, 2] == hi)), (f"Both multipliers {lo}", ins & (meta[:, 1] == lo) & (meta[:, 2] == lo)), (f"One {lo}, one {hi}<br>(both orders)", ins & (meta[:, 1] != meta[:, 2]))]}
            b = with_coefs(items[meta[ins][0, 3].item()], hi, lo)
            swap_example = f"Example (first multiplier {hi}, second {lo}): <i>{b['x_name']} = {b['chain']['x']} · {b['queried_term']} = {dict(b['definitions'])[b['queried_term']]} · … · {b['question']}</i>  →  y = {b['chain']['y']}, c2·y = {b['chain']['c2y']}, answer = {b['chain']['answer']}"
            about = f"Rows: the same {ins.sum().item() // 4} questions in every row, rewritten so only the two multiplier words change ('twice', 'three times', 'four times', 'five times'). The 10 worked examples in the prompt keep their 2s and 3s."
            jobs.append((f"figs/lens_coefswap{lo}{hi}_k{k}", k, r, swap_tgt, swap_rows, about, swap_example))
    for path, k, r, jt, rows, about, ex_text in jobs:
        n_layers = r["rank"].shape[1]
        scores = [s for s in score_text if s != "resid" or "num_p_mean" in r]
        heats, titles = {}, {}
        for score in scores:
            v = scored(r, jt, score)
            heats[score] = t.stack([v[m][:, min_layer:, N_Q - n_show_q:, TARGETS.index(name)].mean(0) for m in rows.values() for name, _ in cols])
            low = max(v[m][:, :min_layer, N_Q - n_show_q:, TARGETS.index(name)].mean(0).max().item() for m in rows.values() for name, _ in cols)
            titles[score] = (
                f"<b>Which number is the model working with at each layer and token?</b>  DeepSeek V4 Flash, {k} filler dots ('Filler: . . .') between question and answer"
                f"<br><span style='font-size:13px'>{ex_text}</span>"
                f"<br><span style='font-size:13px'>{score_text[score][2]}</span>"
                f"<br><span style='font-size:13px'>{about}</span><br><span style='font-size:13px'>Layers 0–{min_layer - 1} are not shown (at most {low:.2f} on this scale there). The model has 43 layers (0–42); its answer is read at the last token. The dropdown at the top right changes the scoring.</span>"
            )
        heat = heats[scores[0]]
        dot_step = 1 if k <= 15 else 5 if k <= 25 else 10
        ticks = [(j, f"'{s}'") for j, s in enumerate(["number", "for", "y", "aj", "plus", " ", "36", "?"])] + [(n_show_q, "'F'"), (n_show_q + 1, "'iller'"), (n_show_q + 2, "':'")]
        ticks += [(n_show_q + 2 + d, f"dot {d}") for d in range(1, k + 1) if d == 1 or d % dot_step == 0]
        ticks += [(n_show_q + 3 + k, "'Answer'"), (n_show_q + 4 + k, "':'"), (n_show_q + 5 + k, "‹assistant›"), (n_show_q + 6 + k, "‹answer here›")]
        assert ticks[-1][0] == heat.shape[-1] - 1
        fig = imshow(
            heat,
            facet_col_wrap=4,
            aspect="auto",
            facet_labels=[title for _ in rows for _, title in cols],
            origin="lower",
            color_continuous_scale="Viridis",
            color_continuous_midpoint=None,
            zmin=0,
            zmax=1,
            labels={"x": "token position", "y": "layer", "color": score_text[scores[0]][1]},
            title=titles[scores[0]],
            height=300 * len(rows) + 400,
            width=1900,
            margin={"t": 305, "l": 210, "r": 40, "b": 110},
            return_fig=True,
        )
        score_dropdown(fig, heats, titles)
        fig.update_annotations(font_size=15)
        fig.update_xaxes(tickvals=[j for j, _ in ticks], ticktext=[s for _, s in ticks], tickangle=-60, tickfont={"size": 10}, showticklabels=True)
        fig.update_yaxes(tickvals=list(range(0, n_layers - min_layer, 4)), ticktext=[str(min_layer + j) for j in range(0, n_layers - min_layer, 4)], showticklabels=True)
        for xb in sorted({n_show_q - 0.5, n_show_q + 2.5, n_show_q + 2.5 + k}):
            fig.add_vline(x=xb, line_width=1, line_color="white", opacity=0.6, row="all", col="all")
        row_y = space_rows(fig, 100)
        for (y0, y1), label in zip(row_y, rows):
            fig.add_annotation(text=label, xref="paper", yref="paper", x=-0.035, y=(y0 + y1) / 2, xanchor="right", showarrow=False, align="right", font={"size": 13})
        fig.add_annotation(text="Tick labels on the question tokens are from the example question; white lines mark the end of the question, the 'Filler:' label, and the end of the dots.", xref="paper", yref="paper", x=0, y=-0.085, xanchor="left", showarrow=False, font={"size": 11})
        write_dark_html(fig.update_layout(**DARK), path + ".html")
    print(f"{green}wrote {len(jobs)} heatmaps to figs/{endc}")

#%% Answer position, for readers without context: by layer, the share of questions whose model state at the token where the answer is produced points most to the correct answer, with 0 and with 100 filler dots, for the same three question groups as the heatmaps

plot_answer_position = True
if plot_answer_position:
    res = t.load(f"results/lens_{TAG}.pt")
    c0, c100 = res[0]["correct"], res[100]["correct"]
    n_layers = res[0]["rank"].shape[1]
    groups = {"always right": c0 & c100, "flipped by filler": ~c0 & c100, "always wrong": ~c0 & ~c100}
    fig = go.Figure()
    for (g, m), color in zip(groups.items(), SERIES):
        for k, dash in [(0, "dot"), (100, "solid")]:
            share = (res[k]["rank"][m][:, :, -1, TARGETS.index("answer")] == 0).float().mean(0)
            fig.add_trace(go.Scatter(x=list(range(n_layers)), y=share.tolist(), mode="lines", line={"color": color, "dash": dash, "width": 3 if k else 2}, name=f"{g} (n={m.sum().item()}), {k} dots"))
    first = {g: next((l for l in range(n_layers) if (res[100]["rank"][m][:, l, -1, TARGETS.index("answer")] == 0).float().mean() >= 0.5), None) for g, m in groups.items()}
    fig.update_layout(
        title=(
            "<b>When does the model settle on the correct answer, layer by layer?</b>  DeepSeek V4 Flash, at the token where the answer is produced"
            "<br><span style='font-size:13px'>Share of questions where the model's internal state, read through its own output layer (the 'logit lens'), ranks the correct answer first among the numbers 0–999.</span>"
            "<br><span style='font-size:13px'>Dotted: no filler. Solid: 100 filler dots ('Filler: . . .') between the question and 'Answer:'. Groups: whether the final answer is right with 0 and with 100 dots (always right, flipped by filler, always wrong).</span>"
            f"<br><span style='font-size:13px'>With 100 dots, half the questions have the correct answer on top by layer {first['always right']} (always right) and layer {first['flipped by filler']} (flipped by filler).</span>"
            "<br><span style='font-size:13px'>Without filler, the flipped group peaks near 20% and never gets there.</span>"
        ),
        xaxis_title="layer (0 = first, 42 = last; the answer is read out after layer 42)",
        yaxis_title="share of questions with the correct answer on top",
        yaxis_range=[0, 1.02],
        xaxis_range=[18, 42.5],
        height=700,
        width=1400,
        margin={"t": 190},
        legend={"title": "question group, filler"},
        **DARK,
    )
    write_dark_html(fig, "figs/lens_answer_position.html")
    print(f"{green}wrote figs/lens_answer_position.html; first layer with >= 50% at 100 dots: {first}{endc}")

#%% Decode strength by depth, for readers without context: at each layer, how high each chain value (x, c1·x, y, c2·y, answer) ranks among the numbers 0–999 in the model's state, averaged over the tokens after the question ('Filler:', the dots when included, 'Answer:' and the token where the answer is read) and over questions.
# Columns 0, 10 and 100 dots; rows all questions and the questions that 100 dots flip from wrong to right. Two figures: the dots left out of the average (figs/lens_rank_depth.html) and included (figs/lens_rank_depth_dots.html).

plot_rank_depth = True
if plot_rank_depth:
    ks = [0, 10, 100]
    labels = {"x": "x: the given number", "c1x": "c1·x: the first product", "y": "y: the hidden first result", "c2y": "c2·y: the second product", "answer": "the final answer"}
    colors = SERIES + ["#a66bd6"]
    res = t.load(f"results/lens_{TAG}.pt") | t.load(f"results/lens_{TAG}_small.pt")
    c0, c100 = res[0]["correct"], res[100]["correct"]
    groups = {"all questions": t.ones_like(c0), "flipped by 100 dots": ~c0 & c100}
    n_layers = res[0]["rank"].shape[1]
    for include_dots in [False, True]:
        fig = make_subplots(
            rows=2,
            cols=3,
            shared_yaxes=True,
            subplot_titles=[f"{g} (n={m.sum().item()}), {k} dots" for g, m in groups.items() for k in ks],
            horizontal_spacing=0.04,
            vertical_spacing=0.13,
        )
        for col, k in enumerate(ks, 1):
            n_pos = res[k]["rank"].shape[2]  # the last k+31 tail positions: 24 question tokens, 'F' 'iller' ':', the k dots, 'Answer' ':' '<｜Assistant｜>', '</think>'
            span = list(range(n_pos - 7 - k, n_pos)) if include_dots else list(range(n_pos - 7 - k, n_pos - 4 - k)) + list(range(n_pos - 4, n_pos))
            place = t.log10(1 + res[k]["rank"][:, :, span].float())  # [question, layer, token, value]: log10 of the value's place among the numbers (1 = ranked first)
            for row, (g, m) in enumerate(groups.items(), 1):
                mean_place = 10 ** place[m].mean((0, 2))  # [layer, value]: geometric mean over questions and tokens
                for j, name in enumerate(TARGETS):
                    fig.add_trace(go.Scatter(x=list(range(n_layers)), y=mean_place[:, j].tolist(), mode="lines", line={"color": colors[j], "width": 2.5}, name=labels[name], legendgroup=name, showlegend=(row == 1 and col == 1)), row=row, col=col)
                if g == "all questions":
                    best = mean_place.min(0)
                    print(f"{cyan}{'dots included' if include_dots else 'dots left out'}, {k} dots, all questions, best geometric-mean place per value: " + ", ".join(f"{name} {best.values[j]:.1f} at layer {best.indices[j]}" for j, name in enumerate(TARGETS)) + endc)
        fig.update_yaxes(type="log", range=[3, -0.03], tickvals=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
        fig.update_yaxes(title_text="geometric mean place among 0–999 (1 = first)", col=1)
        fig.update_xaxes(title_text="layer (0 = first, 42 = last)", row=2)
        fig.update_layout(
            title=(
                "<b>How strongly does the model's state point to each step of the chain, layer by layer?</b>  DeepSeek V4 Flash, logit lens over the tokens after the question"
                "<br><span style='font-size:13px'>Each question gives a number x and asks for c2·y ± b, where y = c1·x ± a is never written. At every layer, the model's state at each token is read through its own output layer (the 'logit lens')</span>"
                "<br><span style='font-size:13px'>and each chain value is ranked among the numbers 0–999. Lines: the geometric mean of a value's place (1 = ranked first, about 370 = chance) over questions and over the tokens 'Filler:', 'Answer:',</span>"
                "<br><span style='font-size:13px'>'<｜Assistant｜>' and '</think>' (where the answer is read)"
                + (", and the filler dots." if include_dots else "; the filler dots are left out of the average.")
                + " Higher on the plot = decoded more strongly.</span>"
                "<br><span style='font-size:13px'>Columns: 0, 10 and 100 filler dots ('Filler: . . .') between the question and 'Answer:'. Rows: all questions, and the questions whose final answer is wrong with 0 dots and right with 100 ('flipped by 100 dots').</span>"
            ),
            height=960,
            width=1500,
            margin={"t": 260},
            legend={"title": "chain value", "orientation": "h", "y": -0.08},
            **DARK,
        )
        path = "figs/lens_rank_depth" + ("_dots" if include_dots else "") + ".html"
        write_dark_html(fig, path)
        print(f"{green}wrote {path}{endc}")

#%% Filler uplift by multiplier, for readers without context: mean probability of the correct answer (or greedy accuracy) at 0, 10 and 100 filler dots and the change from 0 dots, on the coefficient-swapped questions (the same base questions in every kind), with 95% bootstrap intervals over questions

plot_uplift_coef = True
if plot_uplift_coef:
    ks = [0, 10, 100]
    # ks = [0, 5, 10]
    metric = "p_correct"
    # metric = "greedy"
    n_boot = 5000
    what, axis_name, definition = {
        "p_correct": ("the chance of a correct answer", "Probability of the correct answer", "Probability of the correct answer: the model's probability of the right number (its expected accuracy when sampling)."),
        "greedy": ("the share of questions answered correctly", "Share answered correctly (greedy)", "Answered correctly: the model's single most likely answer is the right number (greedy decoding, pass@1)."),
    }[metric]
    coef_res = {**t.load(f"results/lens_coef_{TAG}.pt")["res"], **t.load(f"results/lens_coef_{TAG}_k100.pt")["res"]}
    meta = coef_res[0]["meta"]
    assert all(t.equal(coef_res[k]["meta"], meta) for k in ks)
    base = sorted(set(meta[:, 3].tolist()))
    kinds = {"×2 then ×2": (0, [(2, 2)]), "×2 and ×3<br>(either order)": (0, [(2, 3), (3, 2)]), "×3 then ×3": (0, [(3, 3)]), "×2 and ×4<br>(either order)": (1, [(2, 4), (4, 2)]), "×4 then ×4": (1, [(4, 4)])}
    p = t.zeros(len(ks), len(base), len(kinds) + 1)  # per base question: the metric averaged over the kind's orders; last column averages the kinds
    for i, k in enumerate(ks):
        pc = coef_res[k]["model_logp"].float().exp() if metric == "p_correct" else coef_res[k]["correct"].float()
        for j, (si, combos) in enumerate(kinds.values()):
            for c1, c2 in combos:
                sel = (meta[:, 0] == si) & (meta[:, 1] == c1) & (meta[:, 2] == c2)
                assert meta[sel, 3].tolist() == base
                p[i, :, j] += pc[sel] / len(combos)
        p[i, :, -1] = p[i, :, :-1].mean(-1)
    names = list(kinds) + ["<b>average of<br>the five</b>"]
    boot = p[:, t.randint(0, len(base), (n_boot, len(base)), generator=t.Generator().manual_seed(0))].mean(2)  # [k, boot, kind]
    m = p.mean(1)
    order = m[-1, :-1].argsort(descending=True).tolist() + [len(names) - 1]
    up = boot[1:] - boot[:1]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], subplot_titles=[axis_name, "Change from no filler (percentage points)"])
    for i, (k, color) in enumerate(zip(ks, ["#8c8c8c", SERIES[0], SERIES[3]])):
        lo, hi = boot[i].quantile(0.025, 0), boot[i].quantile(0.975, 0)
        fig.add_trace(go.Bar(x=[names[j] for j in order], y=m[i, order].tolist(), error_y={"type": "data", "symmetric": False, "array": (hi - m[i])[order].tolist(), "arrayminus": (m[i] - lo)[order].tolist()}, marker_color=color, name=f"{k} filler dots", legendgroup=str(k)), row=1, col=1)
        if k:
            d = m[i] - m[0]
            ulo, uhi = up[i - 1].quantile(0.025, 0), up[i - 1].quantile(0.975, 0)
            fig.add_trace(go.Bar(x=[names[j] for j in order], y=(100 * d[order]).tolist(), error_y={"type": "data", "symmetric": False, "array": (100 * (uhi - d))[order].tolist(), "arrayminus": (100 * (d - ulo))[order].tolist()}, marker_color=color, name=f"{k} filler dots", text=[f"{100 * v:+.0f}" for v in d[order].tolist()], textposition="inside", insidetextanchor="end", legendgroup=str(k), showlegend=False), row=1, col=2)
    d_hi, d_mid = 100 * (m[2] - m[0]), 100 * (m[1] - m[0])
    b = with_coefs(items[base[0]], 3, 2)
    fig.update_layout(
        title=(
            f"<b>{ks[2]} filler dots change {what} by {d_hi[:-1].min():+.0f} to {d_hi[:-1].max():+.0f} percentage points across multiplier pairs; {ks[1]} dots by {d_mid[:-1].min():+.0f} to {d_mid[:-1].max():+.0f}</b>  (DeepSeek V4 Flash)"
            f"<br><span style='font-size:13px'>Task: two-step arithmetic written in words. Example (×3 then ×2): <i>{b['x_name']} = {b['chain']['x']} · {b['queried_term']} = {dict(b['definitions'])[b['queried_term']]} · (3 more definitions) · {b['question']}</i>  →  {b['queried_term']} = {b['chain']['y']}, answer = {b['chain']['answer']}</span>"
            f"<br><span style='font-size:13px'>Labels below name the two multipliers. The same {len(base)} questions in every group, with only the two multiplier words changed; 'either order' groups average both orders.</span>"
            f"<br><span style='font-size:13px'>Filler: 'Filler: . . .' between the question and 'Answer:'; the prompt's 10 worked examples carry the same filler. {definition}</span>"
            f"<br><span style='font-size:13px'>Groups sorted by the {ks[2]}-dot value. Error bars: 95% intervals from resampling questions.</span>"
        ),
        barmode="group",
        height=760,
        width=1700,
        margin={"t": 250},
        legend={"title": "filler length"},
        **DARK,
    )
    fig.update_yaxes(range=[0, 1.1], row=1, col=1)
    path = f"figs/uplift_by_coef{'' if metric == 'p_correct' else '_' + metric}{'' if ks == [0, 10, 100] else '_k' + '_'.join(map(str, ks))}.html"
    write_dark_html(fig, path)
    print(f"{green}wrote {path}{endc}")
    for j in order:
        print(f"{names[j].replace('<br>', ' '):32s} {metric} " + "  ".join(f"{k} dots {m[i, j]:.3f}" for i, k in enumerate(ks)) + f" | change {ks[1]} dots {d_mid[j]:+.1f}, {ks[2]} dots {d_hi[j]:+.1f}")

#%% 2-fact heatmaps for readers without context: at 0 and 10 dots, how strongly the state at each (layer, token) points to fact 1, fact 2 or their sum (a dropdown switches the scoring), rows by whether the answer is right with 0 and with 10 dots; writes figs/lens2_k{0,10}.html

plot_lens_2fact = True
if plot_lens_2fact:
    min_layer = 20
    res = t.load(f"results/lens2_{TAG}.pt")
    assert all(t.equal(r["idx"], res[0]["idx"]) for r in res.values())
    ex = items2[res[0]["idx"][0].item()]
    c0, c10 = res[0]["correct"], res[10]["correct"]
    rows = {
        f"<b>Always right</b><br>right with 0 dots<br>and with 10<br>n={(c0 & c10).sum().item()}": c0 & c10,
        f"<b>Flipped by filler</b><br>wrong with 0 dots,<br>right with 10<br>n={(~c0 & c10).sum().item()}": ~c0 & c10,
        f"<b>Always wrong</b><br>wrong with 0 dots<br>and with 10<br>n={(~c0 & ~c10).sum().item()}": ~c0 & ~c10,
    }
    cols = [("a1", f"fact 1 ({ex['chain']['a1']} in the example)"), ("a2", f"fact 2 ({ex['chain']['a2']})"), ("answer", f"their sum, the answer ({ex['chain']['answer']})")]
    jt = t.tensor([[items2[i]["chain"][name] for name in TARGETS2] for i in res[0]["idx"].tolist()])
    for k in sorted(res):
        r, n_q = res[k], res[k]["n_q"]
        n_layers = r["rank"].shape[1]
        heats, titles = {}, {}
        for score in score_text:
            v = scored(r, jt, score)
            heats[score] = t.stack([v[m][:, min_layer:, :, TARGETS2.index(name)].mean(0) for m in rows.values() for name, _ in cols])
            low = max(v[m][:, :min_layer, :, TARGETS2.index(name)].mean(0).max().item() for m in rows.values() for name, _ in cols)
            titles[score] = (
                f"<b>Which number is the model working with at each layer and token?</b>  DeepSeek V4 Flash, 2-fact addition, {k} filler dots ('Filler: . . .') between question and answer"
                f"<br><span style='font-size:13px'>Example question: <i>{ex['question']}</i>  →  fact 1 = {ex['chain']['a1']}, fact 2 = {ex['chain']['a2']}, answer = {ex['chain']['answer']}. Both facts must be recalled from memory (nothing is written in the prompt) and added.</span>"
                f"<br><span style='font-size:13px'>{score_text[score][2]}</span>"
                f"<br><span style='font-size:13px'>Rows: the same questions at both filler lengths, grouped by whether the model's answer is right with 0 and with 10 dots ({(c0 & ~c10).sum().item()} questions right with 0 but wrong with 10 are not shown). 5 worked examples precede the question.</span>"
                f"<br><span style='font-size:13px'>Layers 0–{min_layer - 1} are not shown (at most {low:.2f} on this scale there). The model has {n_layers} layers; its answer is read at the last token. The dropdown at the top right changes the scoring.</span>"
            )
        heat = heats["top1"]
        hits = (r["rank"] == 0).float()[c0 & c10][:, :, n_q + 3:n_q + 3 + k, :].mean(0)  # [layer, dot, value] on always-right items, top-1
        if k:
            print(f"{cyan}k={k}, always right: hit-weighted mean layer " + ", ".join(f"{name} {(hits[..., j].sum(1) * t.arange(n_layers)).sum() / hits[..., j].sum():.1f}" for j, (name, _) in enumerate(cols)) + "; mean dot " + ", ".join(f"{name} {(hits[..., j].sum(0) * t.arange(1, k + 1)).sum() / hits[..., j].sum():.1f}" for j, (name, _) in enumerate(cols)) + endc)
        tail = prompt_ids(tok, shots2, ex, k)[1]
        label0 = regions(tok, tail, k)["label"][0]
        ticks = [(j, f"'{tok.decode([tid])}'") for j, tid in enumerate(tail[label0 - n_q:label0])] + [(n_q, "'F'"), (n_q + 1, "'iller'"), (n_q + 2, "':'")]
        ticks += [(n_q + 2 + d, f"dot {d}") for d in range(1, k + 1)]
        ticks += [(n_q + 3 + k, "'Answer'"), (n_q + 4 + k, "':'"), (n_q + 5 + k, "‹assistant›"), (n_q + 6 + k, "‹answer here›")]
        assert ticks[-1][0] == heat.shape[-1] - 1
        fig = imshow(
            heat,
            facet_col_wrap=3,
            aspect="auto",
            facet_labels=[title for _ in rows for _, title in cols],
            origin="lower",
            color_continuous_scale="Viridis",
            color_continuous_midpoint=None,
            zmin=0,
            zmax=1,
            labels={"x": "token position", "y": "layer", "color": score_text["top1"][1]},
            title=titles["top1"],
            height=300 * len(rows) + 400,
            width=1500,
            margin={"t": 305, "l": 210, "r": 40, "b": 110},
            return_fig=True,
        )
        score_dropdown(fig, heats, titles)
        fig.update_annotations(font_size=15)
        fig.update_xaxes(tickvals=[j for j, _ in ticks], ticktext=[s for _, s in ticks], tickangle=-60, tickfont={"size": 10}, showticklabels=True)
        fig.update_yaxes(tickvals=list(range(0, n_layers - min_layer, 4)), ticktext=[str(min_layer + j) for j in range(0, n_layers - min_layer, 4)], showticklabels=True)
        for xb in sorted({n_q - 0.5, n_q + 2.5, n_q + 2.5 + k}):
            fig.add_vline(x=xb, line_width=1, line_color="white", opacity=0.6, row="all", col="all")
        row_y = space_rows(fig, 100)
        for (y0, y1), label in zip(row_y, rows):
            fig.add_annotation(text=label, xref="paper", yref="paper", x=-0.035, y=(y0 + y1) / 2, xanchor="right", showarrow=False, align="right", font={"size": 13})
        fig.add_annotation(text="Tick labels on the question tokens are from the example question; white lines mark the end of the question, the 'Filler:' label, and the end of the dots.", xref="paper", yref="paper", x=0, y=-0.085, xanchor="left", showarrow=False, font={"size": 11})
        write_dark_html(fig.update_layout(**DARK), f"figs/lens2_k{k}.html")
    print(f"{green}wrote figs/lens2_k{{0,10}}.html{endc}")
