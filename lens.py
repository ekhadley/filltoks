#%% Setup: Brauer et al.'s logit-lens decode of the chain x -> c1x -> y -> c2y -> answer at every (layer, position) from the question end through the filler to the answer position

import json

import torch as t
from transformers import FineGrainedFP8Config

from mechtools import *

from utils import MODEL_ID, tiny_bridge, fp32_routers, prompt_ids, regions, answer_ids, prefill, tail_logits

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

def run_lens(cache, tails: list[list[int]], chains: list[dict], n_keep: int) -> dict:
    """One forward of right-padded tails. At every decoder layer's output and each row's last n_keep positions: the number lens's top-1 value, and each chain value's rank and log-prob among the numbers 0..999. At the answer position: the full-vocab lens's log-prob and rank of the answer token per layer. Also the model's own answer-position log-prob of the answer and whether it is the argmax."""
    B, lens_ = len(tails), [len(tl) for tl in tails]
    tgt = t.tensor([[ch[name] for name in TARGETS] for ch in chains], device=W.device)
    ans = num_ids[tgt[:, -1]]
    out = {key: [] for key in ["top1", "rank", "logp", "ans_logp", "ans_rank"]}
    def hook(act, hook):
        assert act.ndim == 4 and act.shape[0] == B, act.shape
        pos = t.tensor([[n - n_keep + j for j in range(n_keep)] for n in lens_], device=act.device)
        h = collapse(act[t.arange(B, device=act.device)[:, None], pos])
        logp = (h @ W_num.T).log_softmax(-1)
        tl = logp.gather(-1, tgt[:, None, :].expand(-1, n_keep, -1))
        out["top1"].append(logp.argmax(-1).short().cpu())
        out["rank"].append((logp[:, :, None, :] > tl[..., None]).sum(-1).short().cpu())
        out["logp"].append(tl.half().cpu())
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
        res[k]["idx"] = t.tensor([it["idx"] for it in use])
        print(f"{cyan}k={k}: {len(use)} items, greedy correct {res[k]['correct'].float().mean():.3f}, mean P(answer) {res[k]['model_logp'].exp().mean():.3f}, max last-layer lens gap {res[k]['gap'].max():.4f} nats{endc}")
    t.save(res, f"results/lens_{TAG}.pt")
    print(f"{green}saved results/lens_{TAG}.pt{endc}")

#%% Heatmaps as in Brauer et al.: fraction of items whose top number token is exactly each chain value, per (layer, position), split by whether the model answers correctly at that filler length

plot_heatmaps = True
if plot_heatmaps:
    res = t.load(f"results/lens_{TAG}.pt")
    for k, r in res.items():
        hit = (r["rank"] == 0).float()
        c = r["correct"]
        heat = t.cat([hit[c].mean(0).permute(2, 0, 1), hit[~c].mean(0).permute(2, 0, 1)])
        names = position_names(k)
        for j, name in enumerate(TARGETS):
            for grp, sel in [("correct", c), ("wrong", ~c)]:
                h = hit[sel][..., j].mean(0)
                l, p = divmod(h.argmax().item(), h.shape[1])
                hf_ = h[:, N_Q + 3:N_Q + 3 + k]
                lf, pf = divmod(hf_.argmax().item(), max(k, 1)) if k else (None, None)
                print(f"{cyan}k={k} {grp} (n={sel.sum()}): {name} best {h.max():.3f} at layer {l} {names[p]}" + (f"; best in filler {hf_.max():.3f} at layer {lf} {names[N_Q + 3 + pf]}" if k else "") + endc)
        imshow(
            heat,
            x=names,
            facet_col_wrap=5,
            facet_labels=[f"{name}, {grp} (n={n})" for grp, n in [("correct", c.sum().item()), ("wrong", (~c).sum().item())] for name in TARGETS],
            origin="lower",
            color_continuous_scale="Viridis",
            color_continuous_midpoint=None,
            zmin=0,
            labels={"x": "position", "y": "layer (output)", "color": "frac top-1"},
            title=f"Logit lens over numbers 0..999, {k} dots: fraction of items whose top number is each chain value",
            height=900,
            width=1800,
        )

#%% Answer position by layer, same items at 0 and 100 dots, grouped by greedy correctness at the two lengths: when does each chain value become the lens's top number, and the full-vocab log P(answer)

plot_answer_pos = True
if plot_answer_pos:
    res = t.load(f"results/lens_{TAG}.pt")
    c0, c100 = res[0]["correct"], res[100]["correct"]
    groups = {"flip": ~c0 & c100, "right at both": c0 & c100, "wrong at both": ~c0 & ~c100, "unflip": c0 & ~c100}
    print(f"{cyan}groups: " + ", ".join(f"{g} {s.sum()}" for g, s in groups.items()) + endc)
    for j, name in enumerate(TARGETS):
        if name in ["y", "c2y", "answer"]:
            lines = {f"{g}, {k} dots": (res[k]["rank"][sel][:, :, -1, j] == 0).float().mean(0) for g, sel in groups.items() if g != "unflip" for k in [0, 100]}
            line(list(lines.values()), names=list(lines.keys()), labels={"x": "layer (output)", "y": f"frac top-1 number = {name}"}, title=f"Answer position: lens top number = {name}")
    lines = {f"{g}, {k} dots": res[k]["ans_logp"][sel].mean(0) for g, sel in groups.items() if g != "unflip" for k in [0, 100]}
    line(list(lines.values()), names=list(lines.keys()), labels={"x": "layer (output)", "y": "mean log P(answer token), full vocab"}, title="Answer position: full-vocab lens log P(answer)")
