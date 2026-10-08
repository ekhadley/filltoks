#%% Setup: where does the bf16 batch-composition noise come from? Run-to-run nondeterminism, MoE routing flips, or continuous floating-point propagation

import json

import torch as t
from transformers import FineGrainedFP8Config

from mechtools import *

from utils import MODEL_ID, tiny_bridge, fp32_routers, prompt_ids, prefill, tail_logits

t.set_grad_enabled(False)
items = [json.loads(l) for l in open("data/eval.jsonl")]
shots = [json.loads(l) for l in open("data/fewshot.jsonl")]

TAG = "tiny"
model = tiny_bridge()
# TAG = "v4flash"
# model = load_bridge(MODEL_ID, quantization_config=FineGrainedFP8Config(dequantize=True))
tok = model.tokenizer
hf = model.original_model
L = model.cfg.n_layers
gates = {l: layer.mlp.gate.original_component for l, layer in enumerate(hf.model.layers)}
topk_layers = [l for l, g in gates.items() if not g.__class__.__name__.endswith("HashRouter")]
g = gates[topk_layers[0]]
print(f"{cyan}experts implementation {hf.config._experts_implementation}, router weight {g.weight.dtype}, selection bias {g.e_score_correction_bias.dtype}, top-k layers {topk_layers[0]}..{topk_layers[-1]}{endc}")

#%% Router hooks (record each row-0 token's selected experts and the margin between its k-th and (k+1)-th selection scores, or force row 0's experts) and one run of a batch of tails

route = {"record": None, "force": {}, "n": 0}
def router_hook(mod, args, out):
    logits, w, idx = out
    l, n = mod.layer_idx_, route["n"]
    if route["record"] is not None:
        top = (mod.score_fn(logits) + mod.e_score_correction_bias)[:n].float().topk(mod.top_k + 1, dim=-1).values
        route["record"][l] = (idx[:n].cpu(), (top[:, -2] - top[:, -1]).cpu())
    if l in route["force"]:
        idx = idx.clone()
        idx[:n] = route["force"][l].to(idx.device)
        w = mod.score_fn(logits).gather(1, idx)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20) * mod.routed_scaling_factor
        return logits, w, idx
for l in topk_layers:
    gates[l].layer_idx_ = l
    gates[l].register_forward_hook(router_hook)

SUB = ["blocks.0.attn.hook_out", "blocks.0.mlp.hook_out", "blocks.1.attn.hook_out", "blocks.1.mlp.hook_out"]
def run(cache, rows: list[list[int]], record=False, force=None) -> dict:
    """Row 0's fp32 answer-position log-probs, decoder-layer inputs and layer 0/1 sublayer outputs at its n tail positions, and (record) its routing per top-k layer."""
    n = len(rows[0])
    acts = {}
    def grab(act, hook):
        acts[hook.name] = (act[0] if isinstance(act, tuple) else act)[0, :n].float().cpu()
    route.update(record={} if record else None, force=force or {}, n=n)
    logp = tail_logits(model, cache, rows, [(f"blocks.{l}.hook_in", grab) for l in range(L)] + [(name, grab) for name in SUB])[0, -1].log_softmax(-1).cpu()
    out = {"logp": logp, "acts": acts, "routes": route["record"]}
    route.update(record=None, force={})
    return out

def rel(a: Tensor, b: Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()

def compare(x: dict, ref: dict) -> dict:
    """Divergence of run x from run ref: answer-position KL(ref || x), top-1 log-prob shift, and relative residual difference per layer over all tail positions and at the last."""
    p, q = ref["logp"], x["logp"]
    return {
        "kl": (p.exp() * (p - q)).sum().item(),
        "dtop1": (q[p.argmax()] - p[p.argmax()]).item(),
        "argmax_agree": (p.argmax() == q.argmax()).item(),
        "all_pos": [rel(x["acts"][f"blocks.{l}.hook_in"], ref["acts"][f"blocks.{l}.hook_in"]) for l in range(L)],
        "last_pos": [rel(x["acts"][f"blocks.{l}.hook_in"][-1], ref["acts"][f"blocks.{l}.hook_in"][-1]) for l in range(L)],
        "sub": {name: rel(x["acts"][name], ref["acts"][name]) for name in SUB},
        "bitwise": all(t.equal(x["acts"][k], ref["acts"][k]) for k in ref["acts"]) and t.equal(x["logp"], ref["logp"]),
    }

#%% Per item at k=100 on a shared prefix cache: A = the tail alone; A2 = A again (determinism); A_self = A with its own routing forced (gate: forcing is a no-op); B = the tail batched with another item's tail (natural batch-composition noise); B_forced = B with row 0's routing forced to A's (no flips, continuous propagation only). Variants: HF as loaded (bf16 router logits and routing weights), and fp32 routing as in DeepSeek's reference (fp32 router logits and routing weights, so the 6 expert outputs are also weighted and summed in fp32)

run_noise = True
if run_noise:
    k = 100
    n_items = 16
    variants = ["hf", "fp32_routing"]
    pts = [prompt_ids(tok, shots, it, k) for it in items[:n_items + 1]]
    prefix = pts[0][0]
    results, clean = {}, {}
    for variant in variants:
        route.update(record=None, force={})
        fp32_routers(model, on=variant == "fp32_routing")
        cache = prefill(model, prefix)
        results[variant] = []
        for i in range(n_items):
            tail, other = pts[i][1], pts[i + 1][1]
            A = run(cache, [tail], record=True)
            A2 = run(cache, [tail])
            A_self = run(cache, [tail], force={l: r[0] for l, r in A["routes"].items()})
            B = run(cache, [tail, other], record=True)
            B_forced = run(cache, [tail, other], force={l: r[0] for l, r in A["routes"].items()})
            clean[(variant, i)] = A["logp"]
            flips = {l: (B["routes"][l][0].sort(-1).values != A["routes"][l][0].sort(-1).values).any(-1).sum().item() for l in topk_layers}
            margins = t.stack([A["routes"][l][1] for l in topk_layers])
            r = {
                "item": i,
                "n_tail": len(tail),
                "repeat": compare(A2, A),
                "self_force": compare(A_self, A),
                "batch": compare(B, A),
                "batch_forced": compare(B_forced, A),
                "flips_per_layer": [flips[l] for l in topk_layers],
                "first_flip_layer": next((l for l in topk_layers if flips[l]), None),
                "margin_frac": {thr: (margins <= thr).float().mean().item() for thr in [0.0, 1e-3, 1e-2, 3e-2]},
            }
            results[variant].append(r)
            b, bf = r["batch"], r["batch_forced"]
            print(f"{cyan}{variant} item {i}: repeat bitwise {r['repeat']['bitwise']}, self-force bitwise {r['self_force']['bitwise']}; batch KL {b['kl']:.4f} dtop1 {b['dtop1']:+.3f}, last-pos rel diff at the last layer {b['last_pos'][-1]:.3f}, {sum(r['flips_per_layer'])} flips over {len(topk_layers)} layers (first at layer {r['first_flip_layer']}); routing forced: KL {bf['kl']:.4f} dtop1 {bf['dtop1']:+.3f}, rel diff {bf['last_pos'][-1]:.3f}{endc}")
            print(f"{gray}  layer 0/1 sublayer rel diff (batch): " + ", ".join(f"{name.split('.', 1)[1]} {v:.4f}" for name, v in b["sub"].items()) + f"; margins <= 0 / 1e-3 / 1e-2 / 3e-2: " + " / ".join(f"{v:.3f}" for v in r["margin_frac"].values()) + endc)
    for i in range(n_items):
        p, q = clean[("hf", i)], clean[("fp32_routing", i)]
        print(f"{purple}item {i}: hf vs fp32-routing clean runs: KL {(p.exp() * (p - q)).sum().item():.4f}, top-1 {tok.decode([p.argmax()])!r} vs {tok.decode([q.argmax()])!r}{endc}")
    json.dump({v: [{**r, "margin_frac": {str(k): x for k, x in r["margin_frac"].items()}} for r in rs] for v, rs in results.items()}, open(f"results/noise_{TAG}.json", "w"))
    print(f"{green}saved results/noise_{TAG}.json{endc}")

#%% Per-layer residual divergence from batch composition, with natural routing vs routing forced to the batch-1 run's, and routing flips per layer

plot_noise = True
if plot_noise:
    res = json.load(open(f"results/noise_{TAG}.json"))
    for variant, rs in res.items():
        lines = {f"{kind}, {pos}": t.tensor([r[kind][pos] for r in rs]).mean(0) for kind in ["batch", "batch_forced"] for pos in ["all_pos", "last_pos"]}
        line(list(lines.values()), names=list(lines.keys()), labels={"x": "layer (decoder-layer input)", "y": "relative difference, mean over items"}, title=f"Batch-composition divergence, {variant}", log_y=True)
        flips = t.tensor([r["flips_per_layer"] for r in rs], dtype=t.float).mean(0) / t.tensor([r["n_tail"] for r in rs], dtype=t.float).mean()
        n_layers, n_topk = len(rs[0]["batch"]["all_pos"]), len(rs[0]["flips_per_layer"])
        line(flips, x=list(range(n_layers - n_topk, n_layers)), labels={"x": "layer", "y": "fraction of tail tokens with a different expert set"}, title=f"Routing flips from batch composition, {variant}")
