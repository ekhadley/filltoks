"""Equivalence gates on a tiny random-weight V4 (fp32, CPU) with the real tokenizer, prompts and data. Run: HF_HUB_OFFLINE=1 python test_local.py"""
import json
import random

import torch as t

from mechtools import *

from utils import tiny_bridge, edit_item, donors, readout_numbers, valid_donors, prompt_ids, regions, answer_ids, prefill, tail_logits, decode, answer_logprob, transplant, EOS
from transformers.cache_utils import DynamicCache

t.set_grad_enabled(False)
items = [json.loads(l) for l in open("data/eval.jsonl")]
shots = [json.loads(l) for l in open("data/fewshot.jsonl")]
api = [json.loads(l) for l in open("data/phase2_full.jsonl")]
model = tiny_bridge()
tok = model.tokenizer
hf = model.original_model
rng = random.Random(0)

def full_logits(ids: list[int]) -> t.Tensor:
    """Plain no-cache forward, fp32 logits at every position."""
    return tail_logits(model, DynamicCache(config=hf.config), [ids], n_last=len(ids))[0]

print(f"{purple}=== items ==={endc}")
for it in items + shots:
    assert edit_item(it) == {k: v for k, v in it.items() if k not in ("distractors", "rivals")}, it["idx"]
for it in rng.sample(items, 50):
    d = edit_item(it, x=rng.randint(10, 99), k1=rng.randint(1, 50), k2=rng.randint(1, 50))
    c = d["chain"]
    assert c["answer"] == c["c2y"] + (1 if it["operation"] == "plus" else -1) * d["constant"] and d["values"][it["x_name"]] == c["x"]
print(f"{green}edit_item round-trips all {len(items) + len(shots)} items{endc}")

print(f"{purple}=== prompts ==={endc}")
for r in api:
    if r["cond"].startswith("dots"):
        k = int(r["cond"].split("_")[1])
        prefix, tail = prompt_ids(tok, shots, items[r["idx"]], k)
        assert len(prefix) + len(tail) == r["prompt_tokens"], (r["idx"], k, len(prefix) + len(tail), r["prompt_tokens"])
        regions(tok, tail, k)
print(f"{green}token counts match the API's prompt_tokens on all dots rows; regions parse{endc}")
for it in rng.sample(items, 50):
    while not valid_donors(it, dons := donors(it, rng.randint(10, 99), rng.randint(1, 50), rng.randint(1, 50))):
        pass
    tail = prompt_ids(tok, shots, it, 100)[1]
    for kind, n_diff in [("xk2", 2), ("x", 1), ("k1", 1), ("k2", 1)]:
        dtail = prompt_ids(tok, shots, dons[kind], 100)[1]
        assert len(dtail) == len(tail) and sum(a != b for a, b in zip(tail, dtail)) == n_diff, (it["idx"], kind)
    assert all(len(answer_ids(tok, n)) == 1 for n in readout_numbers(it, dons).values())
print(f"{green}donor tails keep length and differ only at the edited tokens{endc}")

print(f"{purple}=== prefix cache, padding, decode, scoring ==={endc}")
for k in [0, 100, 300]:
    its = rng.sample(items, 3)
    pts = [prompt_ids(tok, shots, it, k) for it in its]
    prefix = pts[0][0]
    tails = [tl for _, tl in pts]
    cache = prefill(model, prefix)
    n_last = 5
    batched = tail_logits(model, cache, tails, n_last=n_last)
    again = tail_logits(model, cache, tails, n_last=n_last)
    full = t.stack([full_logits(prefix + tl)[-n_last:] for tl in tails])
    diff = (batched - full).abs().max().item()
    print(f"{cyan}k={k} (tails {[len(tl) for tl in tails]}): cached right-padded batch vs full forward max |dlogit| {diff:.2e}, reuse {(batched - again).abs().max().item():.1e}{endc}")
    assert diff < 1e-4 and t.equal(batched, again)
    chunked = tail_logits(model, prefill(model, prefix, chunk=64), tails, n_last=n_last)
    print(f"{cyan}k={k}: prefix prefilled in 64-token chunks vs at once max |dlogit| {(chunked - batched).abs().max().item():.2e}{endc}")
    assert (chunked - batched).abs().max().item() < 1e-4
    gens, first = decode(model, cache, tails)
    for tl, g in zip(tails, gens):
        ids = t.tensor([prefix + tl], device=model.cfg.device)
        ref = hf.generate(ids, max_new_tokens=6, do_sample=False, eos_token_id=EOS, pad_token_id=EOS)[0, len(prefix + tl):].tolist()
        assert g == ref[:len(g)] and (EOS in g or len(g) == 6), (g, ref)
    answers = [answer_ids(tok, n) for n in [264, 1000, 7]]
    lp = answer_logprob(model, cache, tails, answers)
    for tl, a, l in zip(tails, answers, lp):
        ids = prefix + tl + a + [EOS]
        ref = full_logits(ids).log_softmax(-1)[len(prefix + tl) - 1:-1].gather(1, t.tensor(a + [EOS])[:, None]).sum()
        assert abs(l - ref) < 1e-4, (l, ref)
print(f"{green}decode matches hf.generate; answer_logprob matches the full-forward sum (incl. a 2-token answer){endc}")

print(f"{purple}=== transplant ==={endc}")
it = items[0]
dons = donors(it, 23, 17, 9)
prefix, tail = prompt_ids(tok, shots, it, 100)
src_tails = [tail] + [prompt_ids(tok, shots, dons[kind], 100)[1] for kind in ["xk2", "x", "k1", "k2"]]
reg = regions(tok, tail, 100)
F = reg["filler"]
L = model.cfg.n_layers
everything = [p for name in ["q_defs", "q_line", "label", "filler", "post"] for p in reg[name]]
scopes = [
    {"name": "none", "src": 1, "pos": [], "layers": [], "frozen": []},
    {"name": "frozen_target", "src": 1, "pos": [], "layers": [], "frozen": F},
    {"name": "all_but_last", "src": 1, "pos": [0] + everything, "layers": range(L), "frozen": []},
    {"name": "filler", "src": 1, "pos": F, "layers": range(L), "frozen": F},
    {"name": "filler_post_frozen", "src": 1, "pos": F, "layers": range(L), "frozen": F + reg["post"]},
    {"name": "first_50", "src": 1, "pos": F[:50], "layers": range(L), "frozen": F},
    {"name": "first_50_free", "src": 2, "pos": F[:50], "layers": range(L), "frozen": []},
    {"name": "layers_ge_2", "src": 1, "pos": F, "layers": range(2, L), "frozen": F},
    {"name": "q_defs", "src": 1, "pos": reg["q_defs"], "layers": range(L), "frozen": reg["q_line"] + reg["label"] + F},
]
cache = prefill(model, prefix)
logp = transplant(model, cache, src_tails, scopes)
n_src = len(src_tails)
row = {sc["name"]: logp[n_src + i] for i, sc in enumerate(scopes)}
gap = (logp[0] - logp[1]).abs().max().item()
for name, ref_row in [("none", 0), ("frozen_target", 0), ("all_but_last", 1)]:
    d = (row[name] - logp[ref_row]).abs().max().item()
    print(f"{cyan}{name} vs source row {ref_row}: {d:.1e} (clean target vs donor gap {gap:.2f}){endc}")
    assert d < 1e-5
alone = transplant(model, cache, src_tails, scopes[3:4])[n_src]
print(f"{cyan}filler scope alone vs in a batch of {len(scopes)}: {(alone - row['filler']).abs().max().item():.1e}{endc}")
assert (alone - row["filler"]).abs().max() < 1e-5

# Independent reference: no prefix cache, no in-batch sourcing. Capture every row's decoder-layer inputs with plain
# forward pre-hooks, then rerun the target with pre-hooks that overwrite each scope's positions layer by layer.
layers = hf.model.layers
P = len(prefix)
def run_with_pre_hooks(ids: list[int], hook_fn) -> t.Tensor:
    handles = [layers[l].register_forward_pre_hook(lambda mod, args, l=l: hook_fn(l, args)) for l in range(L)]
    out = full_logits(ids)[-1].log_softmax(-1)
    for h in handles:
        h.remove()
    return out
acts = []
for tl in src_tails:
    acts.append([None] * L)
    run_with_pre_hooks(prefix + tl, lambda l, args: acts[-1].__setitem__(l, args[0][0].clone()))
for sc in scopes:
    S = t.full((L, len(tail)), -1)  # -1: computed normally
    S[:, sc["frozen"]] = 0
    for l in sc["layers"]:
        S[l, sc["pos"]] = sc["src"]
    def overwrite(l, args, S=S):
        h = args[0].clone()
        for p in (S[l] >= 0).nonzero()[:, 0].tolist():
            h[0, P + p] = acts[S[l, p]][l][P + p]
        return (h, *args[1:])
    ref = run_with_pre_hooks(prefix + tail, overwrite)
    d = (ref - row[sc["name"]]).abs().max().item()
    print(f"{cyan}{sc['name']:>20}: in-batch transplant vs pre-hook reference {d:.1e}; shift from clean {(row[sc['name']] - logp[0]).abs().max().item():.2f}{endc}")
    assert d < 1e-4
print(f"{green}all gates pass{endc}")
