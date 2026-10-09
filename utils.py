import os
import re
import sys
import copy
import json
import types

import torch as t
import torch.nn.functional as F
from torch import Tensor
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, DeepseekV4Config, DeepseekV4ForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as mdv4
from transformer_lens.model_bridge import TransformerBridge

sys.path.append(os.path.dirname(hf_hub_download("deepseek-ai/DeepSeek-V4-Flash", "encoding/encoding_dsv4.py")))  # DeepSeek's prompt renderer (V4 has no chat template)
from encoding_dsv4 import encode_messages

from prompts import build_messages

MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"
EOS = 1  # <｜end▁of▁sentence｜>
EXPR = re.compile(r"(twice|three times) the number for (\w+) (plus|minus) (\d+)")
COEF = {"twice": 2, "three times": 3}
SIGN = {"plus": 1, "minus": -1}

def tiny_bridge(seed: int = 0, device: str = "cpu") -> TransformerBridge:
    """Random-weight fp32 V4 with the real tokenizer and vocab: 4 layers (sliding, CSA, HCA, CSA; the first 3 MoE hash-routed), for local tests."""
    cfg = json.load(open(hf_hub_download(MODEL_ID, "config.json")))
    for key in ["quantization_config", "torch_dtype", "transformers_version"]:
        cfg.pop(key)
    cfg.update(hidden_size=256, num_hidden_layers=4, compress_ratios=[0, 4, 128, 4], num_attention_heads=4, head_dim=64, qk_rope_head_dim=16, q_lora_rank=64, o_groups=2, o_lora_rank=32, index_n_heads=4, index_head_dim=32, n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=64, initializer_range=0.05)
    t.manual_seed(seed)
    hf = DeepseekV4ForCausalLM(DeepseekV4Config(**cfg)).to(device).eval()
    for name, buf in hf.named_buffers():  # init leaves the hash routing table at zero (every token to expert 0)
        if name.endswith("tid2eid"):
            buf.copy_(t.randint_like(buf, 0, cfg["n_routed_experts"]))
    for name, p in hf.named_parameters():  # and these at zero
        if name.endswith(("position_bias", "sinks", "hc_head.base", "attn_hc.base", "ffn_hc.base")):
            p.data.normal_(0, 0.5)
    snap = os.path.dirname(hf_hub_download(MODEL_ID, "tokenizer.json"))
    model = TransformerBridge.boot_transformers(snap, hf_model=hf, tokenizer=AutoTokenizer.from_pretrained(snap), dtype=t.float32, device=device)
    model.eval()
    model.requires_grad_(False)
    return model

def _fp32_topk_forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    logits = F.linear(hidden_states.reshape(-1, self.hidden_dim).float(), self.weight.float())
    scores = self.score_fn(logits)
    indices = t.topk(scores + self.e_score_correction_bias.float(), self.top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    return logits, weights / (weights.sum(dim=-1, keepdim=True) + 1e-20) * self.routed_scaling_factor, indices

def _fp32_hash_forward(self, hidden_states: Tensor, input_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    logits = F.linear(hidden_states.reshape(-1, self.hidden_dim).float(), self.weight.float())
    indices = self.tid2eid[input_ids.reshape(-1)].long()
    weights = self.score_fn(logits).gather(1, indices)
    return logits, weights / (weights.sum(dim=-1, keepdim=True) + 1e-20) * self.routed_scaling_factor, indices

def fp32_routers(model: TransformerBridge, on: bool = True):
    """Route in fp32 as DeepSeek's reference inference/model.py does, or (on=False) restore HF's routing in the bf16 stream dtype. fp32 means fp32 router logits (the gate weight is stored in bf16, so the upcast is exact) and fp32 routing weights, so the 6 expert outputs are also weighted and summed in fp32 like the reference; the selection bias already loads in fp32. Patches each router instance; under a device_map, accelerate's hook calls the instance's _old_forward, so that is the slot replaced."""
    for layer in model.original_model.model.layers:
        gate = layer.mlp.gate.original_component
        is_hash = isinstance(gate, mdv4.DeepseekV4HashRouter)
        slot = "_old_forward" if hasattr(gate, "_old_forward") else "forward"
        if not hasattr(gate, "hf_forward_"):
            gate.hf_forward_ = getattr(gate, slot)
        setattr(gate, slot, types.MethodType(_fp32_hash_forward if is_hash else _fp32_topk_forward, gate) if on else gate.hf_forward_)
        x = t.zeros(1, 1, gate.hidden_dim, dtype=gate.weight.dtype, device=gate.weight.device)
        logits = gate(x, t.zeros(1, 1, dtype=t.long, device=x.device))[0] if is_hash else gate(x)[0]
        assert logits.dtype == (t.float32 if on else gate.weight.dtype), (slot, logits.dtype)
        assert is_hash or gate.e_score_correction_bias.dtype == t.float32

def edit_item(item: dict, x: int | None = None, k1: int | None = None, k2: int | None = None) -> dict:
    """The item with x's literal, y's constant k1 and/or the question's constant k2 replaced, and every value, the chain and the answer recomputed."""
    q = item["queried_term"]
    defs = []
    for name, v in item["definitions"]:
        if name == item["x_name"] and x is not None:
            v = x
        if name == q and k1 is not None:
            v = re.sub(r"\d+$", str(k1), v)
        defs.append([name, v])
    vals = {}
    for name, v in defs:  # definitions only refer to earlier ones
        m = EXPR.fullmatch(str(v))
        vals[name] = v if m is None else COEF[m[1]] * vals[m[2]] + SIGN[m[3]] * int(m[4])
    k2 = item["constant"] if k2 is None else k2
    y, x_val, c2 = vals[q], vals[item["x_name"]], item["coefficient"]
    answer = c2 * y + SIGN[item["operation"]] * k2
    chain = {"x": x_val, "c1x": COEF[EXPR.fullmatch(dict(defs)[q])[1]] * x_val, "y": y, "c2y": c2 * y, "answer": answer}
    edited = {k: v for k, v in item.items() if k not in ("distractors", "rivals")}  # stale after an edit
    return {**edited, "definitions": defs, "values": vals, "queried_value": y, "constant": k2, "question": re.sub(r"\d+\?$", f"{k2}?", item["question"]), "answer": answer, "chain": chain}

def with_coefs(item: dict, c1: int, c2: int) -> dict:
    """The item with y's coefficient set to c1 and the question's to c2 (x, both constants and every other definition unchanged), and y, the chain and the answer recomputed."""
    word = {2: "twice", 3: "three times", 4: "four times", 5: "five times"}
    q, x = item["queried_term"], item["chain"]["x"]
    m = EXPR.fullmatch(dict(item["definitions"])[q])
    assert m[2] == item["x_name"]
    defs = [[name, f"{word[c1]} the number for {m[2]} {m[3]} {m[4]}" if name == q else v] for name, v in item["definitions"]]
    question, n = re.subn(r"^What is (twice|three times) ", f"What is {word[c2]} ", item["question"])
    assert n == 1, item["question"]
    y = c1 * x + SIGN[m[3]] * int(m[4])
    answer = c2 * y + SIGN[item["operation"]] * item["constant"]
    chain = {"x": x, "c1x": c1 * x, "y": y, "c2y": c2 * y, "answer": answer}
    kept = {k: v for k, v in item.items() if k not in ("values", "distractors", "rivals")}  # stale after the edit
    return {**kept, "definitions": defs, "coefficient": c2, "queried_value": y, "question": question, "answer": answer, "chain": chain}

def donors(item: dict, x: int, k1: int, k2: int) -> dict[str, dict]:
    """The four donor items for one (x', k1', k2') draw: the main x+k2 donor and the x-, k1- and k2-only controls."""
    return {"xk2": edit_item(item, x=x, k2=k2), "x": edit_item(item, x=x), "k1": edit_item(item, k1=k1), "k2": edit_item(item, k2=k2)}

def readout_numbers(item: dict, dons: dict[str, dict]) -> dict[str, int]:
    """Answers read at the answer position: the target's, through_y (target's second hop on the donor's y, i.e. the x-only donor's answer), and each other donor's."""
    return {"a_T": item["answer"], "ty": dons["x"]["answer"], "a_xk2": dons["xk2"]["answer"], "a_k1": dons["k1"]["answer"], "a_k2": dons["k2"]["answer"]}

def valid_donors(item: dict, dons: dict[str, dict], min_gap: int = 5) -> bool:
    """Every value non-negative, readout numbers single-token (0..999), pairwise at least min_gap apart, and none of them a number written in any of the prompts."""
    nums = sorted(readout_numbers(item, dons).values())
    shown = {int(s) for it in [item, *dons.values()] for s in re.findall(r"\d+", json.dumps(it["definitions"]) + it["question"])}
    non_negative = all(min(d["values"].values()) >= 0 for d in dons.values())
    return non_negative and 0 <= nums[0] and nums[-1] < 1000 and min(b - a for a, b in zip(nums, nums[1:])) >= min_gap and shown.isdisjoint(nums)

def prompt_ids(tok, shots: list[dict], item: dict, k: int) -> tuple[list[int], list[int]]:
    """Olivia's k-dot prompt as (shared prefix: system + few-shot turns, item tail: last user turn + '<｜Assistant｜></think>')."""
    text = encode_messages(build_messages(shots, item, "dots", k), thinking_mode="chat")
    cut = text.rindex("<｜User｜>")
    prefix, tail = (tok(s, add_special_tokens=False).input_ids for s in (text[:cut], text[cut:]))
    assert prefix + tail == tok(text, add_special_tokens=False).input_ids
    return prefix, tail

def regions(tok, tail: list[int], k: int) -> dict[str, list[int]]:
    """Tail positions of each prompt region. The last position ('</think>') reads out the answer."""
    T = len(tail)
    q_start = [i for i, tid in enumerate(tail) if tok.decode([tid]).strip() == "Question"]
    assert len(q_start) == 1, q_start
    reg = {
        "q_defs": list(range(1, q_start[0])),  # after '<｜User｜>', through the last definition
        "q_line": list(range(q_start[0], T - 7 - k)),
        "label": list(range(T - 7 - k, T - 4 - k)),
        "filler": list(range(T - 4 - k, T - 4)),
        "post": list(range(T - 4, T - 1)),
        "last": [T - 1],
    }
    text = {name: tok.decode([tail[i] for i in pos]) for name, pos in reg.items()}
    assert text["label"] == ("Filler:" if k else "Filler:\n\n") and text["filler"] == (" ." * k + "\n\n" if k else ""), text
    assert text["post"] == "Answer:<｜Assistant｜>" and text["last"] == "</think>" and text["q_line"].startswith("Question:"), text
    return reg

def answer_ids(tok, n: int) -> list[int]:
    return tok(str(n), add_special_tokens=False).input_ids

def prefill(model: TransformerBridge, ids: list[int]) -> DynamicCache:
    cache = DynamicCache(config=model.original_model.config)
    model(t.tensor([ids], device=model.cfg.device), past_key_values=cache, use_cache=True, logits_to_keep=1)
    return cache

def expand_cache(cache: DynamicCache, n: int) -> DynamicCache:
    """Fresh copy of a batch-1 cache repeated to batch n, V4 compressor and indexer state included. A forward mutates the cache it continues from, so every forward needs its own copy."""
    cache = copy.deepcopy(cache)
    rep = lambda v: v.expand(n, *v.shape[1:]).clone() if t.is_tensor(v) and v.ndim else v
    for layer in cache.layers:
        for key, v in vars(layer).items():
            setattr(layer, key, {kk: rep(vv) for kk, vv in v.items()} if isinstance(v, dict) else rep(v))
    return cache

def tail_logits(model: TransformerBridge, prefix_cache: DynamicCache, rows: list[list[int]], fwd_hooks: list = [], n_last: int = 1) -> Tensor:
    """fp32 logits [B, n_last, V] at each row's last n_last real positions. Rows are right-padded with EOS (exact on V4; left padding is not) and continue the batch-1 prefix cache. fp32 unembed of the final normed state, since bf16 logits carry ~0.1 nat of rounding."""
    lens = [len(r) for r in rows]
    T, keep = max(lens), max(lens) - min(lens) + n_last
    ids = t.tensor([r + [EOS] * (T - len(r)) for r in rows], device=model.cfg.device)
    final = {}
    def grab(act, hook):
        final["h"] = act
    model.run_with_hooks(ids, fwd_hooks=[*fwd_hooks, ("unembed.hook_in", grab)], past_key_values=expand_cache(prefix_cache, len(rows)), use_cache=True, logits_to_keep=keep)
    h = final["h"]
    idx = t.tensor([[n - T + keep - n_last + j for j in range(n_last)] for n in lens], device=h.device)
    h = h[t.arange(len(rows), device=h.device)[:, None], idx]
    W = model.original_model.lm_head.weight
    return h.to(W.device).float() @ W.float().T

def decode(model: TransformerBridge, prefix_cache: DynamicCache, tails: list[list[int]], n_new: int = 6, temp: float = 0.0) -> tuple[list[list[int]], Tensor]:
    """Greedy (temp=0) or sampled continuations of each tail, up to and including EOS, and the first step's log-probs [B, V]. Reruns tail+generated every step (right-padded rows cannot share a continued cache)."""
    gens = [[] for _ in tails]
    for step in range(n_new):
        logits = tail_logits(model, prefix_cache, [tl + g for tl, g in zip(tails, gens)])[:, -1]
        if step == 0:
            first = logits.log_softmax(-1)
        nxt = logits.argmax(-1) if temp == 0 else t.multinomial((logits / temp).softmax(-1), 1)[:, 0]
        gens = [g if EOS in g else g + [x] for g, x in zip(gens, nxt.tolist())]
        if all(EOS in g for g in gens):
            break
    return gens, first

def answer_logprob(model: TransformerBridge, prefix_cache: DynamicCache, tails: list[list[int]], answers: list[list[int]]) -> Tensor:
    """Teacher-forced log P(answer tokens, then EOS | tail) per row: the probability that a T=1 sample is exactly that answer."""
    n_last = max(map(len, answers)) + 1
    logp = tail_logits(model, prefix_cache, [tl + a for tl, a in zip(tails, answers)], n_last=n_last).log_softmax(-1)
    return t.stack([logp[i, range(n_last - len(a) - 1, n_last), a + [EOS]].sum() for i, a in enumerate(answers)])

def scope_S(n_src: int, scopes: list[dict], n_layers: int, T: int) -> Tensor:
    """Source map S [n_src + len(scopes), n_layers, T]: the row whose layer-l input row b takes at tail position p. Source rows and unlisted positions read themselves (computed normally); a scope's `frozen` positions read row 0 (the clean target) at every layer; its `pos` positions read row `src` at its `layers`, row 0 at the others when also frozen."""
    S = t.arange(n_src + len(scopes))[:, None, None].repeat(1, n_layers, T)
    for i, sc in enumerate(scopes):
        b = n_src + i
        S[b][:, sc["frozen"]] = 0
        for l in sc["layers"]:
            S[b, l, sc["pos"]] = sc["src"]
    return S

def source_hooks(S: Tensor) -> list:
    """Hooks at every decoder-layer input (the [B, T, 4, D] stream stack) that give row b, position p the activation of row S[b, l, p]."""
    def hook(act, hook):
        assert act.shape[:2] == (S.shape[0], S.shape[2]), act.shape
        return act[S[:, hook.layer()].to(act.device), t.arange(S.shape[2], device=act.device)]
    return [(f"blocks.{l}.hook_in", hook) for l in range(S.shape[1])]

def transplant(model: TransformerBridge, prefix_cache: DynamicCache, src_tails: list[list[int]], scopes: list[dict]) -> Tensor:
    """One forward of the source rows (row 0 the target, then donors; equal-length tails) plus one target row per scope, each scope's positions sourced per scope_S. Returns answer-position log-probs [n_src + len(scopes), V]; the source rows are the clean runs of the same batch."""
    S = scope_S(len(src_tails), scopes, model.cfg.n_layers, len(src_tails[0]))
    rows = src_tails + [src_tails[0]] * len(scopes)
    return tail_logits(model, prefix_cache, rows, source_hooks(S))[:, -1].log_softmax(-1)
