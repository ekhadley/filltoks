"""Black-box filler uplift through OpenRouter on DeepSeek V4.1 Flash 1017, Qwen3.8-27B and gpt-oss-120b, no reasoning, greedy (T=0).
Four tasks: Olivia's equations, 1-fact addition (the V4 Flash known-fact set), Brauer's 2-fact addition (first 500), 3-addend sums. k dots on the 'Filler:' line, the same filler in every shot.
gpt-oss CoT-control checkpoints (and base gpt-oss for a same-sampler baseline) run on Tinker with the same harmony prompt, greedy.
Usage: python or_bench.py probe <model> <provider>... | run <model> | plot <figure name> <model>... | sweep <model> | plot_sweep <model>.... Rows go to results/or_{model}.jsonl.
sweep: easier task variants (easy_tasks), 100 items each at 0 and 100 dots, to find where gpt-oss and Qwen solve 10-80% with no CoT; rows in results/sweep_{model}.jsonl."""
import asyncio
import json
import random
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import tinker
from transformers import AutoTokenizer

from mechtools import openrouter as o

from prompts import build_messages

o.load_env()
KS = [0, 10, 100]
# mode "chat": /chat/completions with reasoning disabled; "harmony": raw /completions on the gpt-oss chat template (reasoning low) with an empty analysis message, then the final channel (gpt-oss cannot turn reasoning off; opening the final channel directly gave "...", "?" and blank answers)
MODELS = {
    "deepseek-v4.1-flash": {"model": "deepseek/deepseek-v4.1-flash-1017", "hf": "deepseek-ai/DeepSeek-V4.1-Flash-1017", "mode": "chat", "provider": "parasail"},
    "qwen3.8-27b": {"model": "qwen/qwen3.8-27b", "hf": "Qwen/Qwen3.8-27B", "mode": "chat", "provider": "deepinfra"},
    "gpt-oss-120b": {"model": "openai/gpt-oss-120b", "hf": "openai/gpt-oss-120b", "mode": "harmony", "provider": "crusoe"},
    # mode "tinker": the harmony prompt sampled on Tinker (model_path None = base)
    "gpt-oss-120b-tinker": {"model": "openai/gpt-oss-120b", "hf": "openai/gpt-oss-120b", "mode": "tinker", "model_path": None},
    "grpo3-s100": {"model": "openai/gpt-oss-120b", "hf": "openai/gpt-oss-120b", "mode": "tinker", "model_path": "tinker://ec5bf27e-1436-5923-a854-96774a207315:train:0/sampler_weights/cotctl-grpo3-step100"},
    "manytypes1b-s150": {"model": "openai/gpt-oss-120b", "hf": "openai/gpt-oss-120b", "mode": "tinker", "model_path": "tinker://f6204b00-be4d-51fd-b1d8-6d467ff4ba75:train:0/sampler_weights/cotctl-manytypes1b-step150"},
    "st1-ignore2-s60": {"model": "openai/gpt-oss-120b", "hf": "openai/gpt-oss-120b", "mode": "tinker", "model_path": "tinker://1e287f2d-ec6a-5554-8582-db39196492df:train:0/sampler_weights/cotctl-st1-ignore2-step60"},
}
TK_PARAMS = tinker.SamplingParams(temperature=0.0, max_tokens=8, stop=[200007, 200002, 200012])  # <|end|>, <|return|>, <|call|>


def tasks() -> dict[str, tuple[list[dict], list[dict]]]:
    """task -> (shots, items)."""
    jl = lambda p: [json.loads(l) for l in open(p)]
    d = json.load(open("data/2fact_addition.json"))
    two = lambda e, i: {"idx": i, "type": "2fact_addition", "question": f"What is {e['fact_phrase_1']} plus {e['fact_phrase_2']}?", "answer": e["answer"]}
    three = [{"idx": i, "type": "nfact_addition", "question": r["question"], "answer": r["answer"]} for i, r in enumerate(r for r in jl("data/facts/addition_3_4.jsonl") if r["num_addends"] == 3)]
    return {
        "equations": (jl("data/fewshot.jsonl"), jl("data/eval.jsonl")),
        "1-fact addition": (jl("data/facts/fewshot_v4flash_packed.jsonl"), jl("data/facts/eval_v4flash_packed.jsonl")),
        "2-fact addition": ([two(e, i) for i, e in enumerate(d["few_shot_facts"])], [two(e, i) for i, e in enumerate(d["examples"][:500])]),
        "3-addend sum": (three[:10], three[10:]),
    }


COEF = {1: "the number for", 2: "twice the number for", 3: "three times the number for"}


def gen_equation(rng: random.Random, terms: int, coefs: list[int]) -> dict:
    """One-hop item in Olivia's format: the queried variable is a plain number (chain length 0); other terms alternate numbers and single-hop expressions of an earlier number."""
    names = []
    while len(names) < terms:
        n = rng.choice("bcdfghjklmnpqrstvwxyz") + rng.choice("aeiou") + rng.choice("bcdfghjklmnpqrstvwxyz")
        if n not in names:
            names.append(n)
    defs, values = [], {}
    for i, n in enumerate(names):
        if i % 2 == 0 or terms == 1:
            values[n] = rng.randint(10, 99)
            defs.append([n, values[n]])
        else:
            src, c, k, op = names[i - 1], rng.choice(coefs), rng.randint(1, 50), rng.choice(["plus", "minus"])
            values[n] = c * values[src] + (k if op == "plus" else -k)
            defs.append([n, f"{COEF[c]} {src} {op} {k}"])
    q = rng.choice([n for n in names if isinstance(dict(defs)[n], int)])
    c, k, op = rng.choice(coefs), rng.randint(1, 50), rng.choice(["plus", "minus"])
    return {"type": "chained_var_binding", "definitions": defs, "question": f"What is {COEF[c]} {q} {op} {k}?", "answer": c * values[q] + (k if op == "plus" else -k)}


def gen_arith(rng: random.Random, kind: str) -> dict:
    d2 = lambda: rng.randint(10, 99)
    if kind == "add2": a, b = d2(), d2(); q, ans = f"{a} + {b}", a + b
    elif kind == "add3": a, b, c = d2(), d2(), d2(); q, ans = f"{a} + {b} + {c}", a + b + c
    elif kind == "add3digit": a, b = rng.randint(100, 999), rng.randint(100, 999); q, ans = f"{a} + {b}", a + b
    elif kind == "mul2x1": a, b = d2(), rng.randint(2, 9); q, ans = f"{a} x {b}", a * b
    elif kind == "mul2x2": a, b = d2(), d2(); q, ans = f"{a} x {b}", a * b
    else: raise ValueError(kind)
    return {"type": "arithmetic", "question": f"What is {q}?", "answer": ans}


def easy_tasks(n: int = 100) -> dict[str, tuple[list[dict], list[dict]]]:
    """Easier variants, each with its own 10 shots from the same generator: one-hop equations (2 or 5 terms, coefficients 2-3; 5 terms with coefficient 1), plain arithmetic, and the 1-fact set with the addend dropped (bare retrieval) or replaced by a 1-digit number."""
    out = {}
    for name, terms, coefs in [("eq1hop_t2", 2, [2, 3]), ("eq1hop_t5", 5, [2, 3]), ("eq1hop_c1_t5", 5, [1])]:
        rng = random.Random(name)
        out[name] = ([gen_equation(rng, terms, coefs) for _ in range(10)], [gen_equation(rng, terms, coefs) | {"idx": i} for i in range(n)])
    for kind in ["add2", "add3", "add3digit", "mul2x1", "mul2x2"]:
        rng = random.Random(kind)
        out[kind] = ([gen_arith(rng, kind) for _ in range(10)], [gen_arith(rng, kind) | {"idx": i} for i in range(n)])
    jl = lambda p: [json.loads(l) for l in open(p)]
    shots, items = jl("data/facts/fewshot_v4flash_packed.jsonl"), jl("data/facts/eval_v4flash_packed.jsonl")[:n]
    bare = lambda it: it | {"question": re.sub(r" plus \d+\?$", "?", it["question"]), "answer": it["fact"]}
    out["fact_bare"] = ([bare(s) for s in shots], [bare(it) for it in items])
    rng = random.Random("fact_plus1")
    small = lambda it: (lambda d: it | {"question": re.sub(r" plus \d+\?$", f" plus {d}?", it["question"]), "answer": it["fact"] + d})(rng.randint(2, 9))
    out["fact_plus1"] = ([small(s) for s in shots], [small(it) for it in items])
    return out


async def sweep(name: str, boxed: bool = False):
    m, tok = MODELS[name], AutoTokenizer.from_pretrained(MODELS[name]["hf"])
    client = None if m["mode"] != "tinker" else await tinker.ServiceClient().create_sampling_client_async(**({"model_path": m["model_path"]} if m["model_path"] else {"base_model": m["model"]}))
    jobs = [(task, it, k, build_messages(shots, it, k=k, boxed=boxed)) for task, (shots, items) in easy_tasks().items() for it in items for k in [0, 100]]
    rs = await o.gather_bar([ask(m, tok, msgs, m.get("provider"), client) for *_, msgs in jobs], concurrency=48, desc=f"sweep {name}")
    with open(f"results/sweep_{'boxed_' if boxed else ''}{name}.jsonl", "w") as f:
        for (task, it, k, _), r in zip(jobs, rs):
            row = {"task": task, "idx": it["idx"], "k": k, "answer": it["answer"], **(r or {"text": None, "error": True})}
            row["correct"] = lead_int(row["text"]) == it["answer"]
            f.write(json.dumps(row) + "\n")
    print(f"{name}: {sum(r is None for r in rs)} / {len(rs)} failed, ${sum(r['cost'] for r in rs if r):.2f}")
    for task in easy_tasks():
        print(f"  {task:14s} " + "  ".join(f"k={k}: {sum(r['correct'] for (t, _, kk, _), r in zip(jobs, rs) if t == task and kk == k and r)}/100" for k in [0, 100]))


def plot_sweep(names: list[str]):
    """Accuracy at 0 and 100 dots per easy task (x) per model (panel), plain prompt and with the \\boxed{ prefill; the 10-80% band is shaded."""
    plt.style.use("dark_background")
    levels = list(easy_tasks())
    fig, axes = plt.subplots(1, len(names), figsize=(8 * len(names), 4.8), sharey=True, squeeze=False)
    for ax, n in zip(axes[0], names):
        arms = [("plain", "", "#5b8def", "#e0605b"), ("boxed", "boxed_", "#2d4f9e", "#9c3a36")]
        for a, (label, pre, c0, c100) in enumerate(arms):
            rows = [json.loads(l) for l in open(f"results/sweep_{pre}{n}.jsonl")]
            for j, (k, col) in enumerate([(0, c0), (100, c100)]):
                acc = [np.mean([r["correct"] for r in rows if r["task"] == t and r["k"] == k]) for t in levels]
                ax.bar(np.arange(len(levels)) + (2 * a + j - 1.5) * 0.2, acc, 0.19, color=col, label=f"{label}, {k} dots", yerr=[1.96 * np.sqrt(x * (1 - x) / 100) for x in acc], ecolor="#bbbbbb", capsize=1.5)
        ax.axhspan(0.1, 0.8, color="#ffffff", alpha=0.06)
        ax.set_xticks(range(len(levels)), levels, rotation=30, ha="right", fontsize=8)
        ax.set_title(n)
        ax.grid(axis="y", alpha=0.2)
    axes[0][0].set_ylabel("greedy accuracy, no reasoning (n=100)")
    axes[0][0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Difficulty sweep for no-CoT filler tasks: easier variants at 0 and 100 dots, plain prompt vs \\boxed{ answer prefill; shaded = the 10-80% band", fontsize=11)
    fig.tight_layout()
    out = "/home/ek/wgmn/mats12_sprint/figures/filler_sweep.png"
    fig.savefig(out, dpi=130)
    print(out)


def split_prefill(messages: list[dict]) -> tuple[list[dict], str]:
    """A trailing assistant message is the answer prefill."""
    return (messages[:-1], messages[-1]["content"]) if messages[-1]["role"] == "assistant" else (messages, "")


def harmony(tok, messages: list[dict]) -> str:
    messages, prefill = split_prefill(messages)
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, reasoning_effort="low") + "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>" + prefill


def qwen_raw(tok, messages: list[dict]) -> str:
    """Qwen's chat template with thinking off (an empty think block) and the prefill inside the assistant turn, for raw completions."""
    messages, prefill = split_prefill(messages)
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False) + prefill


def lead_int(text: str | None) -> int | None:
    """The leading integer: crusoe does not stop at gpt-oss's <|end|> and returns the next turn as text after the answer. Same as prompts.parse_answer on the chat models' bare-integer replies."""
    m = re.match(r"\s*(?:\\boxed\{)?\s*(-?\d+)(?!\d)", text or "")  # a provider may echo the \boxed{ prefill
    return int(m.group(1)) if m else None


async def ask(m: dict, tok, messages: list[dict], provider: str, client=None) -> dict:
    if m["mode"] == "tinker":
        ids = tok.encode(harmony(tok, messages), add_special_tokens=False)
        seq = (await client.sample_async(prompt=tinker.ModelInput.from_ints(ids), num_samples=1, sampling_params=TK_PARAMS)).sequences[0]
        return {"text": tok.decode(seq.tokens), "reasoning": None, "finish_reason": str(seq.stop_reason), "provider": "tinker", "model": m["model_path"] or m["model"], "prompt_tokens": len(ids),
                "completion_tokens": len(seq.tokens), "reasoning_tokens": 0, "cost": (0.33 * len(ids) + 0.84 * len(seq.tokens)) / 1e6, "n_local": len(ids)}
    if m["mode"] == "chat":
        # a trailing assistant message is sent as a prefill (DeepInfra continues it and echoes it in the reply)
        body = await o.chat(messages, m["model"], provider=provider, reasoning=False, max_tokens=16, temperature=0.0, stop=["}"] if split_prefill(messages)[1] else None)
        n_local = None
    else:
        prompt = harmony(tok, messages)
        body = await o.complete(prompt, m["model"], provider=provider, max_tokens=16, temperature=0.0, stop=["<|end|>", "<|return|>"] + (["}"] if split_prefill(messages)[1] else []))
        n_local = len(tok.encode(prompt, add_special_tokens=False))
    return {**o.flat(body), "n_local": n_local}


async def probe(name: str, providers: list[str]):
    """One equation item at k=0 and k=100 per provider: the reply, any reasoning, prompt tokens (vs the local count for harmony)."""
    m, tok = MODELS[name], AutoTokenizer.from_pretrained(MODELS[name]["hf"])
    shots, items = tasks()["equations"]
    for p in providers:
        for k in [0, 100]:
            try:
                r = await ask(m, tok, build_messages(shots, items[0], k=k), p)
                print(f"{p:14s} k={k:3d} text={r['text']!r} want={items[0]['answer']} reasoning={(r['reasoning'] or '')[:60]!r} rtok={r['reasoning_tokens']} ptok={r['prompt_tokens']} local={r['n_local']} finish={r['finish_reason']} ${r['cost']:.5f}")
            except Exception as e:
                print(f"{p:14s} k={k:3d} ERROR {str(e)[:200]}")


async def run(name: str):
    m, tok = MODELS[name], AutoTokenizer.from_pretrained(MODELS[name]["hf"])
    client = None if m["mode"] != "tinker" else await tinker.ServiceClient().create_sampling_client_async(**({"model_path": m["model_path"]} if m["model_path"] else {"base_model": m["model"]}))
    jobs = [(task, it, k, build_messages(shots, it, k=k)) for task, (shots, items) in tasks().items() for it in items for k in KS]
    rs = await o.gather_bar([ask(m, tok, msgs, m.get("provider"), client) for *_, msgs in jobs], concurrency=48, desc=name)
    with open(f"results/or_{name}.jsonl", "w") as f:
        for (task, it, k, _), r in zip(jobs, rs):
            row = {"task": task, "idx": it["idx"], "k": k, "answer": it["answer"], **(r or {"text": None, "error": True})}
            row["correct"] = lead_int(row["text"]) == it["answer"]
            f.write(json.dumps(row) + "\n")
    print(f"{name}: {sum(r is None for r in rs)} / {len(rs)} failed, ${sum(r['cost'] for r in rs if r):.2f}")


def plot(out_name: str, names: list[str]):
    """Greedy accuracy at k = 0, 10, 100 per task (panel) and model (group), 95% CIs; the numbers over the bars are uplift vs k=0 in pp with a paired 95% CI."""
    plt.style.use("dark_background")
    rows = {n: [json.loads(l) for l in open(f"results/or_{n}.jsonl")] for n in names}
    fig, axes = plt.subplots(1, 4, figsize=(5 * len(names) + 3, 5.2), sharey=True)
    colors = ["#5b8def", "#f2a541", "#e0605b"]
    top = 0.0
    for ax, task in zip(axes, tasks()):
        for i, n in enumerate(names):
            by = {(r["idx"], r["k"]): r["correct"] for r in rows[n] if r["task"] == task}
            idxs = sorted({i_ for i_, _ in by})
            c = {k: np.array([by[(j, k)] for j in idxs], float) for k in KS}
            for j, k in enumerate(KS):
                x, acc = i + (j - 1) * 0.27, c[k].mean()
                top = max(top, acc)
                ax.bar(x, acc, 0.25, color=colors[j], label=f"{k} dots" if i == 0 else None, yerr=1.96 * np.sqrt(acc * (1 - acc) / len(idxs)), ecolor="#bbbbbb", capsize=2)
                if k:
                    d = c[k] - c[0]
                    ax.text(x, acc + 0.012 * (1 if top < 0.5 else 5), f"{100 * d.mean():+.1f}\n±{196 * d.std(ddof=1) / np.sqrt(len(d)):.1f}", ha="center", fontsize=7.5, color="#dddddd")
        ax.set_xticks(range(len(names)), names, fontsize=8, rotation=20 if len(names) > 3 else 0)
        ax.set_title(f"{task} (n={len(idxs)})")
        ax.grid(axis="y", alpha=0.2)
    for ax in axes:
        ax.set_ylim(0, min(1.2, 2.2 * top))  # the axis fits the data: the organisms sit under 10%
    axes[0].set_ylabel("greedy accuracy, no reasoning")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Filler uplift, no reasoning: k dots on the 'Filler:' line, same filler in every shot, T=0. Numbers: uplift vs 0 dots, pp ± paired 95% CI", fontsize=11)
    fig.tight_layout()
    out = f"/home/ek/wgmn/mats12_sprint/figures/{out_name}.png"
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "probe":
        asyncio.run(probe(sys.argv[2], sys.argv[3:]))
    elif cmd == "run":
        asyncio.run(run(sys.argv[2]))
    elif cmd == "sweep":
        asyncio.run(sweep(sys.argv[2], boxed="--boxed" in sys.argv))
    elif cmd == "plot_sweep":
        plot_sweep(sys.argv[2:])
    else:
        plot(sys.argv[2], sys.argv[3:])
