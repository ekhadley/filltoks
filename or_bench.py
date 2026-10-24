"""Black-box filler uplift through OpenRouter on DeepSeek V4.1 Flash 1017, Qwen3.8-27B and gpt-oss-120b, no reasoning, greedy (T=0).
Four tasks: Olivia's equations, 1-fact addition (the V4 Flash known-fact set), Brauer's 2-fact addition (first 500), 3-addend sums. k dots on the 'Filler:' line, the same filler in every shot.
Usage: python or_bench.py probe <model> <provider>... | run <model> | plot. Rows go to results/or_{model}.jsonl."""
import asyncio
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
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
}


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


def harmony(tok, messages: list[dict]) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, reasoning_effort="low") + "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"


def lead_int(text: str | None) -> int | None:
    """The leading integer: crusoe does not stop at gpt-oss's <|end|> and returns the next turn as text after the answer. Same as prompts.parse_answer on the chat models' bare-integer replies."""
    m = re.match(r"\s*(-?\d+)(?!\d)", text or "")
    return int(m.group(1)) if m else None


async def ask(m: dict, tok, messages: list[dict], provider: str) -> dict:
    if m["mode"] == "chat":
        body = await o.chat(messages, m["model"], provider=provider, reasoning=False, max_tokens=16, temperature=0.0)
        n_local = None
    else:
        prompt = harmony(tok, messages)
        body = await o.complete(prompt, m["model"], provider=provider, max_tokens=16, temperature=0.0, stop=["<|end|>", "<|return|>"])
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
    jobs = [(task, it, k, build_messages(shots, it, k=k)) for task, (shots, items) in tasks().items() for it in items for k in KS]
    rs = await o.gather_bar([ask(m, tok, msgs, m["provider"]) for *_, msgs in jobs], concurrency=48, desc=name)
    with open(f"results/or_{name}.jsonl", "w") as f:
        for (task, it, k, _), r in zip(jobs, rs):
            row = {"task": task, "idx": it["idx"], "k": k, "answer": it["answer"], **(r or {"text": None, "error": True})}
            row["correct"] = lead_int(row["text"]) == it["answer"]
            f.write(json.dumps(row) + "\n")
    print(f"{name}: {sum(r is None for r in rs)} / {len(rs)} failed, ${sum(r['cost'] for r in rs if r):.2f}")


def plot():
    """Greedy accuracy at k = 0, 10, 100 per task (panel) and model (group), 95% CIs; the numbers over the bars are uplift vs k=0 in pp with a paired 95% CI."""
    plt.style.use("dark_background")
    names = [n for n in MODELS if os.path.exists(f"results/or_{n}.jsonl")]
    rows = {n: [json.loads(l) for l in open(f"results/or_{n}.jsonl")] for n in names}
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), sharey=True)
    colors = ["#5b8def", "#f2a541", "#e0605b"]
    for ax, task in zip(axes, tasks()):
        for i, n in enumerate(names):
            by = {(r["idx"], r["k"]): r["correct"] for r in rows[n] if r["task"] == task}
            idxs = sorted({i_ for i_, _ in by})
            c = {k: np.array([by[(j, k)] for j in idxs], float) for k in KS}
            for j, k in enumerate(KS):
                x, acc = i + (j - 1) * 0.27, c[k].mean()
                ax.bar(x, acc, 0.25, color=colors[j], label=f"{k} dots" if i == 0 else None, yerr=1.96 * np.sqrt(acc * (1 - acc) / len(idxs)), ecolor="#bbbbbb", capsize=2)
                if k:
                    d = c[k] - c[0]
                    ax.text(x, acc + 0.06, f"{100 * d.mean():+.1f}\n±{196 * d.std(ddof=1) / np.sqrt(len(d)):.1f}", ha="center", fontsize=7.5, color="#dddddd")
        ax.set_xticks(range(len(names)), names, fontsize=9)
        ax.set_title(f"{task} (n={len(idxs)})")
        ax.set_ylim(0, 1.2)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("greedy accuracy, no reasoning")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Filler uplift through OpenRouter: k dots on the 'Filler:' line, same filler in every shot, T=0. Numbers: uplift vs 0 dots, pp ± paired 95% CI", fontsize=11)
    fig.tight_layout()
    out = "/home/ek/wgmn/mats12_sprint/figures/filler_uplift_or.png"
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "probe":
        asyncio.run(probe(sys.argv[2], sys.argv[3:]))
    elif cmd == "run":
        asyncio.run(run(sys.argv[2]))
    else:
        plot()
