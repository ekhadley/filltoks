#%% Setup: fact pool, model, tokenizer. Brauer et al.'s 1-fact addition on V4 Flash: "What is <fact phrase> plus x?", facts from Greenblatt's compose_facts (data/facts/, gitignored)

import json
import math
import random
import re
from collections import Counter

import torch as t

from mechtools import *

from prompts import parse_answer
from utils import MODEL_ID, EOS, tiny_bridge, fp32_routers, prompt_ids, answer_ids, prefill, decode, answer_logprob

t.set_grad_enabled(False)
PHRASE = [(re.compile(r"At what age did (.+?) die\?"), r"the age at which \1 died"), (re.compile(r"What is (the atomic number of .+|the number of .+)\?"), r"\1")]
facts = [{"type": "1fact_addition", "question": f["question"], "answer": f["answer"], "kind": name} for name in ["age", "atomic", "static"] for f in json.load(open(f"data/facts/{name}_facts.json"))]
for f in facts:
    f["phrase"] = next(pat.sub(rep, f["question"]) for pat, rep in PHRASE if pat.fullmatch(f["question"]))
order = list(range(len(facts)))
random.Random(42).shuffle(order)
for i, j in enumerate(order):
    facts[j]["idx"] = i
print(f"{cyan}{len(facts)} facts: {Counter(f['kind'] for f in facts)}{endc}")

TAG = "tiny"
model = tiny_bridge()
# TAG = "v4flash_packed"
# model = load_bridge(MODEL_ID)
# fp32_routers(model)
tok = model.tokenizer

def bench(shots: list[dict], items: list[dict], ks: list[int], batch_size: int = 16) -> list[dict]:
    """Greedy answer and exact P(answer + EOS) per item and k, one prefix cache per k."""
    records = []
    for k in ks:
        pts = [prompt_ids(tok, shots, it, k) for it in items]
        assert all(prefix == pts[0][0] for prefix, _ in pts)
        cache = prefill(model, pts[0][0])
        for i in pbar(range(0, len(items), batch_size), desc=f"k={k}"):
            batch, tails = items[i:i + batch_size], [tail for _, tail in pts[i:i + batch_size]]
            gens, first = decode(model, cache, tails)
            logp = answer_logprob(model, cache, tails, [answer_ids(tok, it["answer"]) for it in batch])
            for j, it in enumerate(batch):
                text = tok.decode([x for x in gens[j] if x != EOS])
                records.append({"idx": it["idx"], "k": k, "text": text, "parsed": parse_answer(text), "correct": parse_answer(text) == it["answer"], "logp": logp[j].item()})
        done = [r for r in records if r["k"] == k]
        n_correct = sum(r["correct"] for r in done)
        ci = wilson(n_correct, len(done))
        print(f"{cyan}k={k}: greedy {n_correct}/{len(done)} = {n_correct / len(done):.3f} [{ci[0]:.3f}, {ci[1]:.3f}], mean P(correct) {sum(math.exp(r['logp']) for r in done) / len(done):.3f}{endc}")
    return records

#%% 1-fact uplift: the bare fact question ("At what age did X die?"), 10-shot with other fact questions, at k = 0, 10, 100. Known facts (for the addition sets) are those right at k=0. Saves results/facts_known_{TAG}.json

KS = [0, 10, 100]

def report(items: list[dict], records: list[dict]):
    """Greedy accuracy by fact kind at each k, and the 0 -> k flip counts."""
    by = {(r["idx"], r["k"]): r for r in records}
    for kind in ["age", "atomic", "static"]:
        sel = [it["idx"] for it in items if it["kind"] == kind]
        if sel: print(f"{gray}{kind}: n={len(sel)}, greedy " + ", ".join(f"k={k} {sum(by[(i, k)]['correct'] for i in sel) / len(sel):.3f}" for k in KS) + endc)
    for k in KS[1:]:
        up = sum(not by[(it["idx"], 0)]["correct"] and by[(it["idx"], k)]["correct"] for it in items)
        down = sum(by[(it["idx"], 0)]["correct"] and not by[(it["idx"], k)]["correct"] for it in items)
        print(f"{cyan}0 -> {k}: wrong to right {up}, right to wrong {down}{endc}")

knowledge_check = True
if knowledge_check:
    shots, pool = [facts[j] for j in order[:10]], [facts[j] for j in order[10:]]
    recs = bench(shots, pool, KS)
    report(pool, recs)
    known = [f for f, r in zip(pool, [r for r in recs if r["k"] == 0]) if r["correct"]]
    print(f"{cyan}known at k=0: {len(known)}/{len(pool)}: {Counter(f['kind'] for f in known)}{endc}")
    json.dump({"shots": shots, "known": known, "records": recs}, open(f"results/facts_known_{TAG}.json", "w"))

#%% Build the 1-fact addition sets from the known facts (Brauer: x in 10..99, every fact once, then repeats with a new x, 10 shots); writes data/facts/eval_{TAG}.jsonl and fewshot_{TAG}.jsonl

build_sets = True
if build_sets:
    n_items = 500
    rng = random.Random(42)
    known = json.load(open(f"results/facts_known_{TAG}.json"))["known"]
    def with_x(f, idx):
        x = rng.randint(10, 99)
        return {"idx": idx, "type": "1fact_addition", "question": f"What is {f['phrase']} plus {x}?", "fact": f["answer"], "x": x, "answer": f["answer"] + x, "kind": f["kind"]}
    shots = [with_x(f, i) for i, f in enumerate(known[:10])]
    pool = known[10:]
    items = [with_x(f, i) for i, f in enumerate(pool + [rng.choice(pool) for _ in range(n_items - len(pool))])]
    for name, rows in [("fewshot", shots), ("eval", items)]:
        with open(f"data/facts/{name}_{TAG}.jsonl", "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"{cyan}{len(items)} items from {len(pool)} facts, answers {min(r['answer'] for r in items)}..{max(r['answer'] for r in items)}, multi-token answers {sum(len(answer_ids(tok, r['answer'])) > 1 for r in items)}{endc}")

#%% 1-fact addition uplift: greedy and P(correct) at k = 0, 10, 100 on the 1-fact addition sets; saves results/facts_bench_{TAG}.json

run_bench = True
if run_bench:
    items = [json.loads(l) for l in open(f"data/facts/eval_{TAG}.jsonl")]
    shots = [json.loads(l) for l in open(f"data/facts/fewshot_{TAG}.jsonl")]
    records = bench(shots, items, KS)
    report(items, records)
    json.dump(records, open(f"results/facts_bench_{TAG}.json", "w"))

#%% 2-fact addition uplift: Brauer et al.'s released set (data/2fact_addition.json: 1500 pairs of atomic numbers, "What is the atomic number of Helium plus the atomic number of Neon?", their 5 few-shot pairs), at k = 0, 10, 100; saves results/facts2_bench_{TAG}.json

run_2fact = True
if run_2fact:
    d = json.load(open("data/2fact_addition.json"))
    as_item = lambda e, i: {"idx": i, "type": "2fact_addition", "question": f"What is {e['fact_phrase_1']} plus {e['fact_phrase_2']}?", "facts": [e["fact_value_1"], e["fact_value_2"]], "answer": e["answer"], "kind": "atomic"}
    shots2 = [as_item(e, i) for i, e in enumerate(d["few_shot_facts"])]
    items2 = [as_item(e, i) for i, e in enumerate(d["examples"])]
    records = bench(shots2, items2, KS)
    report(items2, records)
    json.dump(records, open(f"results/facts2_bench_{TAG}.json", "w"))

#%% N-addend sums: Greenblatt's multi_hop addition generator (generate_addition_dataset.py, seed 42 + N, facts < 100 from ages, atomic numbers 11..100, static trivia, US House seats, state legislature seats; "What is (At what age did X die) + (atomic number of Y) + ...?"), 3 and 4 addends, 310 each in data/facts/addition_3_4.jsonl: first 10 are the shots; saves results/factsN_bench_{TAG}.json

run_nadd = True
if run_nadd:
    rows = [json.loads(l) for l in open("data/facts/addition_3_4.jsonl")]
    for n in [3, 4]:
        sel = [{"idx": i, "type": "nfact_addition", "question": r["question"], "facts": [c["value"] for c in r["chain"]], "answer": r["answer"], "kind": "mixed"} for i, r in enumerate(r for r in rows if r["num_addends"] == n)]
        print(f"{cyan}{n} addends: {len(sel) - 10} items, answers {min(r['answer'] for r in sel)}..{max(r['answer'] for r in sel)}, multi-token answers {sum(len(answer_ids(tok, r['answer'])) > 1 for r in sel)}{endc}")
        records = bench(sel[:10], sel[10:], KS)
        report(sel[10:], records)
        json.dump(records, open(f"results/facts{n}_bench_{TAG}.json", "w"))

#%% Uplift plot: mean P(correct) (bands: bootstrap over items) and greedy accuracy (hollow markers) vs k, 2 x 3 panels: Olivia's equations (P from the dose sweep, greedy from the lens runs), fact retrieval, 1-fact addition, 2-fact addition, 3- and 4-addend sums; writes figs/facts_uplift.html

plot_uplift = True
if plot_uplift:
    n_boot = 5000
    def from_records(recs: list[dict], kinds_by_idx: dict) -> tuple:
        """(P(correct) [n, len(KS)], greedy correct [n, len(KS)], kind per item) from bench records."""
        idxs = sorted({r["idx"] for r in recs})
        rec = {(r["idx"], r["k"]): r for r in recs}
        p = t.tensor([[math.exp(rec[(i, k)]["logp"]) for k in KS] for i in idxs])
        g = t.tensor([[float(rec[(i, k)].get("correct", math.nan)) for k in KS] for i in idxs])
        return p, g, [kinds_by_idx[i] for i in idxs]
    eq_items = [json.loads(l) for l in open("data/eval.jsonl")]
    eq_p, _, eq_kinds = from_records(json.load(open(f"results/dose_{TAG}.json")), {it["idx"]: f"first hop x{it['chain']['c1x'] // it['chain']['x']}" for it in eq_items})
    lens = {**t.load(f"results/lens_{TAG}.pt", weights_only=False), **t.load(f"results/lens_{TAG}_small.pt", weights_only=False)}
    eq_g = t.full((len(eq_items), len(KS)), math.nan)  # the lens runs cover the 598 items with single-token chain values; nan elsewhere
    for j, k in enumerate(KS):
        eq_g[lens[k]["idx"], j] = lens[k]["correct"].float()
    known = json.load(open(f"results/facts_known_{TAG}.json"))
    tasks = {
        "system of equations (Olivia's task)<br><span style='font-size:12px'>5 definitions, 2 hops: y = 3x - 30, answer = 2y + 29</span>": (eq_p, eq_g, eq_kinds),
        "fact retrieval<br><span style='font-size:12px'>At what age did X die?</span>": from_records(known["records"], {f["idx"]: f"{f['kind']} facts" for f in facts}),
        "1-fact addition<br><span style='font-size:12px'>What is the age at which X died plus 37?</span>": from_records(json.load(open(f"results/facts_bench_{TAG}.json")), {it["idx"]: f"{it['kind']} facts" for it in map(json.loads, open(f"data/facts/eval_{TAG}.jsonl"))}),
        "2-fact addition<br><span style='font-size:12px'>What is the atomic number of He plus the atomic number of Ne?</span>": from_records(json.load(open(f"results/facts2_bench_{TAG}.json")), {i: "all" for i in range(1500)}),
        **{f"{n}-addend sum<br><span style='font-size:12px'>What is (age at which X died) + (atomic number of Y) + ...?</span>": from_records(json.load(open(f"results/facts{n}_bench_{TAG}.json")), {i: "all" for i in range(310)}) for n in [3, 4]},
    }
    colors = {"all": "#e8e6dc", "age facts": SERIES[0], "atomic facts": SERIES[1], "static facts": SERIES[2], "first hop x2": SERIES[1], "first hop x3": SERIES[0]}
    fig = make_subplots(rows=2, cols=3, subplot_titles=list(tasks), shared_yaxes=True, vertical_spacing=0.14)
    for n, (task, (p, g, kinds)) in enumerate(tasks.items()):
        row, col = n // 3 + 1, n % 3 + 1
        lines = {"all": t.ones(len(p), dtype=t.bool), **{kind: t.tensor([kd == kind for kd in kinds]) for kind in sorted(set(kinds)) if kind != "all"}}
        for name, m in lines.items():
            pm, color = p[m], colors[name]
            boot = pm[t.randint(0, len(pm), (n_boot, len(pm)), generator=t.Generator().manual_seed(0))].mean(1)
            lo, hi = boot.quantile(0.025, 0), boot.quantile(0.975, 0)
            fig.add_trace(go.Scatter(x=KS + KS[::-1], y=hi.tolist() + lo.flip(0).tolist(), fill="toself", fillcolor=color, opacity=0.15, line={"width": 0}, hoverinfo="skip", showlegend=False), row=row, col=col)
            fig.add_trace(go.Scatter(x=KS, y=pm.mean(0).tolist(), mode="lines+markers", line={"color": color, "width": 4 if name == "all" else 2}, name=name, legendgroup=name, showlegend=name not in {tr.name for tr in fig.data}, hovertemplate=f"n={m.sum().item()}<br>P %{{y:.3f}}"), row=row, col=col)
            fig.add_trace(go.Scatter(x=KS, y=g[m].nanmean(0).tolist(), mode="markers", marker={"color": color, "symbol": "circle-open", "size": 10, "line": {"width": 2}}, name=f"{name} greedy", legendgroup=name, showlegend=False, hovertemplate="greedy %{y:.3f}"), row=row, col=col)
    fig.update_layout(
        title=(
            "<b>Filler uplift by task</b>  DeepSeek V4 Flash, k dots on the 'Filler:' line, worked examples with the same filler"
            "<br><span style='font-size:13px'>Lines: mean probability of writing exactly the right number and stopping (bands: 95% bootstrap over questions). Hollow markers: greedy accuracy. Equations: Olivia's 600 items, 10 shots (greedy from the 598 items of the lens runs). Facts from Greenblatt's compose_facts, 10 shots (the 1-fact addition items use only facts the model retrieves correctly with no filler); the 2-fact set is Brauer et al.'s, 5 shots; the 3- and 4-addend sets are from Greenblatt's multi_hop generator (facts below 100, 300 items each, 10 shots).</span>"
        ),
        height=1000, width=1320, margin={"t": 130}, legend={"title": "questions"}, **DARK,
    )
    fig.update_xaxes(title="number of filler dots", tickvals=KS, row=2)
    fig.update_yaxes(title="mean probability of the correct answer", range=[0, 1], col=1)
    write_dark_html(fig, "figs/facts_uplift.html")
    print(f"{green}wrote figs/facts_uplift.html{endc}")
