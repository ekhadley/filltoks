#%% Setup: data, model, tokenizer, and the knockout conditions

import html
import json
import math

import torch as t
from transformers import FineGrainedFP8Config

from mechtools import *

from utils import MODEL_ID, EOS, tiny_bridge, fp32_routers, prompt_ids, regions, answer_ids, prefill, tail_logits, knockout, blocked_rows, seeded_prompt, seed_cond

t.set_grad_enabled(False)
items = [json.loads(l) for l in open("data/eval.jsonl")]
shots = [json.loads(l) for l in open("data/fewshot.jsonl")]

TAG = "tiny"
model = tiny_bridge()
# TAG = "v4flash"
# model = load_bridge(MODEL_ID)  # packed FP8 + FP4 experts, ~160 GB; needs `uv add kernels`
# fp32_routers(model)  # route in fp32 like DeepSeek's reference inference (HF routes in bf16)
tok = model.tokenizer

# Attention knockouts over the item's own filler. Each marks m[i, j] = True where tail query i may not attend to tail key j, at every layer, in the
# exact window and in the pooled entries covering j. The dots keep everything before the filler (prompt, question, 'Filler:' label) and the
# 'Answer:' tokens and the answer position keep every dot, unless stated. A dot always sees itself.
def dots_see(m, reg, see):  # earlier dot b hidden from dot a unless see(a, b)
    f, kk = reg["filler"][0], len(reg["filler"])
    a, b = t.arange(kk)[:, None], t.arange(kk)[None, :]
    m[f:f + kk, f:f + kk] |= (b < a) & ~see(a, b)
def dots_hidden_after(m, reg):  # every position after the filler ('Answer:', the answer) blind to the dots
    m[reg["filler"][-1] + 1:, reg["filler"]] = True
def alone_no_label(m, reg):  # each dot sees only the prompt, and not the 'Filler:' label either
    dots_see(m, reg, lambda a, b: a == b)
    m[reg["filler"][0]:reg["filler"][-1] + 1, reg["label"]] = True
CONDS = {  # name: (marker, keep an HCA entry that pools blocked dots together with unblocked tokens)
    "full": (lambda m, reg: None, False),
    "alone": (lambda m, reg: dots_see(m, reg, lambda a, b: a == b), False),
    "alone_keep_hca": (lambda m, reg: dots_see(m, reg, lambda a, b: a == b), True),
    "alone_no_label": (alone_no_label, False),
    **{f"blocks_{w}": (lambda m, reg, w=w: dots_see(m, reg, lambda a, b: a // w == b // w), False) for w in [4, 8, 25]},
    "sliding_8": (lambda m, reg: dots_see(m, reg, lambda a, b: a - b <= 8), False),
    "halves": (lambda m, reg: dots_see(m, reg, lambda a, b: (a < 50) | (b >= 50)), False),
    "answer_blind": (dots_hidden_after, False),
}

#%% Gates on one question, every comparison within one batch: blocking nothing, or only future keys, leaves the logits bit-identical; under each knockout, replacing the dots hidden from a probe dot by commas moves the probe's stream at every layer by at most max_ratio of what the same replacement does with nothing blocked, and replacing its visible dots moves it more; likewise for the answer position under answer_blind

check_gates = True
if check_gates:
    k, probe = 100, 60  # the dot whose stream is compared
    max_ratio = 0.05  # fp32 tiny model: about 1e-6
    # max_ratio = 0.3  # bf16 real model: batch composition changes rounding and MoE routing, so even changes a position cannot see move its stream by a few percent
    COMMA = tok(" ,", add_special_tokens=False).input_ids[0]
    prefix, tail = prompt_ids(tok, shots, items[0], k)
    reg = regions(tok, tail, k)
    f = reg["filler"]
    cache = prefill(model, prefix)
    all_dots = set(range(len(f)))
    mixed = set(range(max(0, (len(prefix) + len(tail)) // 128 * 128 - len(prefix) - f[0])))  # dots before the end of the last complete 128-token block: pooled into an HCA entry together with the question
    def edited(keep: set[int]) -> list[int]:  # the tail with every dot replaced by a comma, except those in `keep` and the last (merged with '\n\n')
        return [COMMA if p in f[:-1] and f.index(p) not in keep else tid for p, tid in enumerate(tail)]
    states = {}
    def grab(act, hook):
        states[hook.layer()] = act.float().cpu()
    def run(cond, rows, keep_hca=False):
        with knockout(blocked_rows([reg] * len(rows), len(tail), cond), len(prefix), keep_hca):
            lp = tail_logits(model, cache, rows, [(f"blocks.{l}.hook_out", grab) for l in range(model.cfg.n_layers)])[:, -1].log_softmax(-1)
        return lp, t.stack([states[l] for l in range(model.cfg.n_layers)], 1)  # [B, layers, T, 4, D]
    def moved(st, row, pos) -> float:  # relative change (L2) of row `row`'s stream at tail position `pos` against row 0, worst layer
        return ((st[row, :, pos] - st[0, :, pos]).flatten(1).norm(dim=1) / st[0, :, pos].flatten(1).norm(dim=1)).max().item()
    plain = tail_logits(model, cache, [tail, edited(set())])[:, -1].log_softmax(-1)
    assert t.equal(run(CONDS["full"][0], [tail, edited(set())])[0], plain), "blocking nothing changed the logits"
    assert t.equal(run(lambda m, reg: m.__ior__(t.ones_like(m).triu(1)), [tail, edited(set())])[0], plain), "blocking future keys changed the logits"
    checks = {  # condition: (dots hidden from the probe dot, dots visible to it)
        "alone": (all_dots - {probe}, set()),
        "alone_keep_hca": (all_dots - {probe} - mixed, set()),
        "alone_no_label": (all_dots - {probe}, set()),
        "blocks_8": (all_dots - set(range(56, 64)), set(range(56, 60))),
        "sliding_8": (set(range(max(0, probe - 8 * model.cfg.n_layers))), set(range(52, 60))),  # a hidden dot still reaches the probe through intermediate dots, one hop of 8 per layer
        "halves": (set(range(50)), set(range(50, 60))),
    }
    print(f"{gray}{len(mixed)} dots share a pooled HCA entry with the question{endc}")
    for name, (hidden, visible) in checks.items():
        rows = [tail, edited(all_dots - hidden)] + ([edited(all_dots - visible)] if visible else [])
        _, st = run(CONDS[name][0], rows, CONDS[name][1])
        _, st_full = run(CONDS["full"][0], rows)
        d_hidden, d_full = moved(st, 1, f[probe]), moved(st_full, 1, f[probe])
        d_vis = moved(st, 2, f[probe]) if visible else None
        print(f"{cyan}{name}: replacing the {len(hidden)} dots hidden from dot {probe} moves its stream by {d_hidden:.1e} (relative), {d_full:.1e} with nothing blocked" + (f"; replacing the {len(visible)} visible dots moves it by {d_vis:.1e}" if visible else "") + endc)
        assert d_hidden <= max_ratio * d_full and (d_vis is None or d_vis > d_hidden), (name, d_hidden, d_full, d_vis)
    lp, st = run(CONDS["answer_blind"][0], [tail, edited(set())])
    lp_full, st_full = run(CONDS["full"][0], [tail, edited(set())])
    d_blind, d_full = moved(st, 1, len(tail) - 1), moved(st_full, 1, len(tail) - 1)
    print(f"{cyan}answer_blind: replacing every dot moves the answer position's stream by {d_blind:.1e} (relative) and its log-probs by {(lp[1] - lp[0]).abs().max():.1e}; with nothing blocked {d_full:.1e} and {(lp_full[1] - lp_full[0]).abs().max():.1e}{endc}")
    assert d_blind <= max_ratio * d_full, (d_blind, d_full)
    print(f"{green}gates passed{endc}")

#%% Knockout sweep: exact P(correct answer + EOS) and the greedy first token for every item under each knockout at k dots, plus unmodified runs at a few smaller filler lengths for comparison with the block conditions; saves results/knockout_{TAG}.json

run_knockout = True
if run_knockout:
    k = 100
    ref_ks = [0, 4, 8, 25]  # unmodified runs with this many dots (the dose sweep has 0, 5, ..., 100)
    n_items = 600
    # n_items = 32
    batch_size = 16
    records = []
    for kk in ref_ks + [k]:
        pts = [prompt_ids(tok, shots, it, kk) for it in items[:n_items]]
        assert all(prefix == pts[0][0] for prefix, _ in pts)
        prefix = pts[0][0]
        cache = prefill(model, prefix)
        regs = [regions(tok, tail, kk) for _, tail in pts]
        answers = [answer_ids(tok, it["answer"]) for it in items[:n_items]]
        for name in ["full"] if kk != k else CONDS:
            cond, keep_hca = CONDS[name]
            for i in pbar(range(0, n_items, batch_size), desc=f"k={kk} {name}"):
                sl = slice(i, i + batch_size)
                rows = [tail + a for (_, tail), a in zip(pts[sl], answers[sl])]
                n_last = max(map(len, answers[sl])) + 1
                with knockout(blocked_rows(regs[sl], max(map(len, rows)), cond), len(prefix), keep_hca):
                    logits = tail_logits(model, cache, rows, n_last=n_last)
                logp = logits.log_softmax(-1)
                for j, (it, a) in enumerate(zip(items[sl], answers[sl])):
                    first = n_last - len(a) - 1
                    records.append({"idx": it["idx"], "k": kk, "cond": name, "logp": logp[j, range(first, n_last), a + [EOS]].sum().item(), "top1": (logits[j, first].argmax() == a[0]).item()})
            done = [r for r in records if r["k"] == kk and r["cond"] == name]
            print(f"{cyan}k={kk} {name}: mean P(correct) {sum(math.exp(r['logp']) for r in done) / len(done):.3f}, greedy first token right {sum(r['top1'] for r in done) / len(done):.3f}{endc}")
            json.dump(records, open(f"results/knockout_{TAG}.json", "w"))
    print(f"{green}saved results/knockout_{TAG}.json{endc}")

#%% Explainer for readers without context: what the knockout experiment does to attention. A toy prompt with 24 dots, the real blocked_rows masks for each condition (query x key, with the causal future and the knocked-out keys marked), the key axis a V4 query sees (exact window and pooled entries) and how a blocked key removes the pooled entries that cover it; written to figs/knockout_explainer.html

plot_explainer = True
if plot_explainer:
    kk = 24
    toks = ["<User>"] + ["x = 66", "y = 29", "z = 3x + 12", "w = 3y - 50", "v = 3x + 43"] + ["Question:", "What is", "2z - 38", "?"] + ["F", "iller", ":"] + ["."] * kk + ["Answer", ":", "<Asst>"] + ["</think>"]
    T = len(toks)
    reg = {"q_defs": list(range(1, 6)), "q_line": list(range(6, 10)), "label": list(range(10, 13)), "filler": list(range(13, 13 + kk)), "post": list(range(13 + kk, 16 + kk)), "last": [T - 1]}
    toy = {  # the real conditions, with halves at 12/12 for 24 dots
        "full": CONDS["full"], "alone": CONDS["alone"], "alone, label hidden": CONDS["alone_no_label"], "blocks of 8": CONDS["blocks_8"], "sliding 8": CONDS["sliding_8"],
        "halves": (lambda m, reg: dots_see(m, reg, lambda a, b: (a < kk // 2) | (b >= kk // 2)), False), "answer blind": CONDS["answer_blind"],
    }
    future = t.ones(T, T, dtype=t.bool).triu(1)
    fig = make_subplots(rows=2, cols=4, subplot_titles=list(toy), horizontal_spacing=0.03, vertical_spacing=0.1)
    bounds = [reg[r][0] - 0.5 for r in ["q_line", "label", "filler", "post", "last"]]
    centers = {"definitions": 3, "question": 7.5, "'Filler:'": 11, f"{kk} dots": 12.5 + kk / 2, "'Answer:'": 14 + kk, "answer": T - 1}
    for n, (name, (cond, _)) in enumerate(toy.items()):
        m = blocked_rows([reg], T, cond)[0]
        z = t.zeros(T, T) + future * 1 + (m & ~future) * 2  # 0 visible, 1 future (never visible), 2 knocked out
        row, col = n // 4 + 1, n % 4 + 1
        fig.add_trace(go.Heatmap(z=z.tolist(), colorscale=[[0, "#4a4a46"], [0.5, "#1a1a19"], [1, SERIES[1]]], zmin=0, zmax=2, showscale=False, hovertemplate="query %{y} (%{customdata[0]})<br>key %{x} (%{customdata[1]})<br>%{text}<extra></extra>", text=[["visible" if v == 0 else "later token" if v == 1 else "knocked out" for v in r] for r in z.tolist()], customdata=[[(toks[i], toks[j]) for j in range(T)] for i in range(T)]), row=row, col=col)
        for b in bounds:
            fig.add_vline(x=b, line={"color": "#c3c2b7", "width": 0.6}, row=row, col=col)
            fig.add_hline(y=b, line={"color": "#c3c2b7", "width": 0.6}, row=row, col=col)
        fig.update_xaxes(tickvals=list(centers.values()), ticktext=list(centers), tickangle=45, tickfont={"size": 9}, row=row, col=col)
        fig.update_yaxes(autorange="reversed", tickvals=list(centers.values()), ticktext=list(centers) if col == 1 else [""] * len(centers), tickfont={"size": 9}, row=row, col=col)
    fig.update_layout(height=820, width=1500, margin={"t": 60, "b": 80}, **DARK)
    # the key axis for one dot under 'alone': exact-window keys, then the pooled entries, and which entries a blocked key takes down
    past, rate = 1850, 4  # the k=100 prefix length and the CSA rate; the toy tail starts at absolute position 1850
    q = reg["filler"][16]
    m = blocked_rows([reg], T, CONDS["alone"][0])[0]
    n_ent = (past + T) // rate
    lo = [(rate * (e - 1) - past) for e in range(n_ent)]
    hi = [(rate * (e + 1) - past) for e in range(n_ent)]
    ents = [(e, max(lo[e], 0), min(hi[e], T)) for e in range(n_ent) if hi[e] > 0 and lo[e] < T]  # entries that cover tail positions
    cell_w, x0, y_keys, y_ent = 22, 40, 150, 70
    def rect(x, y, w, h, fill, title=""):
        return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="#1a1a19" stroke-width="1"><title>{html.escape(title)}</title></rect>'
    svg = [f'<svg viewBox="0 0 {x0 + cell_w * T + 40} 230" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:1100px;font-family:sans-serif;font-size:11px">']
    svg.append(f'<text x="{x0}" y="{y_keys - 8}" fill="#c3c2b7">tail keys as dot {q - reg["filler"][0]} sees them under "alone" (it sits at the outlined cell; the cached prompt rows to its left are not drawn)</text>')
    for j in range(T):
        fill = "#1a1a19" if j > q else SERIES[1] if m[q, j] else "#4a4a46"
        svg.append(rect(x0 + cell_w * j, y_keys, cell_w, 22, fill, f"{j}: {toks[j]}"))
        svg.append(f'<text x="{x0 + cell_w * j + cell_w / 2}" y="{y_keys + 36}" fill="#c3c2b7" text-anchor="middle" font-size="9">{html.escape(toks[j] if len(toks[j]) <= 2 else toks[j][:1])}</text>')
    svg.append(f'<rect x="{x0 + cell_w * q}" y="{y_keys}" width="{cell_w}" height="22" fill="none" stroke="#e8e6dc" stroke-width="2"/>')
    svg.append(f'<text x="{x0}" y="{y_ent - 8}" fill="#c3c2b7">pooled CSA entries over the same keys (each pools 8 tokens: its own 4-token window and the one before; the entry is removed when any key it covers is blocked)</text>')
    for e, a, b in ents:
        blocked = bool(m[q, a:b].any())
        fill = "#1a1a19" if b - 1 > q else SERIES[1] if blocked else "#4a4a46"  # an entry exists only once its window is complete
        y = y_ent + (e % 2) * 14
        svg.append(rect(x0 + cell_w * a + 1, y, cell_w * (b - a) - 2, 12, fill, f"entry {e}: tail positions {a} to {b - 1}" + (" (removed: covers a blocked dot)" if blocked else "")))
    svg.append(f'<text x="{x0}" y="{y_ent + 50}" fill="#8a8982" font-size="10">Grey: visible. Orange: knocked out. Black: later than the query (never visible). Entries are drawn on two rows because neighbouring entries overlap by 4 tokens.</text>')
    svg.append("</svg>")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Attention knockouts on filler dots</title>
<style>html, body {{ background: #1a1a19; color: #c3c2b7; font-family: sans-serif; margin: 0; }} main {{ max-width: 1500px; margin: 0 auto; padding: 24px; }} h1 {{ color: #e8e6dc; font-size: 22px; }} h2 {{ color: #e8e6dc; font-size: 17px; margin-top: 28px; }} p, li {{ font-size: 14px; line-height: 1.5; max-width: 1100px; }} code {{ color: #e8e6dc; }} table {{ border-collapse: collapse; font-size: 13px; }} td, th {{ padding: 3px 12px; border-bottom: 1px solid #3a3a37; text-align: left; }} .o {{ color: {SERIES[1]}; }}</style></head><body><main>
<h1>What the attention knockouts do</h1>
<p>DeepSeek V4 Flash answers two-step arithmetic questions more often when the prompt carries a line of filler dots between the question and <code>Answer:</code> (0.47 → 0.62 mean probability of the right answer with 100 dots). The knockout experiment asks whether the dots need to <i>see each other</i> for that, or whether each dot helps on its own. It runs the same prompt with the attention mask edited so that chosen positions cannot attend to chosen other positions, at every layer, and reads the probability of the right answer at the last position.</p>
<h2>The prompt</h2>
<p>System prompt and 10 worked examples (about 1,850 tokens, computed once and cached), then the item: five variable definitions, the question, the label <code>Filler:</code>, k dots, <code>Answer:</code>, and the position where the answer is read. Only the item part is edited; the toy below shortens it to 24 dots and one-token definitions.</p>
<h2>What a position can attend to in this model</h2>
<p>Every layer has two branches. The exact branch sees the last 128 tokens one by one. The long-range branch sees <i>pooled entries</i>: in half the layers one entry per 4 tokens (each entry pools its own 4-token window and the one before, and an indexer keeps the 512 most relevant entries), in the other half one entry per 128 tokens. A knockout therefore has to block a key in the exact window <i>and</i> remove every pooled entry that covers it, otherwise the blocked token leaks through its entry. Removing an entry also hides the unblocked tokens it covers; with 100 dots that costs a few question tokens for about half the items, which the <code>alone_keep_hca</code> variant restores (same result).</p>
{"".join(svg)}
<h2>The conditions as masks</h2>
<p>Rows are query positions, columns are key positions, both in prompt order. Grey cells are visible, black cells are later tokens (never visible), <span class="o">orange</span> cells are knocked out. A dot always sees itself and everything before the filler (the rows stay grey left of the dots), and the <code>Answer:</code> tokens and the answer position see every dot unless stated. Hover a cell for the two tokens.</p>
<ul><li><b>full</b>: nothing blocked, the ordinary prompt.</li><li><b>alone</b>: a dot sees no other dot, so each dot is an independent computation and only the answer position can combine them.</li><li><b>alone, label hidden</b>: as alone, and the dots cannot see the <code>Filler:</code> label either.</li><li><b>blocks of n</b> (4, 8, 25 in the real run): a dot sees only the dots of its own block of n; compared with a prompt that has just n dots.</li><li><b>sliding 8</b>: a dot sees the 8 dots before it; information can still travel farther by hopping from dot to dot, one hop per layer.</li><li><b>halves</b>: the second half of the dots cannot see the first half.</li><li><b>answer blind</b>: the dots see everything as usual, but <code>Answer:</code> and the answer position cannot see any dot.</li></ul>
{fig.to_html(full_html=False, include_plotlyjs="cdn")}
<h2>What came out (100 dots, 600 questions, mean probability of the right answer)</h2>
<table><tr><th>condition</th><th>all questions</th><th>the 79 questions the dots flip to right</th><th>the 229 right either way</th></tr>
<tr><td>no filler</td><td>0.47</td><td>0.13</td><td>0.94</td></tr><tr><td>100 dots, nothing blocked</td><td>0.62</td><td>0.92</td><td>0.98</td></tr>
<tr><td>alone</td><td>0.35</td><td>0.24</td><td>0.71</td></tr><tr><td>alone, label hidden too</td><td>0.18</td><td>0.07</td><td>0.39</td></tr>
<tr><td>blocks of 4 / 8 / 25</td><td>0.43 / 0.48 / 0.57</td><td>0.33 / 0.46 / 0.71</td><td>0.83 / 0.88 / 0.96</td></tr><tr><td>4 / 8 / 25 dots with nothing blocked</td><td>0.54 / 0.53 / 0.54</td><td>0.41 / 0.42 / 0.55</td><td>0.95 / 0.93 / 0.94</td></tr>
<tr><td>sliding 8</td><td>0.57</td><td>0.74</td><td>0.96</td></tr><tr><td>halves</td><td>0.60</td><td>0.83</td><td>0.98</td></tr><tr><td>answer blind</td><td>0.39</td><td>0.22</td><td>0.78</td></tr></table>
<p>Dots that cannot see each other are worse than no filler at all, and blocks of n dots are worse than n dots alone, so the dots are not independent attempts: the gain needs the dots to read each other, and a chain of short hops (sliding 8) recovers most of it. Blinding the answer to the dots removes the gain and more, so the answer position reads the dots directly rather than through the question tokens.</p>
</main></body></html>"""
    open("figs/knockout_explainer.html", "w").write(page)
    print(f"{green}wrote figs/knockout_explainer.html{endc}")

#%% Per-dot seeds: gates on one question. With every block hidden, changing the blocks moves the logits by at most max_ratio of what it does with the blocks seen; with each dot seeing its own block only, changing the other blocks moves dot i's stream (worst layer, relative) by at most max_ratio of what changing its own block does; block 0 of the 'before' layout starts at an absolute multiple of 8. (Not bit-exact even in fp32: the MoE kernels reduce over whichever tokens share an expert, so unrelated positions move by about 1e-6.)

check_seed_gates = True
if check_seed_gates:
    k, probe = 50, 30
    max_ratio = 0.01  # fp32 tiny model: about 1e-5
    # max_ratio = 0.3  # bf16 real model: batch composition changes rounding and MoE routing
    states = {}
    def grab(act, hook):
        states[hook.layer()] = act.float().cpu()
    for placement in ["before", "inline"]:
        prefix, tail, reg = seeded_prompt(tok, shots, items[0], k, placement)
        assert placement == "inline" or (len(prefix) + reg["blocks"][0][0]) % 8 == 0
        assert (len(prefix) + len(tail)) // 4 <= model.original_model.config.index_topk, "the CSA indexer would drop entries"
        cache = prefill(model, prefix)
        rng = t.Generator().manual_seed(1)
        def swapped(which: list[int]) -> list[int]:  # the tail with every integer token of the listed blocks replaced by another three-digit number
            pos = {p for i in which for p in reg["blocks"][i] if tok.decode([tail[p]]).strip().isdigit()}
            return [tok(str(t.randint(100, 1000, (1,), generator=rng).item()), add_special_tokens=False).input_ids[0] if p in pos else tid for p, tid in enumerate(tail)]
        def run(see, alone, rows):
            with knockout(blocked_rows([reg] * len(rows), len(tail), seed_cond(see, alone)), len(prefix)):
                lp = tail_logits(model, cache, rows, [(f"blocks.{l}.hook_out", grab) for l in range(model.cfg.n_layers)])[:, -1].log_softmax(-1)
            return lp, t.stack([states[l] for l in range(model.cfg.n_layers)], 1)
        def moved(st, pos) -> float:
            return ((st[1, :, pos] - st[0, :, pos]).flatten(1).norm(dim=1) / st[0, :, pos].flatten(1).norm(dim=1)).max().item()
        others = [i for i in range(k) if i != probe]
        lp, _ = run("none", False, [tail, swapped(list(range(k)))])
        lp_seen, _ = run("all", False, [tail, swapped(list(range(k)))])
        d_hidden, d_seen = (lp[0] - lp[1]).abs().max().item(), (lp_seen[0] - lp_seen[1]).abs().max().item()
        _, st = run("own", True, [tail, swapped(others)])
        d_others, d_ans = moved(st, reg["filler"][probe]), moved(st, len(tail) - 1)
        _, st = run("own", True, [tail, swapped([probe])])
        d_own = moved(st, reg["filler"][probe])
        print(f"{cyan}{placement}: swapping every block moves the answer log-probs by {d_hidden:.1e} with the blocks hidden, {d_seen:.1e} seen; own block only, dots alone: swapping the other {k - 1} blocks moves dot {probe}'s stream by {d_others:.1e} (relative) and the answer position's by {d_ans:.1e}; swapping its own block moves it by {d_own:.1e}{endc}")
        assert d_hidden <= max_ratio * d_seen and d_others <= max_ratio * d_own and d_ans > d_others, (d_hidden, d_seen, d_others, d_own, d_ans)
    print(f"{green}seed gates passed{endc}")

#%% Per-dot seeds sweep: every question at k dots with one block of 4 random three-digit integers per dot, placed above the definitions ('before') or just before each dot ('inline'), under blocks hidden from everything / each dot seeing only its own block / blocks seen by everything, with the dots seeing each other or alone; plus plain dots, full and alone. Exact P(correct answer + EOS) and the greedy first token; saves results/seeded_{TAG}.json

run_seeded = True
if run_seeded:
    k, n_items, batch_size = 50, 600, 16
    # n_items = 32
    SEE = {"hidden": "none", "own": "own", "seen": "all"}
    records = []
    for placement in ["plain", "before", "inline"]:
        pts = [seeded_prompt(tok, shots, it, k, placement) for it in items[:n_items]]
        prefix = pts[0][0]
        assert all(p == prefix for p, _, _ in pts) and (len(prefix) + max(len(tl) for _, tl, _ in pts)) // 4 <= model.original_model.config.index_topk
        cache = prefill(model, prefix)
        answers = [answer_ids(tok, it["answer"]) for it in items[:n_items]]
        for see in (["hidden"] if placement == "plain" else SEE):
            for alone in [False, True]:
                name = f"{placement}_{see}_{'alone' if alone else 'full'}"
                for i in pbar(range(0, n_items, batch_size), desc=name):
                    sl = slice(i, i + batch_size)
                    rows = [tail + a for (_, tail, _), a in zip(pts[sl], answers[sl])]
                    n_last = max(map(len, answers[sl])) + 1
                    with knockout(blocked_rows([reg for _, _, reg in pts[sl]], max(map(len, rows)), seed_cond(SEE[see], alone)), len(prefix)):
                        logits = tail_logits(model, cache, rows, n_last=n_last)
                    logp = logits.log_softmax(-1)
                    for j, (it, a) in enumerate(zip(items[sl], answers[sl])):
                        first = n_last - len(a) - 1
                        records.append({"idx": it["idx"], "placement": placement, "see": see, "alone": alone, "logp": logp[j, range(first, n_last), a + [EOS]].sum().item(), "top1": (logits[j, first].argmax() == a[0]).item()})
                done = records[-n_items:]
                print(f"{cyan}{name}: mean P(correct) {sum(math.exp(r['logp']) for r in done) / len(done):.3f}, greedy first token right {sum(r['top1'] for r in done) / len(done):.3f}{endc}")
                json.dump(records, open(f"results/seeded_{TAG}.json", "w"))
    print(f"{green}saved results/seeded_{TAG}.json{endc}")

#%% Figure for readers without context: mean P(correct) under each knockout, for all questions, the questions the filler flips, and the questions right either way (groups from the dose sweep); written to figs/knockout.html

plot_knockout = True
if plot_knockout:
    n_boot = 2000
    dose_file = f"results/dose_{TAG}.json"
    # dose_file = "results/dose_v4flash_packed.json"  # groups from the real model when plotting tiny-model records
    recs = json.load(open(f"results/knockout_{TAG}.json"))
    dose = {(r["idx"], r["k"]): math.exp(r["logp"]) for r in json.load(open(dose_file))}
    idxs = sorted({r["idx"] for r in recs})
    groups = {
        "all questions": idxs,
        "questions the filler flips (P(correct) < 0.3 with no filler, ≥ 0.6 with 100 dots)": [i for i in idxs if dose[(i, 0)] < 0.3 and dose[(i, 100)] >= 0.6],
        "questions right either way (P(correct) > 0.7 with no filler)": [i for i in idxs if dose[(i, 0)] > 0.7],
    }
    bars = [  # (dots, condition, label, color)
        (0, "full", "no filler", "#888888"),
        (100, "full", "100 dots,<br>unmodified", "#e8e6dc"),
        (100, "alone", "each dot sees<br>only the prompt", SERIES[0]),
        (100, "alone_keep_hca", "same, lenient rule<br>for pooled entries", SERIES[0]),
        (100, "alone_no_label", "each dot sees only<br>the prompt, 'Filler:'<br>label hidden too", SERIES[0]),
        (100, "blocks_4", "dots in blocks of 4,<br>each block sees<br>only itself", SERIES[1]),
        (4, "full", "4 dots,<br>unmodified", "#888888"),
        (100, "blocks_8", "blocks of 8", SERIES[1]),
        (8, "full", "8 dots,<br>unmodified", "#888888"),
        (100, "blocks_25", "blocks of 25", SERIES[1]),
        (25, "full", "25 dots,<br>unmodified", "#888888"),
        (100, "sliding_8", "each dot sees the<br>8 dots before it", SERIES[2]),
        (100, "halves", "last 50 dots cannot<br>see the first 50", SERIES[2]),
        (100, "answer_blind", "'Answer:' and the<br>answer position<br>cannot see the dots", SERIES[3]),
    ]
    p = {(r["idx"], r["k"], r["cond"]): math.exp(r["logp"]) for r in recs}
    def stat(idx: list[int], kk: int, cond: str) -> tuple[float, float, float]:  # mean P(correct) over questions with a 95% bootstrap interval
        assert idx, (kk, cond)
        v = t.tensor([p[(i, kk, cond)] for i in idx])
        boot = v[t.randint(0, len(v), (n_boot, len(v)), generator=t.Generator().manual_seed(0))].mean(1)
        return v.mean().item(), boot.quantile(0.025).item(), boot.quantile(0.975).item()
    fig = make_subplots(rows=len(groups), cols=1, subplot_titles=[f"{g} (n={len(ix)})" for g, ix in groups.items()], vertical_spacing=0.1)
    for row, ix in enumerate(groups.values(), 1):
        m, lo, hi = zip(*[stat(ix, kk, c) for kk, c, _, _ in bars])
        err = {"type": "data", "symmetric": False, "array": [h - v for v, h in zip(m, hi)], "arrayminus": [v - l for v, l in zip(m, lo)]}
        fig.add_trace(go.Bar(x=[b[2] for b in bars], y=m, marker_color=[b[3] for b in bars], error_y=err, showlegend=False), row=row, col=1)
        fig.add_trace(go.Scatter(x=[b[2] for b in bars], y=[h + 0.05 for h in hi], text=[f"{v:.2f}" for v in m], mode="text", textfont={"size": 13}, showlegend=False, hoverinfo="skip"), row=row, col=1)
    a = {(kk, c): stat(idxs, kk, c)[0] for kk, c, _, _ in bars}
    fig.update_layout(
        title={"y": 0.985, "yanchor": "top", "text": (
            f"<b>When each filler dot can attend only to the prompt and not to the other dots, the correct answer has probability {a[(100, 'alone')]:.2f}, against {a[(100, 'full')]:.2f} with the dots unmodified and {a[(0, 'full')]:.2f} with no filler</b>"
            f"<br>Dots in blocks of 4 / 8 / 25 that see only themselves: {a[(100, 'blocks_4')]:.2f} / {a[(100, 'blocks_8')]:.2f} / {a[(100, 'blocks_25')]:.2f}; just 4 / 8 / 25 dots: {a[(4, 'full')]:.2f} / {a[(8, 'full')]:.2f} / {a[(25, 'full')]:.2f}; 'Answer:' blind to the dots: {a[(100, 'answer_blind')]:.2f}.  (DeepSeek V4 Flash, {len(idxs)} questions, 100 dots)"
            "<br><span style='font-size:13px'>Task: two-step arithmetic in words. A number x is given, y = c₁·x ± k₁ is defined, and the question asks for c₂·y ± k₂. 100 filler dots sit between the question and 'Answer:'; the prompt's 10 worked examples carry the same filler.</span>"
            "<br><span style='font-size:13px'>Knockout: at every layer, the named positions are prevented from attending to the named other positions (their attention there is zeroed, the rest renormalized); everything else runs as usual. 'Prompt' = instructions, worked examples,</span>"
            "<br><span style='font-size:13px'>the definitions, the question and the 'Filler:' label. A dot always sees itself. Unless stated, 'Answer:' and the answer position see every dot. Pooled long-range entries that cover a hidden dot are hidden too (lenient rule: kept if they also cover other tokens).</span>"
            "<br><span style='font-size:13px'>Probability of the correct answer: the model's probability of writing exactly the right number and stopping. Error bars: 95% intervals from resampling questions. Grey bars: unmodified runs for comparison (their worked examples carry that many dots).</span>"
        )},
        height=1500,
        width=1900,
        margin={"t": 245, "b": 120},
        **DARK,
    )
    fig.update_yaxes(title_text="mean probability of the correct answer", range=[0, 1.08])
    fig.update_xaxes(tickangle=0, tickfont={"size": 11})
    write_dark_html(fig, "figs/knockout.html")
    print(f"{green}wrote figs/knockout.html{endc}")
    for g, ix in groups.items():
        print(f"{g} (n={len(ix)}): " + "  ".join(f"{c}@{kk}={stat(ix, kk, c)[0]:.2f}" for kk, c, _, _ in bars))
