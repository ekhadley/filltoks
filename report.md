This project: [GitHub](https://github.com/ekhadley/filltoks) · [Google Doc](https://docs.google.com/document/d/1YWxAb-BHpC_mN2T6Bcz4gehqz7j1HlgzUFIzf1VaqK8/edit)

Follow-up to [Olivia Velte's write-up](https://docs.google.com/document/d/14aauJ6KSyBfIffi305RwyBenI7KkB6UM-uioYiic2x8/edit) ([code and data](https://github.com/orvelte/fillertokensminiproject)).

# Filler tokens on DeepSeek V4 Flash (draft)

A short white-box follow-up to Olivia Velte's black-box study of filler-token uplift on DeepSeek V4 Flash, and to Brauer et al. ([arXiv 2607.03502](https://arxiv.org/abs/2607.03502)). The question: when a model answers more accurately because meaningless tokens sit between a question and its answer, what is it doing with them?

## Setup

- **Model:** the open DeepSeek V4 Flash weights (284B parameters, 13B active, 43 layers), run locally in their native FP8/FP4 format on 2 GPUs. The reproduction in takeaway 1 used a bf16 copy on 8 GPUs.
- **Task:** Olivia's 600 two-step arithmetic questions. A few variable definitions give a number x and define y = c1·x ± k1, with distractor definitions mixed in. The question asks for c2·y ± k2, and y is never written. The multipliers c1 and c2 are 2 or 3.
- **Prompt:** 10 worked examples, then the question, then `Filler: . . .` with k dots, then `Answer:`. Every worked example carries the same filler. The model answers immediately, with no reasoning tokens.
  - The question's own turn as tokens: [figs/prompt_k10_turn.html](figs/prompt_k10_turn.html). The full prompt: [figs/prompt_k10.html](figs/prompt_k10.html).
- **Measure:** P(correct), the model's probability of writing exactly the right number and stopping. This is its expected accuracy when sampling at temperature 1.
- **Noise:** batch composition and MoE routing flips make per-question numbers jittery. Re-measuring a question at 95 instead of 100 dots moves its P(correct) with a standard deviation of 0.16. Group averages are reliable; per-question changes below about 0.5 are not.

## Takeaways

### 1. The uplift reproduces locally

The open weights show the same filler uplift as Olivia's API runs, on the same questions.

- **Overall:** with 100 dots, local greedy accuracy goes from 55% to 64% and P(correct) from 0.49 to 0.62. Through the API, greedy accuracy went from 53.5% to 63.5%.
- **Per question:** local P(correct) tracks the API's sampled pass rate (r = 0.93 without filler, 0.97 with 100 dots).
- **Same flips:** the questions that flip from wrong to right are largely the same; 55 of 64 local flips are also API flips.
- **The model used for everything else** (FP8/FP4) goes from 0.47 to 0.62.

Plot: [figs/reproduction.html](figs/reproduction.html). More detail: [figs/dose_curve.html](figs/dose_curve.html).

### 2. Different questions get very different uplift

Most questions barely move with filler, while a minority go from mostly wrong to nearly always right. The fraction of error removed is bimodal, piled up at 0 and at 1.

- With 100 dots the mean change in P(correct) is +0.16 but the median only +0.02. 93 questions rise by more than 0.5 and 6 fall by more than 0.5.
- Of the 327 questions below 0.5 without filler, 100 dots remove over half the error for 101 and over 90% for 66.
- With 10 dots almost nothing changes on average (+0.01).

Plot: [figs/uplift_hist.html](figs/uplift_hist.html). More detail: [figs/uplift_p0_vs_p100.html](figs/uplift_p0_vs_p100.html), [figs/uplift_mixture.html](figs/uplift_mixture.html).

### 3. Filler type doesn't matter much

Dots, Wikipedia text, counting and random integers give about the same uplift, and they fix mostly the same questions.

- **Same level:** all four reach about 0.60–0.64 somewhere between 100 and 500 tokens, from 0.47 with no filler.
- **Same questions:** take the questions that are wrong without filler and that a given filler fixes at 100 tokens. 73–96% of them are also fixed by 100 dots. Chance would be 27%, and re-measuring dots at 95 instead of 100 gives 88%.
- **Differences:**
  - Wikipedia text and counting help a little sooner (+4–5 pp at 10 tokens).
  - Wikipedia text dips at 100 tokens before recovering.
- **The exception is the alphabet.** One 'A B ... Z' (27 tokens) gives about two thirds of the gain of 27 dots. Repeating it to 100 tokens adds nothing (0.55, against 0.62 for 100 dots).

Plot: [figs/filler_types.html](figs/filler_types.html). More detail: [figs/dose_curve_wiki.html](figs/dose_curve_wiki.html), [figs/dose_curve_count.html](figs/dose_curve_count.html), [figs/uplift_alpha.html](figs/uplift_alpha.html).

### 4. Later filler positions don't hold better attempts

Our hypothesis was:
- the filler sharpens the intermediate results, so each later pass over the filler gets closer to the right answer;
- the answer then favors these later, better attempts.

We don't see this.

- **Method:** at each of 100 dots, we read the model's state at every layer through its output layer (the logit lens). We count a value as present at that dot if it is the top number at any layer.
- **No rise from early to late dots,** in any group. For the questions filler flips from wrong to right, dots 1–49 vs dots 50–99 give:
  - y: 0.19 vs 0.19
  - c2·y: 0.14 vs 0.14
  - the answer: 0.23 vs 0.21
- **The questions filler fixes most look the same.** These are the ones losing over half their error.
- **Later dots also don't count more:** swapping in the last half of the filler moves the answer less than swapping in the first half (takeaway 5).
- **Caveat:** the lens only sees a value when it comes out on top. A version using the value's rank among all numbers shows no early-to-late trend either. Sharpening the lens can't read would not show up.

Plot: [figs/lens_dot_position.html](figs/lens_dot_position.html). More detail: [figs/lens_dot_position_fer.html](figs/lens_dot_position_fer.html), the top panel of [figs/transplant_where.html](figs/transplant_where.html).

### 5. The model mostly copies its answer out of the filler's late layers, and early filler counts slightly more than late filler

**Method:** we run the model on a target question and replace its internal activations at the filler positions with those from a run on a donor question. The donor has different numbers and a different answer. We then measure how often the model gives the donor's answer. Numbers are given as always-right / flipped questions (93 / 57 after filtering).

- **Swapping the whole filler moves the answer to the donor's** (0.88 / 0.80). The target's own answer falls to 0.01 / 0.00.
  - This happens even on questions the model gets right with no filler.
  - A donor that differs only in the final constant, which enters only at the last step, still moves the answer (0.77 / 0.73). So the filler holds a nearly finished answer.
- **Late layers:**

  | swap only the filler's layers | moves the answer |
  |---|---|
  | ≥ 36 | 0.66 / 0.68 |
  | ≥ 38 | 0.50 / 0.59 |
  | ≥ 40 | 0.04 / 0.06 |
  | below 38 | nothing |

  So what decides the answer is in the filler's layers 36–39, where the dots are finishing the calculation. Their final layers don't matter.
- **Early vs late:**
  - Swapping the first 50 dots moves the answer more than swapping the last 50 (0.12 vs 0.03 for always-right; 0.19 vs 0.05 for flipped). The differences are +0.09 [0.05, 0.14] and +0.14 [0.07, 0.23].
  - With 75-dot sets the result is mixed: 0.34 vs 0.26 for always-right, 0.34 vs 0.40 for flipped.
- **Caveat:** in a partial swap the remaining dots still hold the target's run, so the model sees conflicting evidence.

Plots: [figs/transplant_where.html](figs/transplant_where.html) (by layer below, by dot range on top), [figs/transplant_regions.html](figs/transplant_regions.html) (which regions and which donors move the answer).

## Appendix: other questions, with preliminary results

**A1. Does the gain keep growing with more filler?**
- No. It rises to about 100 dots, stays flat to 500 (0.62 / 0.63 / 0.64 at 100 / 200 / 500), then drops a little at 1000 (0.60).
- An unexplained dip sits at 10 dots: 0.51 / 0.48 / 0.53 at 5 / 10 / 15.

Plots: [dose_curve](figs/dose_curve.html), [dose_curve_long](figs/dose_curve_long.html).

**A2. Does it matter where the filler goes?**
- Yes. 100 filler tokens placed before the question cost 1–12 pp (dots 0.35, random integers 0.38, Wikipedia 0.42, counting 0.46, against 0.47 with no filler).
- Adding 100 dots after the question on top of that recovers the full uplift (0.58–0.65), so the two effects add.

Plots: [filler_position](figs/filler_position.html), [filler_position_marg](figs/filler_position_marg.html).

**A3. Is the uplift a vote over the model's usual answers?**
- No. With filler, the model gives right answers it almost never gives without: 14 of 42 questions never answered right in 48 no-filler samples get right answers (Olivia).
- We ran 9 versions of each question, with different text before it, then chose among their answers:

  | method | accuracy |
  |---|---|
  | one plain run | 0.52 |
  | majority vote | 0.55 |
  | most confident run | 0.57 |
  | any run right | 0.73 |
  | 100 dots after the question | 0.64 |

- A test giving each dot its own random "seed" numbers was inconclusive; the seed was too weak.

Plots: [selector](figs/selector.html), [seeded](figs/seeded.html).

**A4. Is the answer spread across the filler? (preliminary)**
- No contiguous block of up to 25 dots moves the answer (at most 0.02). It takes about 75 dots to reach 0.26–0.40.
- Only contiguous blocks were tested. Brauer et al. found 12 scattered positions, picked with the lens, carrying 93% of the effect, and we never tried that kind of selection.

Plot: top panel of [transplant_where](figs/transplant_where.html).

**A5. Do filler positions need to see each other?** Partly. Mean P(correct), all questions / flipped:

| condition | all | flipped |
|---|---|---|
| no filler | 0.47 | 0.13 |
| 100 dots | 0.62 | 0.92 |
| each dot blind to the other dots | 0.35 | 0.24 |
| 4 blocks of 25 dots, each blind to the others | 0.57 | 0.71 |
| one plain filler of 25 dots | 0.54 | 0.55 |
| answer blind to the dots | 0.39 | 0.22 |

- Independent blocks of 25 dots do better than one plain filler of 25, so independent work adds something. Letting the blocks see each other adds more.
- Isolating single dots drops the model below the no-filler level overall, which is likely damage from an unnatural attention pattern.

Plots: [knockout](figs/knockout.html), [knockout_explainer](figs/knockout_explainer.html).

**A6. What does a filler position compute, and does it differ by task? (not settled)**
- **Equations:** at a dot, the values appear in chain order over depth: x at layer 24, y at 30, c2·y at 33, the answer at 37 (10 dots). Many dots carry only the start of the chain.
- **2-fact addition** ("atomic number of X plus atomic number of Y"): the two facts and the sum sit in a narrow band of layers (35–38), often at different dots.
- Both patterns also appear without filler, so this may be the same use of filler applied to differently shaped tasks rather than two uses.

Plots: [lens_k10](figs/lens_k10.html), [lens_k100](figs/lens_k100.html), [lens2_k10](figs/lens2_k10.html).

**A7. Why does the no-filler pass fail?**
- A ×3 first hop is the hard case: greedy accuracy is 0.34 / 0.21 for ×3 first, against 0.77 / 0.82 for ×2 first.
- With ×3 first, y appears about 3 layers later and c2·y rarely follows. This looks like running out of layers.
- Filler doesn't move y earlier. As a fraction of the remaining error, it removes most ×2-first errors (0.64 and 0.92) and few ×3-first errors (0.25 and 0.08).

Plots: [uplift_by_coef](figs/uplift_by_coef.html), [lens_answer_position](figs/lens_answer_position.html), [lens_rank_depth](figs/lens_rank_depth.html).

**A8. Why do some questions gain much more than others?**
- The distribution is lopsided: the mean change is +0.16 but the median only +0.02. 195 questions rise, 40 fall, and the rest barely move.
- Only 87 of the 195 risers move by more than a question's own noise.
- Among ×3-first questions, we don't know what separates risers from non-risers. In the lens they look the same until the answer appears.

Plots: [uplift_hist](figs/uplift_hist.html), [uplift_mixture](figs/uplift_mixture.html), [lens_uplift_k100](figs/lens_uplift_k100.html).

**A9. Other tasks.**
- 1-fact addition is near ceiling: 87% → 90% greedy with 100 dots.
- 2-fact addition goes 34% → 42% at 10 dots and 38% at 100, the only task where 10 dots beat 100. Its wrong answers have both facts and a slightly wrong sum.
- 3- and 4-addend sums are at the floor.

Plot: [facts_uplift](figs/facts_uplift.html).

**A10. Do small models hide an uplift subset under a zero average?** We can't tell on this task. Qwen3.6 35B-A3B is at the floor (2% greedy at every filler length), and no subset beyond about 1% of questions rises.

Plot: [uplift_hist_qwen36_35b](figs/uplift_hist_qwen36_35b.html).

**A11. Does the filler in the worked examples matter?** Probably.
- At the 'F' of 'Filler:', which comes before the question's own dots, y decodes in 11% of questions when the examples carry no filler, and in 65–69% when they carry 15 or more dots.
- How much of the uplift comes through the examples is untested.

Plots: the 'F' column of [lens_k0](figs/lens_k0.html) vs [lens_k100](figs/lens_k100.html).

## Handoff

**Where things are**
- `CLAUDE.md` is the full log: every experiment's setup, numbers and files, plus the model's and the tooling's quirks.
- Code is in flat scripts with `#%%` cells (`bench.py`, `lens.py`, `knockout.py`, `transplant.py`, `facts.py`, `qwen.py`), and all model-facing helpers are in `utils.py`.
- Results are in `results/` and figures in `figs/`. The three largest lens dumps (`lens_v4flash_packed.pt`, `lens_coef_*`, `lens_alpha_*`) are gitignored.

**Running things**
- **Locally:** every script runs on a tiny random V4 (`tiny_bridge`), and all plot cells run from `results/` without the model.
- **Real model:** a Vast.ai box with 2x RTX PRO 6000 96 GB (about $2.8/h).
  - Load with `load_bridge(MODEL_ID); fp32_routers(model)`.
  - Keep the HF hub online; the FP8 kernel is fetched from it.
  - Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Running cells:** `python cells.py script.py cell1,cell2 'TAG = "tiny"=>TAG = "v4flash_packed"' 'model = tiny_bridge()=>model = load_bridge(MODEL_ID); fp32_routers(model)'` turns on only the named cells.
- **Noise:** compare conditions within one batch, and don't trust a single question's change below about 0.5.

**Suggestions**
- **Fraction of error removed may be a better measure of what filler does** than the change in P(correct). For example, the multiplier groups gain similar amounts of P(correct), but the fraction of error removed separates them (A7). We're not sure about this; per question, it is unstable when little error is left, which is why the histograms only use questions below 0.5 without filler.
- **A smaller model could stand in if it has its own riser questions,** even with no average uplift: find the questions filler fixes and study only those. Qwen3.6 35B-A3B had none beyond about 1% on this task (A10), so this needs an easier task variant or another model.

**Next experiments**, roughly by value for cost:
1. **Swap only the dots where the lens shows the answer** (a new scope in `transplant.py`). This settles whether a small scattered set carries the answer (A4), Brauer's version of the test.
2. **How uplift depends on no-filler accuracy.** Quantify how the uplift, or the share of questions that rise, depends on no-filler accuracy P0, e.g. binned by log P0. The dose results give exact P(correct) down to very small values, so this can start offline. The question is whether filler can only help where the model already has non-trivial no-filler accuracy, or whether it opens a qualitatively different regime. Starting points: 14 of 42 questions never answered right in 48 no-filler samples get right answers with filler (Olivia); the risers fall into clusters starting from P0 of about 0.15 and 0.56, plus a partial one from about 0.03 (`uplift_mixture`); and Qwen3.6 35B-A3B, at the floor, gets nothing (A10).
3. **Filler in the worked examples only, and in the question only** (`bench.py`, `prompt_ids` with per-turn fillers). Black box, under an hour of GPU time (A11).
4. **Count distinct complete y values per question** from the saved lens results, which hold the top number per (layer, position). This needs no GPU. It tests rival attempts against one spread-out computation.
5. **A two-fact-shaped arithmetic task**, two independent products and a sum. It separates task shape from fact recall (A6).
6. **Block the question's exact attention window vs its compressed summaries** (the knockout helpers in `utils.py`), the half of the window experiment that was never run.
7. **What separates ×3-first risers from non-risers** (A8). This works offline, from the dose results and the questions' numbers.
