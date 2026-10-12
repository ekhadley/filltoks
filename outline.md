# Report outline: filler tokens on DeepSeek V4 Flash

"doc img N" is the figure's number in the current Google Doc; plots not yet in the doc are marked (not in doc).

## Takeaways

1. **The uplift reproduces locally.** The open DeepSeek V4 Flash weights, run locally, show the same filler uplift as the API: mean P(correct) 0.47 → 0.62 with 100 dots, and the same questions flip.
   - Plots: accuracy bars, API vs local (doc img 2); `uplift_p0_vs_p100` (doc img 11).
2. **Filler type doesn't matter much.** Dots, Wikipedia text, counting and random integers give about the same uplift and fix mostly the same questions.
   - Plots: `dose_curve_wiki` (doc img 20), `dose_curve_count` (doc img 21), `filler_position` (doc img 22).
3. **Our hypothesis that later filler positions hold better attempts is not supported.** We expected the filler to sharpen the intermediate results, so that later positions hold better attempts and the answer favors them. Later dots don't hold better attempts, and they don't count more.
   - Plots: `lens_dot_position` or `lens_dot_position_fer` (not in doc); `transplant_where`, top panel (not in doc).
4. **The model mostly copies its answer out of the filler's late layers, and early filler matters slightly more than late filler.**
   - Plots: `transplant_where` (not in doc), `transplant_regions` (not in doc).

## Preliminary results

- The answer is spread across the filler: no contiguous block of up to a quarter of the dots carries it (`transplant_where`, top panel). Only contiguous blocks were tested, not scattered positions picked by the lens, which is how Brauer et al. found 12 positions carrying 93% of the effect.

## Not settled

- Different-looking filler computations for different tasks: on the equations each dot holds the chain spread over layers; on 2-fact addition the facts and the sum sit in the same few layers across different dots (`lens_k10`, `lens2_k10`). This may follow from the tasks' own no-filler computations rather than from two different uses of filler. Wording open.
