#!/bin/bash
# alphabet filler on the packed model: bench (greedy + P(correct)) then the lens at 27 and 100 positions
cd ~/filltoks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
SETUP='TAG = "tiny"=>TAG = "v4flash_packed"'
MODEL='model = tiny_bridge()=>model = load_bridge(MODEL_ID); fp32_routers(model)'
$PY cells.py bench.py run_alpha "$SETUP" "$MODEL" && echo BENCH_DONE
$PY cells.py lens.py run_lens_all "$SETUP" "$MODEL" '    kind = "dots"
    # kind = "alpha"=>    kind = "alpha"
    # kind = "dots"' 'ks = [0, 25, 100]=>ks = [27, 100]' 'out = f"results/lens_{TAG}.pt"=>out = f"results/lens_{kind}_{TAG}.pt"' && echo LENS_DONE
echo ALL_DONE
