#!/bin/bash
source /scratch/hz33227/e2c/stage2/outputs/stage2c_anchor_lambda01_20261012/run_env.sh
O=/scratch/hz33227/e2c/stage2/outputs/stage2b_4sample_overfit_20261012
IDS=s1val-00626,s1val-00496
cd $R
B="$PY -m later.scripts.stage2t.bridge_oracle --model $M --s2b $S2B --out $O --init $CK0 --only_ids $IDS"
export CUDA_VISIBLE_DEVICES=0
$B --mode audit >> $O/audit.log 2>&1
grep -q '"PASS": true' $O/audit.json || { echo AUDIT_FAIL; exit 1; }
$B --mode eval >> $O/eval.log 2>&1
prev=""
for s in 5 10 15 20; do
  $B --mode train --stop_at $s ${prev:+--resume $prev} >> $O/train.log 2>&1
  prev=$O/checkpoints/bridge_step$s.pt
  $B --mode eval --ckpt $prev >> $O/eval.log 2>&1
  $PY -c "
import json,sys; e=json.load(open('$O/evaluations/eval_bridge_step$s.json'))['train']
c=int(e['correct'].split('/')[0]); f=int(e['format_ok'].split('/')[0])
sys.exit(1 if (f<=1 or c==0) else 0)" || { echo "EARLY STOP at step $s"; exit 0; }
done
