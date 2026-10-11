# 复现

- 代码分支：`exp/bridge-input-abcd`（基于 d407338）。
- 新增脚本：
  - `later/scripts/bridge/bridge_input_abcd.py`
  - `later/scripts/bridge/abcd_report.py`
- 旧文件只读，没有修改。

```bash
cd /scratch/hz33227/e2c/stage2/LaTER
M=/scratch/hz33227/e2c/Explore-Execute-Chain/outputs/e2c_qwen3_4b_sft/best_hf; O=/scratch/hz33227/e2c/stage2/outputs/bridge_input_abcd
PY=~/.conda/envs/later-b200/bin/python
for m in raw std res res_std; do   # 实际运行时两张 GPU 各跑两组
  srun --jobid=48907334 --overlap --gres=gpu:1 $PY -m later.scripts.bridge.bridge_input_abcd --mode test  --input_mode $m --model $M --out $O --run_dir $O/group_$m
  srun --jobid=48907334 --overlap --gres=gpu:1 $PY -m later.scripts.bridge.bridge_input_abcd --mode train --steps 100 --input_mode $m --model $M --out $O --run_dir $O/group_$m
  srun --jobid=48907334 --overlap --gres=gpu:1 $PY -m later.scripts.bridge.bridge_input_abcd --mode audit --input_mode $m --model $M --out $O --run_dir $O/group_$m
  for s in 50 100; do srun --jobid=48907334 --overlap --gres=gpu:1 $PY -m later.scripts.bridge.bridge_input_abcd --mode final --input_mode $m --ckpt $O/group_$m/bridge_checkpoint/step$s.pt --model $M --out $O --run_dir $O/group_$m; done
done
$PY later/scripts/bridge/abcd_report.py $O /scratch/hz33227/e2c/stage2/outputs/mini_bridge_pilot .
```

## 输出

- `ABCD_REPORT.md`
- `run_manifest.json`
- `train_stats.npz`（只用训练集统计）
- `bridge_input_abcd_metrics.json`
- `heldout_per_pair.jsonl`
- `heldout_generations.md`
- `code_patch.diff`
- `group_{raw,std,res,res_std}/`：每组包含
  - `train_log.jsonl`
  - `unit_tests_smoke.json`
  - `input_representation_audit.json`
  - `final_step{50,100}.json`
  - `generations_step{50,100}.jsonl`
  - `bridge_checkpoint/`
- 每组的 `train_log.jsonl` 即任务书要求的训练日志，按组分开存放。
