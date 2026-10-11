# Sanity checks (all hard gates PASS)

| check | value |
|---|---|
| train_stats_reproduce_saved | True |
| trainable_params | 9732096 |
| total_params_executor | 4032200192 |
| trainable_all_lora | True |
| trainable_module_examples | ['o_proj', 'q_proj', 'v_proj'] |
| bridge_trainable_params | 0 |
| lora_init_score_abs_diff | 1.4901161193847656e-08 |
| lora_init_score_equals_step0 | True |
| online_Z_adapter_disabled_equals_cache_max_abs | 0.0 |
| optimizer_param_count | 9732096 |
| lora_changed_by_smoke | True |
| backbone_bitwise_unchanged_after_smoke | True |
| bridge_bitwise_unchanged | True |
| stats_unchanged | True |
| train_vs_generate_first_token_equal | True |
| executor_prefix_has_no_plan_text | True |
| ALL_HARD_GATES_PASS | True |
| smoke step 0 | loss 0.3880, grad finite True, grad norm 0.1326, Z in slots True, non-LoRA grads None True |
| smoke step 1 | loss 0.3834, grad finite True, grad norm 0.1288, Z in slots True, non-LoRA grads None True |
| post: backbone_bitwise_unchanged_after_training | True |
| post: bridge_bitwise_unchanged_after_training | True |
| post: train_time_s | 80.79021644592285 |
| post: peak_mem_gb | 74.4567289352417 |
| post: total_wall_s | 375.1743104457855 |

Attempt 1 stopped at the gate 'lora_init_score_equals_step0' (False). Diagnosis (tmp/lora_init_check.py): with LoRA B=0 the scores differ from no-LoRA by 6e-8 (repeat-run noise 1.2e-7, first logits identical); the gate compared a 1-sequence batch with a 10-sequence padded batch (different bf16 kernels). Fixed to compare identical batches -> diff 1.5e-8. No training happened in attempt 1 (log: run_attempt1_gate_test_bug.log).
