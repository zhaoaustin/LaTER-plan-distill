"""Pilot gate on the real start checkpoint: cache-vs-full KL, gradient directions, loss/gradient scales."""
import json, sys
import pandas as pd, torch
from later.src.train.losses import stage2_exec_kl_full, stage2_exec_kl_topk_tail
from later.src.train.stage2_common import load_stage2_model, load_tokenizer
from later.src.train.stage2_monitor import Stage2Monitor
from later.src.train.stage2_objective import Stage2Module
from later.src.train.stage2_plan_dataset import Stage2Collator, build_views, stage2_tokens
from later.src.train.stage2_teacher_cache import ExecKLTeacherCache, TeacherBoundaryCache
S = "/scratch/hz33227/e2c/stage2"; T1 = "/scratch/hz33227/e2c/Explore-Execute-Chain/outputs/e2c_qwen3_4b_sft/best_hf"
CK = f"{S}/runs/stage2_L8_rms_2900/checkpoint-500"
tok = load_tokenizer(T1); T = stage2_tokens(tok); col = Stage2Collator(tok.pad_token_id)
val = pd.read_parquet(f"{S}/data/stage2_val.parquet")
kc = ExecKLTeacherCache(f"{S}/exec_kl_pilot/teacher_topk", "val_probe", {})
bc = TeacherBoundaryCache(f"{S}/teacher_cache", "val", {})
teacher = load_stage2_model(T1, latent_embed_builder_no_grad=True, latent_projector_dropout=0.0, dtype=torch.bfloat16).cuda().eval()
student = load_stage2_model(CK, latent_embed_builder_no_grad=False, latent_projector_dropout=0.0, dtype=torch.float32).cuda().eval()
L = list(range(1, 37)); res = {"start_checkpoint": CK, "samples": []}
for i in range(4):
    v = build_views(tok, val.iloc[i].to_dict(), 8, T, 4096); b = col([v]); e = kc.get(v["record_id"])
    offs = e["offsets"].long().cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = student.forward_plan_distill_stage2(input_ids=b["student_input_ids"].cuda(), attention_mask=b["student_attention_mask"].cuda(),
            position_ids=b["student_position_ids"].cuda(), labels=b["student_labels"].cuda(), boundary_positions=b["student_boundary"].cuda(),
            layer_ids=[36], latent_positions=b["latent_positions"].cuda(), latent_start_positions=b["latent_start_positions"].cuda(), return_target_logits=True)
    s_log = out["target_logits"][offs]
    with torch.no_grad():
        hid = teacher.model(input_ids=b["explicit_input_ids"].cuda(), position_ids=b["explicit_position_ids"].cuda(), use_cache=False).last_hidden_state
        t_log = teacher.lm_head(hid[0, v["explicit_boundary"] + offs]).float()
    kf = stage2_exec_kl_full(s_log, t_log); kk = stage2_exec_kl_topk_tail(s_log, e["ids"].cuda(), e["probs"].cuda(), e["tail"].cuda())
    gf = torch.autograd.grad(kf.mean(), s_log, retain_graph=True)[0].flatten(); gk = torch.autograd.grad(kk.mean(), s_log)[0].flatten()
    res["samples"].append({"record_id": v["record_id"], "n_kl_positions": len(offs), "kl_full_live": kf.mean().item(),
        "kl_topk_tail_cache": kk.mean().item(), "abs_diff": abs(kf.mean().item() - kk.mean().item()),
        "grad_cosine": torch.nn.functional.cosine_similarity(gf, gk, dim=0).item(), "teacher_topk_mass_mean": float(e["probs"].sum(-1).mean()),
        "teacher_tail_max": float(e["tail"].max())})
# loss scales at the start checkpoint (1 batch of 4, one forward with the pilot B objective)
views = [build_views(tok, val.iloc[i].to_dict(), 8, T, 4096) for i in range(4)]; b = col(views)
th = bc.get(b["record_ids"], L, "cuda"); klt = [kc.get(r) for r in b["record_ids"]]
mod = Stage2Module(student, dict(lambda_a=0.1, lambda_r=0.3, execution_ce_weight=1.0, alignment_epsilon=1e-6, execution_kl_enabled=True, lambda_kl=0.3), L)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    o = mod(b, th, kl_targets=klt)
res["loss_scales_start"] = {"L_exec": float(o["l_exec"]), "L_align": float(o["l_align"]), "L_explicit": float(o["l_explicit"]),
    "L_exec_kl_raw": float(o["l_exec_kl"]), "L_exec_kl_weighted": 0.3 * float(o["l_exec_kl"]),
    "weighted_align": 0.1 * float(o["l_align"]), "weighted_explicit": 0.3 * float(o["l_explicit"]), "L_total_B": float(o["loss"])}
mon = Stage2Monitor(student, tok, val, bc, L, 8, probe_n=8, probe_batch_size=4, grad_batch_size=4, kl_val_cache=kc, lambdas=(0.1, 0.3))
res["projector_grad_scales_start"] = mon.projector_grad_split(0.1)
assert all(p.grad is None for p in student.parameters())
json.dump(res, open(sys.argv[1], "w"), indent=1); print(json.dumps(res, indent=1))
