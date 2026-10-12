# STATUS

- 2026-10-12 Gate 0：资产齐全（Stage-1 step800、2A step300、2B 数据与 Z 缓存、0.6B reader）。2B-v2 没有运行记录。Teacher 判定 TEACHER_SEMANTICS_WEAK，按诊断性 pilot 继续。
- Gate 1：14/14 硬性单测通过；Tiny Overfit（10 步）弱通过。
- Stage 2B（40 步）：**KD_COLLAPSE**。未见题 KD correct 1.168 ≈ shuffled 1.169；Z_S 跨题 cos 0.987；跨题方差比 0.024。
- 按任务书 7.2 和第 9 节**停止**：2C Control 和 Choice 都没有运行。
- 只读诊断（不消耗 optimizer step）：Teacher reader 自由生成、Q→Z_T ridge 探针、Choice 梯度审计。
- 预算：50/160 步，约 35 分钟。GPU 已释放。没有 push，没有覆盖旧 checkpoint，没有进入 Stage 3。
- **最终标签：KD_COLLAPSE**（另有：ENGINEERING_PASS、TEACHER_SEMANTICS_WEAK；2C 相关标签全部为“未评估”）。
