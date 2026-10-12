# STRATEGY CASE STUDIES

2C 没有运行，所以本轮没有 Student 的自由生成解题文本。仅有的案例是 2A reader 从 Teacher Z 自由生成 Plan（`teacher_reader_freerun.json`），逐对标签见 `TEACHER_SEMANTIC_AUDIT.md`。

2B 第 40 步时，自主 Z_S 在各题之间几乎相同（跨题 cos 0.987），对方法 span 偏好的影响与 crossq/mean 没有区别：held-out 上 |pref(self) − pref(crossq)| = 0.008，|pref(self) − pref(mean)| = 0.007，量级只是 0.01 nat。因此不存在“自主 latent 选择方法”的案例可以展示。逐题数据见 `autonomous_latent_interventions.csv`。
