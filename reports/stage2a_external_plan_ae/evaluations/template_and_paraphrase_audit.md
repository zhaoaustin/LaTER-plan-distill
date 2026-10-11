# 模板与改写审计（step 300）

## 验证集 40 条自由生成（自动指标）

| 条件 | n | 与自身参考的 token F1 | 在 40 条中找回自己的 Plan | 找回 donor 的 Plan | 方法词召回 | 不同生成数 |
|---|---|---|---|---|---|---|
| correct | 40 | 0.5 | 25/40 | None | 0.387 | 40 |
| shuffled | 40 | 0.33 | 0/40 | 25 | 0.023 | 40 |
| mean | 10 | 0.36 | 0/10 | None | 0.0 | 1 |
| zero | 10 | 0.336 | 0/10 | None | 0.0 | 1 |

**通用模板检查：**
- 均值 Z 和零 Z 下，10 条输入都生成同一段固定模板，开头分别是：“1. Define variables for the number of red, blue, and green b…” 和 “1. Identify the positions of the red, white, and blue balls …”。
- 正确 Z 下的 40 条生成各不相同，没有出现“先识别信息，再一步步求解”这类通用模板。
- 打乱 Z 时，生成的是提供 Z 的那条 Plan（donor）的内容，25/40 能找回 donor。

## 改写稳定性（6 条人工改写，只改措辞、不改方法）

| 题 / 方法 | l(原 Plan \| Z 原) | l(原 Plan \| Z 改写) | l(原 Plan \| Z 另一 Plan) | cos(Z 原, Z 改写) | cos(Z 原, Z 另一 Plan) | 改写 Z 的生成是否保留方法 |
|---|---|---|---|---|---|---|
| s1val-00336-befc9279 Lagrange multipliers | -0.359 | -0.360 | -0.466 | 0.863 | 0.664 | 是（Lagrange） |
| s1val-00574-2a030f37 difference-of-squares factoring of 65 | -0.672 | -0.707 | -0.738 | 0.855 | 0.656 | 是（因式分解，方程是编的） |
| s1val-00569-33b9f53f discriminant of a quadratic in x | -0.687 | -0.715 | -1.023 | 0.764 | 0.628 | 否（换元，与原 Z 的生成相同） |
| s1val-00309-9b08eaea Euler's theorem (phi(9)=6) | -0.471 | -0.477 | -0.476 | 0.839 | 0.843 | 否（CRT，无关） |
| s1val-00305-696c8aa4 substitute y=6/x + quadratic formula | -0.397 | -0.408 | -0.463 | 0.814 | 0.785 | 是（代入） |
| s1val-00479-999dadd8 factor into linear x quadratic | -0.973 | -1.116 | -1.108 | 0.792 | 0.653 | 是（有理根分解） |

**结论：**
- 5/6 条改写后的 Z 比另一个 Plan 的 Z 更接近原 Z（00309 持平）。
- 在改写 Z 下，原 Plan 的似然接近原 Z 下的值，5/6 优于另一个 Plan 的 Z。
- 从改写 Z 生成，4/6 保留了原方法。
