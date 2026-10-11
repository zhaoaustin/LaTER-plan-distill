# Test C — representative questions (three-loss L=8 step 500)

Gold plan shown for review only; it is never fed to the student latent path. Per step t: L2 norm, relative distance to the cross-question mean (= ||x_i - mu_t|| / mean_j ||x_j - mu_t||; 1 = average), and the nearest other question (centred cosine). ΔCE = swapped − original, mean over non-repeated execution tokens (controlled forced forward, donor fixed).

## s1val-00000-52160bf1 — keyword type (rough): probability

**Q:** (10) Divide a line segment of length $a$ into three segments. The probability that these three segments can form a triangle is $\qquad$

**Gold plan (excerpt):** 'I need to make a step-by-step plan.\n1. Recognize that the problem involves dividing a segment of length $ a $ into three parts and finding the probability that these parts can form a triangle.  \n2. Use the triangle inequality conditions to determine the necessary and sufficient condition for forming a triangle: each segment must be less than $ \\fra'

| t | ‖h_in‖ | ‖z_out‖ | rel. dist h_in | rel. dist z_out | NN question (z_out, cos) |
|---|---|---|---|---|---|
| 1 | 174.3 | 1.111 | 0.85 | 0.88 | probability (0.67) |
| 2 | 137.6 | 1.111 | 0.90 | 0.87 | other algebra (0.19) |
| 4 | 139.2 | 1.111 | 0.90 | 0.87 | other algebra (0.26) |
| 8 | 139.1 | 1.111 | 0.89 | 0.85 | probability (0.20) |

Step swap ΔCE: swap_step_1 +0.00096, swap_step_8 +0.00065, swap_prefix_1:4 +0.00053, swap_all_1:8 +0.00087

## s1val-00001-80256228 — keyword type (rough): word problem

**Q:** 34. Find the number of 20 -tuples of integers $x_{1}, \ldots, x_{10}, y_{1}, \ldots, y_{10}$ with the following properties:
- $1 \leq x_{i} \leq 10$ and $1 \leq y_{i} \leq 10$ for each $i$;
- $x_{i} \leq x_{i+1}$ for $i=1, \ldots, 9$
- if $x_{i}=x_{i+1}$, then $y_{i} \leq y_{i+1}$.

**Gold plan (excerpt):** "I need to make a step-by-step plan.\n1. Recognize that the problem involves counting 20-tuples $(x_1, \\ldots, x_{10}, y_1, \\ldots, y_{10})$ with specific constraints on non-decreasing values and monotonicity conditions.  \n2. Note that the sequence of $x_i$'s is non-decreasing, and for each block of equal $x_i$'s, the corresponding $y_i$'s must also"

| t | ‖h_in‖ | ‖z_out‖ | rel. dist h_in | rel. dist z_out | NN question (z_out, cos) |
|---|---|---|---|---|---|
| 1 | 176.3 | 1.111 | 0.87 | 0.79 | number theory (0.65) |
| 2 | 138.5 | 1.111 | 0.99 | 0.98 | number theory (0.34) |
| 4 | 138.9 | 1.111 | 0.99 | 0.96 | number theory (0.38) |
| 8 | 138.3 | 1.111 | 1.01 | 0.99 | number theory (0.31) |

Step swap ΔCE: swap_step_1 +0.00113, swap_step_8 +0.00081, swap_prefix_1:4 +0.00002, swap_all_1:8 +0.00101

## s1val-00002-0befa8a1 — keyword type (rough): trigonometry

**Q:** 7. Simplify: (1) $\frac{1}{\sin 45^{\circ} \sin 46^{\circ}}+\frac{1}{\sin 47^{\circ} \sin 48^{\circ}}+\cdots+\frac{1}{\sin 133^{\circ} \sin 134^{\circ}}$;
(2) $\frac{1}{\sin 1^{\circ} \sin 2^{\circ}}+\frac{1}{\sin 2^{\circ} \sin 3^{\circ}}+\cdots+\frac{1}{\sin 89^{\circ} \sin 90^{\circ}}$;
(3) $2 \sin 2^{\circ}+4 \sin 4^{\circ}+6 \sin 6^{\circ}+\cd

**Gold plan (excerpt):** 'I need to make a step-by-step plan.\n1. Identify the general term of the series and confirm the pattern of the sequence.  \n2. Use the trigonometric identity $\\frac{1}{\\sin x \\sin(x + 1^\\circ)} = \\cot x - \\cot(x + 1^\\circ)$ to transform each term.  \n3. Recognize that the transformed series forms a telescoping sum.  \n4. Simplify the telescoping sum by'

| t | ‖h_in‖ | ‖z_out‖ | rel. dist h_in | rel. dist z_out | NN question (z_out, cos) |
|---|---|---|---|---|---|
| 1 | 176.1 | 1.111 | 0.87 | 0.80 | other algebra (0.60) |
| 2 | 136.7 | 1.111 | 1.09 | 1.09 | other algebra (0.39) |
| 4 | 138.0 | 1.111 | 1.11 | 1.10 | other algebra (0.46) |
| 8 | 137.4 | 1.110 | 1.28 | 1.27 | other algebra (0.47) |

Step swap ΔCE: swap_step_1 +0.00041, swap_step_8 +0.00038, swap_prefix_1:4 +0.00001, swap_all_1:8 +0.00054

## s1val-00003-120d5743 — keyword type (rough): other algebra

**Q:** 14. Calculate:
$$
\frac{\frac{1}{1 \times 2}+\frac{1}{2 \times 3}+\frac{1}{3 \times 4}+\frac{1}{4 \times 5}+\ldots+\frac{1}{2021 \times 2022}}{\left(1-\frac{1}{2}\right)\left(1-\frac{1}{3}\right)\left(1-\frac{1}{4}\right)\left(1-\frac{1}{5}\right) \ldots\left(1-\frac{1}{2022}\right)}=
$$

**Gold plan (excerpt):** 'I need to make a step-by-step plan.\n1. Recognize that the numerator is a telescoping series and rewrite each term using the identity $\\frac{1}{n(n+1)} = \\frac{1}{n} - \\frac{1}{n+1}$.  \n2. Simplify the numerator by summing the telescoping series, which results in $1 - \\frac{1}{2022}$.  \n3. Simplify the denominator by rewriting each term $1 - \\frac{1'

| t | ‖h_in‖ | ‖z_out‖ | rel. dist h_in | rel. dist z_out | NN question (z_out, cos) |
|---|---|---|---|---|---|
| 1 | 173.6 | 1.111 | 1.09 | 1.19 | other algebra (0.74) |
| 2 | 137.6 | 1.111 | 1.08 | 1.08 | trigonometry (0.39) |
| 4 | 138.4 | 1.111 | 1.08 | 1.09 | trigonometry (0.46) |
| 8 | 138.3 | 1.111 | 1.07 | 1.07 | trigonometry (0.47) |

Step swap ΔCE: swap_step_1 +0.00016, swap_step_8 +0.00034, swap_prefix_1:4 +0.00027, swap_all_1:8 +0.00027

## s1val-00004-bddde0ed — keyword type (rough): integral / calculus

**Q:** ## Problem Statement

Calculate the limit of the function:

$\lim _{x \rightarrow 1}(\arcsin x)^{\tan \pi x}$

**Gold plan (excerpt):** 'I need to make a step-by-step plan.\n1. Recognize the indeterminate form $ 1^0 $ as $ x \\to 1 $ by evaluating the base and exponent separately.  \n2. Apply the exponential identity $ \\lim_{x \\to a} f(x)^{g(x)} = \\exp\\left( \\lim_{x \\to a} g(x) \\cdot \\ln(f(x)) \\right) $.  \n3. Substitute $ t = x - 1 $ to simplify the limit as $ t \\to 0 $, rewriting both'

| t | ‖h_in‖ | ‖z_out‖ | rel. dist h_in | rel. dist z_out | NN question (z_out, cos) |
|---|---|---|---|---|---|
| 1 | 175.4 | 1.111 | 0.96 | 0.90 | integral / calculus (0.70) |
| 2 | 138.2 | 1.111 | 1.02 | 1.01 | integral / calculus (0.53) |
| 4 | 140.3 | 1.111 | 0.98 | 0.96 | integral / calculus (0.44) |
| 8 | 138.8 | 1.111 | 1.00 | 1.01 | integral / calculus (0.47) |

Step swap ΔCE: swap_step_1 +0.00065, swap_step_8 +0.00051, swap_prefix_1:4 +0.00086, swap_all_1:8 +0.00033

## s1val-00005-2cc00e57 — keyword type (rough): sequence / series

**Q:** ## problem statement

Write the decomposition of vector $x$ in terms of vectors $p, q, r$:

$x=\{3 ;-3 ; 4\}$

$p=\{1 ; 0 ; 2\}$

$q=\{0 ; 1 ; 1\}$

$r=\{2 ;-1 ; 4\}$

**Gold plan (excerpt):** 'I need to make a step-by-step plan.\n1. Set up the vector equation $ x = a p + b q + c r $ by expressing each component of $ x $ as a linear combination of the corresponding components of $ p, q, r $.  \n2. Write the system of linear equations by equating the components of the resulting vector to the components of $ x $.  \n3. Solve the system of equa'

| t | ‖h_in‖ | ‖z_out‖ | rel. dist h_in | rel. dist z_out | NN question (z_out, cos) |
|---|---|---|---|---|---|
| 1 | 175.3 | 1.111 | 0.87 | 0.78 | other algebra (0.33) |
| 2 | 137.2 | 1.111 | 0.93 | 0.95 | sequence / series (0.30) |
| 4 | 138.2 | 1.111 | 0.89 | 0.85 | sequence / series (0.30) |
| 8 | 138.3 | 1.111 | 0.94 | 0.90 | sequence / series (0.34) |

Step swap ΔCE: swap_step_1 -0.00056, swap_step_8 -0.00049, swap_prefix_1:4 -0.00077, swap_all_1:8 -0.00036

## s1val-00007-910fd88b — keyword type (rough): number theory

**Q:** Subject (1). Determine the natural number $a=\frac{p+q}{r}+\frac{q+r}{p}+\frac{r+p}{q}$, where $p, q$, and $r$ are positive prime numbers.

## LUCIAN PETRESCU

**Gold plan (excerpt):** 'I need to make a step-by-step plan.\n1. Recognize that $ a $ is a sum of fractions involving prime numbers $ p, q, r $.  \n2. Simplify the expression by separating each fraction into two terms.  \n3. Group similar terms to form pairs like $ \\frac{p}{r} + \\frac{r}{p} $, which can be expressed as $ \\frac{p^2 + r^2}{pr} $.  \n4. Consider that $ a $ must b'

| t | ‖h_in‖ | ‖z_out‖ | rel. dist h_in | rel. dist z_out | NN question (z_out, cos) |
|---|---|---|---|---|---|
| 1 | 176.3 | 1.111 | 0.74 | 0.65 | number theory (0.63) |
| 2 | 139.8 | 1.111 | 0.83 | 0.78 | other algebra (0.45) |
| 4 | 139.7 | 1.111 | 0.76 | 0.72 | other algebra (0.34) |
| 8 | 139.2 | 1.111 | 0.76 | 0.72 | number theory (0.37) |

Step swap ΔCE: swap_step_1 +0.00049, swap_step_8 +0.00038, swap_prefix_1:4 +0.00039, swap_all_1:8 +0.00033

## s1val-00008-6a7ff50e — keyword type (rough): geometry

**Q:** 9. The curvature of a solid figure is $2 \pi$ minus the sum of the dihedral angles (the interior angles of the faces of a polyhedron). For example, for a cube, the dihedral angle is $2 \pi - 3 \times \frac{\pi}{2} = \frac{\pi}{2}$, so its total curvature is $\frac{\pi}{2} \times 8 = 4 \pi$. The total curvature of a square pyramid (4 triangles and o

**Gold plan (excerpt):** 'I need to make a step-by-step plan.\n1. Identify the number of vertices in the square pyramid (5 vertices).  \n2. Determine the dihedral angles at each vertex: calculate the dihedral angle at the apex and at each base vertex.  \n3. Apply the curvature formula at each vertex: $ \\text{Curvature at a vertex} = 2\\pi - \\sum (\\text{dihedral angles at that v'

| t | ‖h_in‖ | ‖z_out‖ | rel. dist h_in | rel. dist z_out | NN question (z_out, cos) |
|---|---|---|---|---|---|
| 1 | 177.4 | 1.111 | 0.83 | 0.84 | other algebra (0.68) |
| 2 | 139.4 | 1.111 | 1.18 | 1.23 | geometry (0.41) |
| 4 | 139.0 | 1.111 | 1.18 | 1.20 | geometry (0.40) |
| 8 | 138.5 | 1.111 | 1.15 | 1.19 | geometry (0.42) |

Step swap ΔCE: swap_step_1 +0.00066, swap_step_8 -0.00030, swap_prefix_1:4 +0.00011, swap_all_1:8 -0.00016

## s1val-00011-e8056295 — keyword type (rough): polynomial / roots

**Q:** 2. If the positive root of the quadratic equation $x^{2}-p x-q=0$ $\left(p, q \in \mathbf{N}_{+}\right)$ is less than 3, then the number of such quadratic equations is ( ).
(A) 5
(B) 6
(C) 7
(D) 8

**Gold plan (excerpt):** 'I need to make a step-by-step plan.\n1. Identify the quadratic equation and its roots using the quadratic formula.  \n2. Set up the inequality based on the condition that the positive root is less than 3.  \n3. Simplify the inequality to isolate the square root term and square both sides.  \n4. Solve the resulting inequality to find the relationship be'

| t | ‖h_in‖ | ‖z_out‖ | rel. dist h_in | rel. dist z_out | NN question (z_out, cos) |
|---|---|---|---|---|---|
| 1 | 171.3 | 1.111 | 0.87 | 0.76 | other algebra (0.60) |
| 2 | 139.7 | 1.111 | 0.77 | 0.76 | number theory (0.36) |
| 4 | 140.0 | 1.111 | 0.74 | 0.70 | polynomial / roots (0.33) |
| 8 | 139.7 | 1.111 | 0.78 | 0.77 | number theory (0.33) |

Step swap ΔCE: swap_step_1 +0.00048, swap_step_8 +0.00066, swap_prefix_1:4 +0.00047, swap_all_1:8 +0.00070

## s1val-00073-7e3c0b85 — keyword type (rough): counting / combinatorics

**Q:** 24. C4 (BUL) Let \( T \) be the set of ordered triples \((x, y, z)\), where \( x, y, z \) are integers with \( 0 \leq x, y, z \leq 9 \). Players \( A \) and \( B \) play the following guessing game: Player \( A \) chooses a triple \((x, y, z)\) in \( T \), and Player \( B \) has to discover \( A \)'s triple in as few moves as possible. A move consi

**Gold plan (excerpt):** 'I need to make a step-by-step plan.\n1. Recognize that the hidden triple $(x, y, z)$ can be uniquely determined by knowing the pairwise sums $S_1 = x + y$, $S_2 = y + z$, and $S_3 = z + x$.  \n2. Understand that each move provides the sum of absolute differences between the actual pairwise sums and the pairwise sums of the guessed triple.  \n3. Realiz'

| t | ‖h_in‖ | ‖z_out‖ | rel. dist h_in | rel. dist z_out | NN question (z_out, cos) |
|---|---|---|---|---|---|
| 1 | 179.2 | 1.111 | 1.13 | 1.20 | other algebra (0.94) |
| 2 | 136.9 | 1.111 | 0.97 | 1.01 | other algebra (0.36) |
| 4 | 138.5 | 1.111 | 0.95 | 0.98 | number theory (0.35) |
| 8 | 138.5 | 1.111 | 0.91 | 0.93 | number theory (0.34) |

Step swap ΔCE: swap_step_1 +0.00096, swap_step_8 -0.00063, swap_prefix_1:4 +0.00031, swap_all_1:8 +0.00051

## Pair curves: centred distance ||z_i - z_j|| / mean pairwise distance at that step

| pair | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
|---|---|---|---|---|---|---|---|---|
| probability vs word problem | 0.69 | 0.93 | 0.75 | 0.92 | 0.82 | 0.96 | 0.88 | 0.94 |
| trigonometry vs other algebra | 0.67 | 0.84 | 0.69 | 0.79 | 0.80 | 0.85 | 0.73 | 0.86 |
| integral / calculus vs sequence / series | 0.75 | 0.90 | 1.28 | 0.88 | 1.33 | 0.93 | 1.00 | 0.92 |

## Step-swap cases (swap_all_1:8; largest decrease, largest increase, median)

- **s1val-00137-e60566c5** (other algebra), donor s1val-00079-4df68eaa: original CE 0.12618, swap_all 0.12478 (Δ -0.00140), swap_step_1 Δ -0.00030, first-token KL 1.14e-07. Q: '7.1. The sum of the minuend, subtrahend, and difference is 25. Find the minuend.'
- **s1val-00032-d85349a2** (other algebra), donor s1val-00020-709a78b5: original CE 0.13043, swap_all 0.13234 (Δ +0.00191), swap_step_1 Δ -0.00035, first-token KL 4.50e-08. Q: 'If $x$ is real and positive and grows beyond all bounds, then $\\log_3{(6x-5)}-\\log_3{(2x+1)}$ approaches:\n$\\textbf{(A)}\\ 0\\qquad \\textbf{(B)}\\ 1\\qquad \\textbf{('
- **s1val-00118-33315a27** (integral / calculus), donor s1val-00111-7fa4b22f: original CE 0.12425, swap_all 0.12431 (Δ +0.00006), swap_step_1 Δ +0.00002, first-token KL -5.25e-08. Q: '## Problem Statement\n\nCalculate the definite integral:\n\n$$\n\\int_{0}^{1} \\frac{x^{3}+x}{x^{4}+1} d x\n$$'

All 128: swap_all ΔCE range [-0.00140, +0.00191], |Δ| median 0.00032; original per-question CE median 0.1832.
