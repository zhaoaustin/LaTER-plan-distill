# 同题双 Plan：自由生成（step 300，greedy，最多 320 token）

- Decoder 只看到固定指令加 8 个 Z，没有题目，也没有参考 Plan。
- 人工判定标准：生成的方法是否与 P_A / P_B 一致。只换措辞不算切换。

## held-out 8 对（人工判定）

| 题 | 方法 A / B | 判定 | 说明 |
|---|---|---|---|
| s1val-00336-befc9279 | Lagrange multipliers / Cauchy-Schwarz inequality | **清楚切换** | Z_A 生成 Lagrange 乘子（方法 A）；Z_B 生成 AM-GM 不等式加对称（与 B 同属不等式法，但不是 Cauchy-Schwarz）。题目细节是编的（目标函数不同）。 |
| s1val-00574-2a030f37 | difference-of-squares factoring of 65 / testing small n + modular arithmetic | **清楚切换** | Z_A 生成因式分解后列因子对（方法 A 的结构，但方程是编的：x^y+y^x=2016）；Z_B 生成试小的 n（方法 B）。 |
| s1val-00569-33b9f53f | discriminant of a quadratic in x / derivative / critical points | **只恢复 B** | Z_A 生成换元 y=x−1，不是判别式法；Z_B 生成求导找临界点（方法 B）。 |
| s1val-00507-b7055178 | differentiate F to a density + double integral / inclusion-exclusion on the CDF | **只恢复 A** | Z_A 生成二重积分加极坐标（方法 A 的二重积分思路）；Z_B 也是对密度做二重积分，没有恢复 CDF 容斥。 |
| s1val-00479-999dadd8 | factor into linear x quadratic / Vieta's formula on the expanded cubic | **部分恢复** | Z_A 生成有理根分解，最后用 Vieta（方法 A 为主）；Z_B 生成系数对称函数（与 Vieta 相近，表述含混）。 |
| s1val-00309-9b08eaea | Euler's theorem (phi(9)=6) / order of 4 mod 9 is 3 | **失败** | Z_A 生成 n! 的末位数字（无关）；Z_B 生成 CRT 加 Euler 定理（这是 A 的方法，不是 B 的“阶为 3”）。 |
| s1val-00305-696c8aa4 | substitute y=6/x + quadratic formula / factor xy(x+y)+(x+y), then x^2+y^2 identity | **只恢复 A** | 两边都生成“用一个变量表示另一个再代入”（方法 A），Z_B 没有恢复因式分解。 |
| s1val-00486-d1e0b543 | sum-of-cosines-in-AP identity / 7th roots of unity | **清楚切换** | Z_A 生成等差余弦求和公式（方法 A）；Z_B 生成单位圆上的复数和对称性（单位根思路，方法 B）。 |

## 全文：heldout8

### s1val-00336-befc9279

**P_A**:
```
1. Use Lagrange Multipliers to set up the system with the objective function $ f(x, y, z) = \frac{1}{x} + \frac{4}{y} + \frac{9}{z} $ and constraint $ g(x, y, z) = x + y + z - 1 = 0 $.  
2. Compute the partial derivatives of $ f $ and $ g $, then set up the equations $ \nabla f = \lambda \nabla g $.  
3. Solve the system of equations to express $ x, y, z $ in terms of $ \lambda $.  
4. Substitute these expressions into the constraint $ x + y + z = 1 $ to find the value of $ \lambda $.  
5. Use the value of $ \lambda $ to determine the values of $ x, y, z $.  
6. Substitute $ x, y, z $ back into the original objective function to compute the minimum value.
```
**gen(Z_A)**:
```
1. Use Lagrange multipliers to set up the system of equations from the constraint and the function to minimize.  
2. Express the constraint in terms of $ x $, $ y $, and $ z $, and substitute $ z $ from the constraint into the function.  
3. Simplify the function to $ f(x, y, z) = \frac{x}{y} + \frac{y}{z} + \frac{z}{x} $ under the constraint $ x + y + z = 1 $.  
4. Compute the partial derivatives of $ f $ with respect to $ x $, $ y $, and $ z $, and set them equal to zero.  
5. Solve the system of equations obtained from the partial derivatives to find critical points.  
6. Verify that the critical point satisfies the constraint and compute the value of the function at that point.  
7. Conclude that the minimum value occurs at the critical point found in step 5.
```
**P_B**:
```
1. Apply the Cauchy-Schwarz Inequality in the form $(a_1^2 + a_2^2 + a_3^2)(b_1^2 + b_2^2 + b_3^2) \geq (a_1b_1 + a_2b_2 + a_3b_3)^2$ by choosing appropriate vectors.  
2. Express the target expression $\frac{1}{x} + \frac{4}{y} + \frac{9}{z}$ as a dot product of two vectors: $(1, 2, 3)$ and $\left(\frac{1}{x}, \frac{2}{y}, \frac{3}{z}\right)$.  
3. Calculate the squared norms of both vectors: $(1^2 + 2^2 + 3^2) = 14$ and $\left(\frac{1}{x^2} + \frac{4}{y^2} + \frac{9}{z^2}\right)$.  
4. Use the constraint $x + y + z = 1$ to find a relationship between $x$, $y$, and $z$.  
5. Apply the AM-GM Inequality to the terms $\frac{1}{x}, \frac{4}{y}, \frac{9}{z}$ with appropriate weights to minimize the sum.  
6. Verify that equality in AM-GM is achievable by finding values of $x$, $y$, and $z$ that satisfy the constraint and the equality conditions.
```
**gen(Z_B)**:
```
1. Express the given condition $ a^2 + b^2 + c^2 = 1 $ in terms of $ x, y, z $ where $ x = \frac{1}{a}, y = \frac{1}{b}, z = \frac{1}{c} $.  
2. Rewrite the target expression $ \frac{1}{x^2} + \frac{1}{y^2} + \frac{1}{z^2} $ in terms of $ x, y, z $ using the reciprocal identities.  
3. Use the AM-GM inequality to relate the sum of squares to the product of the reciprocals.  
4. Assume symmetry by setting $ x = y = z $ to maximize the sum of reciprocals.  
5. Verify that the symmetric case satisfies the given condition and compute the value of the target expression.
```

### s1val-00574-2a030f37

**P_A**:
```
1. Rewrite the equation $ 4^n + 65 = 9^m $ as $ (2^n)^2 + 65 = (3^m)^2 $.  
2. Define the difference of squares $ (3^m)^2 - (2^n)^2 = 65 $, which factors as $ (3^m - 2^n)(3^m + 2^n) = 65 $.  
3. List all factor pairs of 65 and set up equations for each pair: $ 3^m - 2^n = a $, $ 3^m + 2^n = b $, where $ a \cdot b = 65 $.  
4. Solve each system of equations to find $ m $ and $ n $.  
5. Check which factor pairs yield valid natural number solutions for $ m $ and $ n $.  
6. Verify that no other factor pairs (or permutations) yield valid solutions.  
7. Conclude the solution by listing all valid pairs $ (n, m) $.
```
**gen(Z_A)**:
```
1. Express the equation as $ x^y + y^x = 2016 $ and factor it as $ (x^y - 1)(y^x - 1) = 2016 $.  
2. List all factor pairs of 2016, both positive and negative, and consider only those where $ x^y - 1 $ and $ y^x - 1 $ are positive integers.  
3. For each factor pair, solve for $ x $ and $ y $ such that $ x^y = a $ and $ y^x = b $, where $ ab = 2016 $.  
4. Check if the resulting values of $ x $ and $ y $ are positive integers.  
5. Discard any negative or non-integer solutions.  
6. Collect all valid positive integer solutions $ (x, y) $ that satisfy the original equation.  
7. Verify each solution by substituting back into the original equation to ensure it holds true.
```
**P_B**:
```
1. Test small values of $ n $ to find corresponding $ m $ that satisfy the equation $ 4^n + 65 = 9^m $.  
2. Check if $ 4^n + 65 $ is a perfect power of 9 for these values.  
3. Observe that as $ n $ increases, $ 4^n $ grows exponentially, making $ 9^m $ require increasingly large values of $ m $.  
4. Use modular arithmetic (e.g., modulo 3) to eliminate potential values of $ n $ that cannot lead to $ 9^m $.  
5. Conclude that no other pairs of natural numbers $ (n, m) $ satisfy the equation based on the observed pattern and constraints.
```
**gen(Z_B)**:
```
1. Check small values of $ n $ to find possible solutions.  
2. For each $ n $, compute $ 2^n + 1 $ and check if it is a prime number.  
3. For each prime $ p $ found, check if $ p - 1 $ can be expressed as $ n \cdot k $ for some integer $ k $.  
4. Verify that $ p - 1 = n \cdot k $ for some integer $ k $, and check if $ p - 1 $ is a prime number.  
5. Conclude that the only solutions are when $ p - 1 $ is a prime number and $ p $ is a prime number.
```

### s1val-00569-33b9f53f

**P_A**:
```
1. Substitute $ y = \frac{3x^2 + 3x + 4}{x^2 + x + 1} $ and rearrange to form a quadratic equation in $ x $.  
2. Multiply both sides by the denominator to eliminate the fraction and rearrange terms to form a standard quadratic equation.  
3. Compute the discriminant of the quadratic equation to determine the values of $ y $ for which real solutions exist.  
4. Simplify the discriminant expression and solve the resulting inequality to find the maximum value of $ y $.
```
**gen(Z_A)**:
```
1. Substitute $ y = x - 1 $ to simplify the equation.  
2. Rewrite the equation in terms of $ y $ and simplify the expression.  
3. Solve the resulting equation for $ y $, then substitute back to find $ x $.  
4. Determine the range of $ x $ by analyzing the expression for $ y $.
```
**P_B**:
```
1. Recognize that the function is a rational function and aim to find its maximum value.  
2. Compute the derivative using the quotient rule to find critical points.  
3. Set the derivative equal to zero and simplify the resulting equation.  
4. Factor the simplified equation and solve for the critical points.  
5. Evaluate the function at the critical points to determine the maximum value.
```
**gen(Z_B)**:
```
1. Recognize the function as a rational function and identify the critical points by setting the derivative to zero.  
2. Compute the first derivative of the function.  
3. Set the first derivative equal to zero and solve for $ x $.  
4. Evaluate the function at the critical points to determine the maximum value.  
5. Confirm that the critical point found is indeed a maximum by checking the second derivative or analyzing the behavior of the function.
```

### s1val-00507-b7055178

**P_A**:
```
1. Identify the region of interest as the rectangle defined by $0 \leq x \leq \frac{\pi}{4}$ and $\frac{\pi}{6} \leq y \leq \frac{\pi}{3}$.  
2. Recognize that the probability is the double integral of the joint probability density function $f(x, y)$ over this region.  
3. Express the joint density function $f(x, y)$ by differentiating the distribution function $F(x, y)$ with respect to both $x$ and $y$.  
4. Set up the double integral of $\sin x \cdot \sin y$ over the specified rectangle.  
5. Evaluate the double integral by first integrating with respect to $y$ and then with respect to $x$.
```
**gen(Z_A)**:
```
1. Identify the region of integration by analyzing the inequality $ x^2 + y^2 \leq 1 $, which represents the unit disk centered at the origin.  
2. Determine the bounds for $ x $ and $ y $ by solving the inequality $ x^2 + y^2 \leq 1 $ for $ x $ and $ y $.  
3. Set up the double integral over the unit disk with the appropriate limits for $ x $ and $ y $.  
4. Use polar coordinates to simplify the integral, converting the Cartesian limits to polar coordinates.  
5. Express the integral in polar coordinates and compute the resulting integral.
```
**P_B**:
```
1. Recognize that the probability of $(X, Y)$ falling within a region is the distribution function evaluated at the upper bounds of that region minus the distribution function evaluated at the lower bounds.  
2. Identify the upper and lower bounds of the rectangle defined by $x=0, x=\pi/4$, $y=\pi/6$, and $y=\pi/3$.  
3. Substitute the upper bounds into the distribution function to compute $F(\pi/4, \pi/3)$.  
4. Substitute the lower bounds into the distribution function to compute $F(0, \pi/3)$, $F(\pi/4, \pi/6)$, and $F(0, \pi/6)$.  
5. Apply the inclusion-exclusion principle to subtract the overlapping regions from the total probability.  
6. Simplify the resulting expression to find the final probability.
```
**gen(Z_B)**:
```
1. Recognize the joint probability density function (PDF) $ f_{X,Y}(x, y) $ as a product of two independent uniform PDFs on $[0,1]$.  
2. Identify the region of integration where $ x < y $, which corresponds to the triangle in the unit square.  
3. Set up the double integral over the region $ x < y $ with the given PDF.  
4. Use the symmetry of the PDF to simplify the integral by considering the region $ y \geq x $ and double the result.  
5. Evaluate the integral by integrating with respect to $ y $, then with respect to $ x $.  
6. Substitute the known values of the integrals to compute the final result.
```

### s1val-00479-999dadd8

**P_A**:
```
1. Group and simplify the equation by combining like terms.  
2. Recognize that the equation can be factored into a product of two expressions.  
3. Identify the first factor as a linear term and the second as a quadratic term.  
4. Determine the sum of the roots of the cubic equation by noting that one root comes from the linear factor and the other two from the quadratic factor.  
5. Conclude that the sum of all three roots is the root of the linear factor.
```
**gen(Z_A)**:
```
1. Factor the cubic polynomial by testing rational roots using the Rational Root Theorem.  
2. Identify the real root and factor the cubic polynomial using that root.  
3. Recognize that the polynomial is a sum of three linear terms with the same real root.  
4. Conclude that the sum of these three terms is zero, implying the polynomial is divisible by $(x - r)^3$.  
5. Determine the sum of the roots using Vieta's formulas.
```
**P_B**:
```
1. Combine like terms in the given polynomial equation by grouping and simplifying.  
2. Identify the coefficients of the resulting polynomial in standard form.  
3. Use Vieta’s formula for cubic equations to find the sum of the roots, which is $-b/a$, where $b$ is the coefficient of $x^2$ and $a$ is the coefficient of $x^3$.
```
**gen(Z_B)**:
```
1. Use the sum of coefficients formula for a cubic polynomial to find the sum of the coefficients.  
2. Use the sum of products of coefficients taken two at a time to find the sum of products of coefficients taken two at a time.  
3. Use the sum of products of coefficients taken three at a time to find the product of the coefficients.
```

### s1val-00309-9b08eaea

**P_A**:
```
1. Recognize that finding the last two digits in ternary is equivalent to computing $13^{101} \mod 9$ since the last two digits in base 3 correspond to the value modulo $3^2 = 9$.  
2. Simplify the base modulo 9: $13 \equiv 4 \mod 9$, so $13^{101} \equiv 4^{101} \mod 9$.  
3. Use Euler’s Theorem: since $\gcd(4, 9) = 1$, $4^{\phi(9)} \equiv 1 \mod 9$, where $\phi(9) = 6$, so $4^6 \equiv 1 \mod 9$.  
4. Reduce the exponent modulo 6: $101 \div 6$ gives a remainder of 5, so $4^{101} \equiv 4^5 \mod 9$.  
5. Compute $4^5 \mod 9$: $4^2 = 16 \equiv 7 \mod 9$, then $4^4 = (4^2)^2 \equiv 7^2 = 49 \equiv 4 \mod 9$, and $4^5 = 4^4 \cdot 4 \equiv 4 \cdot 4 = 16 \equiv 7 \mod 9$.  
6. Conclude that the last two digits of $13^{101}$ in ternary are the ternary representation of 7, which is $21_3$.
```
**gen(Z_A)**:
```
1. Recognize that the problem requires finding the last three digits of $ n! $, where $ n $ is a power of 10.  
2. Use the property that $ n! $ modulo 10 is determined by its last few non-zero digits.  
3. Identify that $ n! $ will end in zeros for $ n \geq 10 $, and the last three digits are determined by the remaining non-zero digits.  
4. Note that for $ n \geq 10 $, the product of the remaining non-zero digits will be congruent to 1 modulo 10.  
5. Conclude that the last three digits of $ n! $ are 1, 0, and 0.  
6. Verify that this matches the given answer choices.
```
**P_B**:
```
1. Recognize that finding the last two digits in ternary is equivalent to computing $13^{101} \mod 9$.  
2. Reduce the base modulo 9: $13 \mod 9 = 4$, so the problem becomes finding $4^{101} \mod 9$.  
3. Use modular exponentiation to compute $4^{101} \mod 9$ by breaking down the exponent using patterns or properties of modular arithmetic.  
4. Apply the property $4^3 \equiv 1 \mod 9$ to simplify the exponentiation by reducing $101 \mod 3$.  
5. Calculate $4^{101 \mod 3} \mod 9$ to find the result.
```
**gen(Z_B)**:
```
1. Recognize that the problem requires finding $ 2013^{2013} \mod 1000 $, which is equivalent to finding the last three digits of $ 2013^{2013} $.  
2. Use modular exponentiation to compute $ 2013^{2013} \mod 1000 $.  
3. Simplify $ 2013 \mod 1000 $ to $ 13 $.  
4. Apply the Chinese Remainder Theorem by computing $ 13^{2013} \mod 8 $ and $ 13^{2013} \mod 125 $.  
5. Use Euler's theorem to find $ 13^{\phi(8)} \mod 8 $ and $ 13^{\phi(125)} \mod 125 $.  
6. Combine the results using the Chinese Remainder Theorem to find the final value.
```

### s1val-00305-696c8aa4

**P_A**:
```
1. Use the first equation $ xy = 6 $ to express $ y $ in terms of $ x $, then substitute into the second equation.  
2. Simplify the resulting equation to a quadratic in $ x $, then solve for $ x $ using the quadratic formula.  
3. Use the value of $ x $ to find the corresponding $ y $ using $ xy = 6 $.  
4. Calculate $ x^2 + y^2 $ using the values of $ x $ and $ y $ found in step 3.
```
**gen(Z_A)**:
```
1. Use the first equation to express $ x $ in terms of $ y $: $ x = \frac{y}{2} $.  
2. Substitute $ x = \frac{y}{2} $ into the second equation to form a single equation in terms of $ y $.  
3. Solve the resulting equation for $ y $, then substitute back to find $ x $.  
4. Calculate the product $ x \cdot y $ using the values of $ x $ and $ y $.
```
**P_B**:
```
1. Use the first equation $ xy = 6 $ to factor the second equation by factoring out $ xy $ from terms that share it.  
2. Substitute $ xy = 6 $ into the modified second equation to find $ x + y $.  
3. Use the identity $ x^2 + y^2 = (x + y)^2 - 2xy $ to compute $ x^2 + y^2 $.
```
**gen(Z_B)**:
```
1. Use the first equation to express $ x $ in terms of $ y $: $ x = \frac{y}{2} $.  
2. Substitute $ x = \frac{y}{2} $ into the second equation to solve for $ y $.  
3. Use the value of $ y $ to find $ x $, then compute $ x + y $.
```

### s1val-00486-d1e0b543

**P_A**:
```
1. Use trigonometric identities to simplify the sum of cosines.  
2. Apply the identity for the sum of cosines in an arithmetic sequence.  
3. Substitute specific values of the angles and simplify the expression.  
4. Recognize the periodic and symmetry properties of cosine to evaluate the sum.  
5. Use known mathematical results or symmetry arguments to determine the final value of the sum.
```
**gen(Z_A)**:
```
1. Use the identity for the sum of sines with angles in arithmetic progression.  
2. Simplify the expression using the identity and compute the sum of the sines.  
3. Apply the identity for the sum of cosines with angles in arithmetic progression.  
4. Simplify the expression using the identity and compute the sum of the cosines.  
5. Combine the results of the sums to find the value of the original expression.
```
**P_B**:
```
1. Recognize that the angles involved are symmetrically distributed around the unit circle.  
2. Use the identity $\cos \theta = \cos(2\pi - \theta)$ to rewrite $\cos\left(\frac{5\pi}{7}\right)$ as $\cos\left(\frac{9\pi}{7}\right)$.  
3. List all angles of the form $\cos\left(\frac{(2k+1)\pi}{7}\right)$ for $k = 0$ to $6$ to capture all odd multiples of $\frac{\pi}{7}$ within the interval $[0, 2\pi)$.  
4. Note that these angles are the real parts of the 7th roots of unity, excluding the real root 1.  
5. Use the known identity for the sum of cosines of odd multiples of $\frac{2\pi}{7}$ to find the result.
```
**gen(Z_B)**:
```
1. Recognize that the problem involves summing the real and imaginary parts of the complex numbers $ z_k = e^{i\theta_k} $ for $ k = 1, 2, \ldots, 2016 $.  
2. Note that $ z_k $ are complex numbers on the unit circle with angles $ \theta_k = \frac{k\pi}{2016} $.  
3. Identify that the sum of these complex numbers forms a regular 2016-sided polygon inscribed in the unit circle.  
4. Determine that the sum of all such complex numbers is zero due to symmetry.  
5. Conclude that the sum of the real parts and imaginary parts of these complex numbers will each be zero.
```


## 全文：extended20

### s1val-00645-de6f711a

**P_A**:
```
1. Use the product-to-sum identities on $\sin 10^\circ \sin 30^\circ$ and $\sin 50^\circ \sin 70^\circ$ separately.  
2. Simplify each product-to-sum expression using trigonometric identities and known values.  
3. Multiply the simplified results together.  
4. Rationalize the denominator of the resulting fraction.  
5. Confirm that the simplified result matches the given value $\frac{1}{16}$.
```
**gen(Z_A)**:
```
1. Use the identity $\sin A \sin B = \frac{1}{2}[\cos(A-B) - \cos(A+B)]$ to simplify $\sin 20^\circ \sin 10^\circ$.  
2. Use the identity $\cos A \cos B = \frac{1}{2}[\cos(A-B) + \cos(A+B)]$ to simplify $\cos 10^\circ \cos 10^\circ$.  
3. Use the identity $\cos A \cos B = \frac{1}{2}[\cos(A-B) + \cos(A+B)]$ to simplify $\cos 10^\circ \cos 20^\circ$.  
4. Use the identity $\cos A \cos B = \frac{1}{2}[\cos(A-B) + \cos(A+B)]$ to simplify $\cos 20^\circ \cos 30^\circ$.  
5. Use the identity $\cos A \cos B = \frac{1}{2}[\cos(A-B) + \cos(A+B)]$ to simplify $\cos 20^\circ \cos 40^\circ$.  
6. Use the identity $\cos A \cos B = \frac{1}{2}[\cos(A-B) + \cos(A+B)]$ to simplify $\cos 40^\circ \cos 50^\circ$.  
7. Use the identity $\cos A \cos B = \frac{1}{2}[\cos(A-B) + \
```
**P_B**:
```
1. Use the identity for product of sines with angles in arithmetic progression: $\sin A \sin(A + d) \sin(A + 2d) \sin(A + 3d)$.  
2. Apply the identity to the given angles, recognizing that $10^\circ, 30^\circ, 50^\circ, 70^\circ$ form an arithmetic sequence with common difference $20^\circ$.  
3. Substitute $A = 10^\circ$ and $d = 20^\circ$ into the identity.  
4. Simplify the right-hand side of the identity using trigonometric values and known results.  
5. Match the simplified expression with the left-hand side to verify the equality.
```
**gen(Z_B)**:
```
1. Use the identity for the product of sines of angles in arithmetic progression: $\sin A \sin B \sin C = \frac{1}{4} \sin A \sin B \sin C$.  
2. Apply the identity to the given angles: $1^\circ, 2^\circ, 3^\circ, \ldots, 100^\circ$.  
3. Express the product in terms of sine functions with arguments in arithmetic progression.  
4. Use the identity for the product of sines of angles in arithmetic progression: $\sin n^\circ \sin (n+1)^\circ \sin (n+2)^\circ = \frac{1}{4} \sin 100^\circ$.  
5. Substitute $n = 1$ into the identity to find the value of the product.
```

### s1val-00359-88eb7515

**P_A**:
```
1. Use the formula for the area of a trapezoid: $ \text{Area} = \frac{1}{2}(AB + CD) \cdot h $.  
2. Plug in the known values for the bases $ AB $ and $ CD $, and denote the height as $ h $.  
3. Apply the formula for the height in terms of the legs and the difference in the bases: $ h = \sqrt{BC^2 - \left( \frac{AB - CD}{2} \right)^2} $.  
4. Calculate the difference between the bases and substitute into the height formula.  
5. Compute the height using the Pythagorean Theorem.  
6. Substitute the height back into the trapezoid area formula to find the area.
```
**gen(Z_A)**:
```
1. Use the formula for the area of a trapezoid: $ \text{Area} = \frac{1}{2} \times (a + b) \times h $, where $ a $ and $ b $ are the lengths of the two bases, and $ h $ is the height.  
2. Calculate the sum of the bases: $ a + b = 12 + 14 = 26 $.  
3. Determine the height using the given diagonal and the Pythagorean theorem: $ h = \sqrt{13^2 + 14^2} = \sqrt{13^2 + 14^2} $.  
4. Substitute the values into the area formula and simplify to find the area.
```
**P_B**:
```
1. Identify the bases and the non-parallel sides of the trapezoid.  
2. Place the trapezoid on a coordinate plane with base $AB$ on the x-axis.  
3. Assign coordinates to points $A$, $B$, $D$, and $C$ based on the given side lengths.  
4. Use the distance formula to express the coordinates of point $C$ in terms of variable $h$.  
5. Set up equations for the lengths of $BC$ and $DA$ using the coordinates.  
6. Solve the resulting system of equations to find the value of $h$.  
7. Calculate the area of the trapezoid using the formula $\frac{1}{2} \times (AB + CD) \times h$.
```
**gen(Z_B)**:
```
1. Identify the given side lengths and the type of trapezoid (isosceles).  
2. Assign variables to the unknown sides based on the given information.  
3. Use the properties of isosceles trapezoids to express the legs in terms of the bases.  
4. Use the area formula for an isosceles trapezoid and the given area to form an equation.  
5. Solve the equation to find the value of the unknown base.  
6. Use the Pythagorean theorem to find the length of the other leg.
```

### s1val-00499-627066cc

**P_A**:
```
1. Recognize that the problem involves counting colorings of a $1 \times n$ grid with an even number of red squares, with the remaining squares colored blue or green.  
2. Note that the total number of unrestricted colorings is $3^n$, since each square can be colored in 3 ways.  
3. Define $f(n)$ as the number of colorings with an even number of red squares, and recognize that this is the sum of binomial coefficients $\sum_{k \text{ even}} \binom{n}{k} \cdot 2^{n-k}$.  
4. Use the identity $f(n) = \frac{3^n + 1}{2}$ to simplify the calculation of $f(n)$.  
5. Compute $f(9)$ and $f(3)$ using the formula.  
6. Divide $f(9)$ by $f(3)$ and factor the result to find the largest prime factor.
```
**gen(Z_A)**:
```
1. Recognize that the problem requires counting the number of 3×3 submatrices with exactly one black cell.  
2. Use the principle of inclusion-exclusion to count the number of 3×3 submatrices with at least one black cell.  
3. Calculate the total number of 3×3 submatrices in the grid.  
4. Subtract the number of 3×3 submatrices with at least one black cell using inclusion-exclusion.  
5. Use the formula for the number of 3×3 submatrices with exactly one black cell: $ \text{Total} - 2 \times \text{Number of 3×3 submatrices with at least one black cell} + \text{Number of 3×3 submatrices with at least two black cells} $.  
6. Simplify the expression to find the number of 3×3 submatrices with exactly one black cell.  
7. Compute the ratio of the number of such submatrices to the total number of submatrices.
```
**P_B**:
```
1. Determine the total number of colorings of a $1 \times n$ rectangle with 3 colors.  
2. Use the symmetry of the problem to relate the count of colorings with even number of red squares to those with odd number of red squares.  
3. Express $f(n)$ in terms of total colorings and a generating function that accounts for the even and odd counts of red squares.  
4. Use a known result to derive a closed-form expression for $f(n)$.  
5. Compute $f(9)$ and $f(3)$ using the derived formula.  
6. Calculate the ratio $f(9)/f(3)$.  
7. Factorize the result and identify its largest prime factor.
```
**gen(Z_B)**:
```
1. Determine the number of black and white squares in the 1×1 grid.  
2. Use the inclusion-exclusion principle to find the number of squares that are both black and white.  
3. Calculate the number of black squares that are not adjacent to any white square using the inclusion-exclusion principle.  
4. Use the inclusion-exclusion principle again to find the number of black squares that are adjacent to at least one white square.  
5. Subtract the result from step 4 from the total number of black squares to find the number of black squares that are adjacent to no white squares.  
6. Use the inclusion-exclusion principle to find the number of white squares that are adjacent to at least one black square.  
7. Subtract the result from step 6 from the total number of white squares to find the number of white squares that are adjacent to no black squares.  
8. Calculate the number of black squares that are adjacent to no white squares by subtracting the number of black squares adjacent to at least one white square from the total number of black squares.  
9. Calculate the number of white squares that are adjacent to no black squares by subtracting the number of white squares adjacent to at least one black square from the total number of white squares.  
10. Use the results from steps 8 and 9 to determine the number of black squares that are adjacent to no white squares.
```

### s1val-00689-85fb747f

**P_A**:
```
1. Recognize the structure of the equation and consider using modular arithmetic or a transformation to simplify.  
2. Define $ a = 5 + 3\sqrt{2} $ and $ b = 3 + 5\sqrt{2} $, then compute $ a \cdot b $.  
3. Observe the relationship between $ a $ and $ b $ to identify a reciprocal or conjugate-like pattern.  
4. Test small integer values of $ m $ and $ n $ to find a solution that satisfies the equation.  
5. Verify the solution by direct substitution or simplification of the equation.
```
**gen(Z_A)**:
```
1. Recognize the equation $ x^3 + y^3 = z^3 $ as a cubic Diophantine equation.  
2. Test small integer values for $ x $ and $ y $ to find potential solutions.  
3. For each candidate, compute $ z $ using the equation $ z = \sqrt[3]{x^3 + y^3} $.  
4. Check if $ z $ is an integer.  
5. Verify that the found integer solution satisfies the original equation.
```
**P_B**:
```
1. Assume both sides of the equation are positive real numbers and take the natural logarithm of both sides.  
2. Simplify the logarithmic equation by expressing it as a sum of logarithms.  
3. Recognize that the equation cannot be satisfied unless both exponents are zero, as the logarithmic expressions are not equal for any other values.  
4. Conclude that the only solution occurs when both exponents are zero.
```
**gen(Z_B)**:
```
1. Divide both sides of the equation by $ x $ to simplify the equation.  
2. Recognize that the equation involves two logarithmic terms with the same base, so equate their arguments.  
3. Simplify the resulting equation by combining like terms and factoring.  
4. Analyze the equation to determine the conditions under which the logarithmic terms can be equal.  
5. Conclude that the only solution is when the arguments of both logarithms are equal, leading to a specific value of $ x $.
```

### s1val-00472-c5e1cac9

**P_A**:
```
1. Recognize that diagonals of a rhombus bisect each other at right angles and use the given lengths to set up equations.  
2. Use the relationship between the side length $a$ and the diagonals to form a system of equations based on the given distances from point $O$ to the vertices.  
3. Solve the system of equations by expressing diagonals in terms of $a$ and substituting into the Pythagorean theorem.  
4. Simplify the resulting equation to find a quadratic equation in $a$.  
5. Solve the quadratic equation using the quadratic formula to find the value of $a$.
```
**gen(Z_A)**:
```
1. Recognize that the quadrilateral is a kite, so it has two pairs of adjacent sides equal.  
2. Use the given distances from the center of the circle to the vertices to form two equations based on the circle's radius.  
3. Set up the equations using the distance formula and solve for the side lengths.  
4. Use the Pythagorean theorem to find the length of the diagonal.  
5. Solve the resulting quadratic equation to find the value of $ x $.
```
**P_B**:
```
1. Place the rhombus on a coordinate plane with diagonal $AC$ horizontal and centered at the origin. Let $A = (-\frac{a}{2}, 0)$ and $C = (\frac{a}{2}, 0)$.  
2. Assume $O$ lies on $AC$, so its coordinates are $(x, 0)$, and use the given $OA = a$ to find $x = -a$.  
3. Assign coordinates to $B$ and $D$ as $(p, q)$ and $(p, -q)$ respectively, ensuring the rhombus is symmetric about the x-axis.  
4. Use the condition $OB = OC = 1$ to set up equations involving the coordinates of $B$.  
5. Solve the system of equations to find the value of $a$.  
6. Determine the correct answer from the options by evaluating the derived value of $a$.
```
**gen(Z_B)**:
```
1. Place the rhombus in the coordinate plane with $ A = (0, 0) $, $ B = (a, 0) $, $ D = (0, b) $, and $ C = (a, b) $.  
2. Use the given point $ P = (1, 2) $ to form equations based on the rhombus properties.  
3. Solve the equations to find the values of $ a $ and $ b $.  
4. Use the coordinates of $ P $ and the known values of $ a $ and $ b $ to determine the angle $ \angle APD $.  
5. Calculate $ \cos \angle APD $ using the coordinates of $ P $, $ A $, and $ D $.  
6. Identify the angle whose cosine is $ \frac{1}{2} $, which corresponds to $ \frac{\pi}{3} $ or $ 60^\circ $.
```

### s1val-00422-f95ff89d

**P_A**:
```
1. Identify the sequences generated by Alice and Bob: Alice counts up by 5s starting from 3, Bob counts down by 4s starting from 2021.  
2. Find the general form of Alice's sequence: $ a_n = 3 + 5n $ for non-negative integers $ n $.  
3. Find the general form of Bob's sequence: $ b_m = 2021 - 4m $ for non-negative integers $ m $.  
4. Set up the equation $ 3 + 5n = 2021 - 4m $ and solve for integer values of $ n $ and $ m $.  
5. Express $ m $ in terms of $ n $: $ m = \frac{2018 - 5n}{4} $ and determine when $ m $ is a non-negative integer.  
6. Test values of $ n $ such that $ 2018 - 5n $ is divisible by 4 and $ m \geq 0 $, ensuring the corresponding numbers fall within the range $[3, 2021]$.  
7. Count the number of valid values of $ n $ that satisfy all conditions.
```
**gen(Z_A)**:
```
1. Identify the arithmetic sequence starting at 1 with a common difference of 2, which is $1, 3, 5, 7, \ldots$.  
2. Express the condition $n \equiv 1 \pmod{2}$ as $n = 2k$ for some integer $k$.  
3. Set up the congruence $2k \equiv 1 \pmod{2}$ and solve for $k$.  
4. Determine the range of $k$ such that $2k$ is an integer and satisfies the congruence.  
5. Count the number of valid integer values of $k$ within the range.
```
**P_B**:
```
1. Identify the sequences: Alice's sequence is $3, 8, 13, 18, \ldots$ (numbers congruent to 3 modulo 5), Bob's sequence is $2021, 2017, 2013, \ldots$ (numbers congruent to 1 modulo 4).  
2. Find the intersection by solving the system of congruences $x \equiv 3 \pmod{5}$ and $x \equiv 1 \pmod{4}$.  
3. Solve the system to determine the common difference between numbers in the intersection sequence.  
4. Determine the first number in the intersection sequence that is at least 3.  
5. Identify the largest number in the intersection sequence that is less than or equal to 2021.  
6. Use the arithmetic progression formula to count how many terms are in the intersection sequence between 3 and 2021.
```
**gen(Z_B)**:
```
1. Identify the arithmetic sequence of numbers divisible by 3: $3, 6, 9, \ldots, 3k$.  
2. Identify the arithmetic sequence of numbers divisible by 4: $4, 8, 12, \ldots, 4k$.  
3. Find the least common multiple (LCM) of 3 and 4, which is 12.  
4. Determine the range of numbers from 1 to 2016 divisible by 12.  
5. Use the formula for the number of terms in the arithmetic sequence to find how many numbers are divisible by both 3 and 4.  
6. Subtract the number of terms divisible by 3 from the total number of terms divisible by 12 to find the count of numbers divisible by neither 3 nor 4.
```

### s1val-00683-4ee5b680

**P_A**:
```
1. Identify the given triangle properties: isosceles with $AB = BC = 20\ \text{cm}$, and $\tan \angle BAC = \frac{4}{3}$.  
2. Use the tangent ratio to find the sine and cosine of $\angle BAC$ using the identity $\tan \theta = \frac{\sin \theta}{\cos \theta}$.  
3. Apply the Law of Sines to relate the sides of the triangle to the angles.  
4. Calculate the semiperimeter of the triangle using its side lengths.  
5. Use the formula for the inradius $r = \frac{A}{s}$, where $A$ is the area of the triangle and $s$ is the semiperimeter.  
6. Compute the area of the triangle using the sine of one of its angles and the formula $\text{Area} = \frac{1}{2}ab\sin C$.  
7. Substitute the known values into the inradius formula to find $r$.
```
**gen(Z_A)**:
```
1. Identify the triangle type (isosceles, scalene, right) and determine the angles.  
2. Use the Law of Cosines to find one angle in terms of the given side $ a $.  
3. Apply the Law of Cosines again to find the other angles.  
4. Use the formula for the area of a triangle with given side lengths and two angles: $ \text{Area} = \frac{1}{2}ab\sin C $.  
5. Calculate the area using the known side lengths and the calculated angles.  
6. Use the area to find the inradius using the formula $ r = \frac{A}{s} $, where $ s $ is the semi-perimeter.  
7. Use the inradius formula $ r = \frac{a + b + c}{2} $ to find the inradius.
```
**P_B**:
```
1. Recognize that triangle $ \triangle ABC $ is isosceles with $ AB = BC = 20 \, \text{cm} $ and use the given tangent value $ \tan(\angle BAC) = \frac{4}{3} $ to find the height from $ B $ to $ AC $.  
2. Set up a right triangle with opposite side 4 and adjacent side 3, and use the hypotenuse to find the proportionality factor.  
3. Use the proportionality factor to determine the height of the isosceles triangle.  
4. Find the base $ AC $ by using the height and the angle, or directly from the triangle's geometry.  
5. Calculate the semi-perimeter $ s $ of the triangle using the side lengths.  
6. Apply the formula for the radius of the inscribed circle $ r = \frac{A}{s} $, where $ A $ is the area of the triangle and $ s $ is the semi-perimeter.
```
**gen(Z_B)**:
```
1. Recognize that the triangle is isosceles with the given angle $ \angle BAC = 120^\circ $ and base angles $ \angle ABC = \angle ACB = 30^\circ $.  
2. Use the Law of Cosines to find the length of the base $ BC $ in terms of the equal sides $ AB = AC = x $.  
3. Apply the Law of Sines to relate the sides and angles, specifically using $ \sin 30^\circ $ to find the base $ BC $.  
4. Use the area formula for a triangle with given angles and sides to calculate the area of triangle $ ABC $.  
5. Apply the Law of Cosines again to find the height $ h $ from vertex $ A $ to side $ BC $.
```

### s1val-00626-1bba07b5

**P_A**:
```
1. Use the formula for the sum of numbers less than $ n $ that are coprime with $ n $: $ \frac{n \cdot \phi(n)}{2} $.  
2. Calculate $ \phi(200) $ by factoring 200 and applying the Euler's totient function formula.  
3. Compute $ \phi(200) = 80 $.  
4. Substitute $ \phi(200) $ into the formula to find the sum: $ \frac{200 \cdot 80}{2} $.
```
**gen(Z_A)**:
```
1. Use the formula for the sum of the first $ n $ positive integers: $ S = \frac{n(n+1)}{2} $.  
2. Substitute $ n = 2016 $ into the formula to find the sum of all positive integers less than or equal to 2016.  
3. Subtract 1 from the result to get the sum of all positive integers strictly less than 2016.
```
**P_B**:
```
1. Factorize 200 to identify its prime factors: $200 = 2^3 \cdot 5^2$.  
2. Apply the inclusion-exclusion principle to count numbers less than 200 that are not coprime with 200.  
3. Subtract the count of non-coprime numbers from 199 to find the count of coprime numbers.  
4. Use the formula for the sum of numbers less than $n$ and coprime with $n$: $\sum_{k=1}^{n-1} \gcd(k, n) = \frac{n \cdot \phi(n)}{2}$.  
5. Calculate $\phi(200)$ using Euler's totient function.  
6. Substitute the value of $\phi(200)$ into the formula to compute the required sum.
```
**gen(Z_B)**:
```
1. Calculate the total number of positive integers less than or equal to 2008 by subtracting 1 from 2008 and adding 1.  
2. Use the formula for the sum of the first $ n $ positive integers to find the sum of all divisors of 2008.  
3. Subtract the sum of divisors of 2007 from the total sum of divisors of 2008 to get the sum of divisors of 2008.  
4. Use the formula for the sum of divisors of a prime number to find the sum of divisors of 2009.  
5. Subtract the sum of divisors of 2009 from the sum of divisors of 2008 to get the final result.
```

### s1val-00496-03739fc1

**P_A**:
```
1. Recognize the expression $ mx + ny $ as a dot product of vectors $ \vec{u} = (m, n) $ and $ \vec{v} = (x, y) $.  
2. Use the identity that the maximum value of the dot product $ \vec{u} \cdot \vec{v} $ is $ \|\vec{u}\| \|\vec{v}\| $ when the vectors are in the same direction.  
3. Substitute $ \|\vec{u}\| = \sqrt{a} $ and $ \|\vec{v}\| = \sqrt{b} $ into the identity to find the maximum value $ \sqrt{ab} $.  
4. Compare the result with the given options and select the correct one.
```
**gen(Z_A)**:
```
1. Recognize the expression as a sum of products of variables, each squared.  
2. Use the identity $(a + b + c)^2 = a^2 + b^2 + c^2 + 2ab + 2ac + 2bc$ to rewrite the expression.  
3. Identify the terms to be maximized: $x^2 + y^2 + z^2$ and $2xy + 2xz + 2yz$.  
4. Assume symmetry by setting $x = y = z = 1$ to maximize the symmetric terms.  
5. Calculate the value of the expression with $x = y = z = 1$.  
6. Compare the result with the given options to determine the correct answer.
```
**P_B**:
```
1. Use the Cauchy-Schwarz inequality in the form $(mx + ny)^2 \leq (m^2 + n^2)(x^2 + y^2)$.  
2. Substitute the given values $m^2 + n^2 = a$ and $x^2 + y^2 = b$ into the inequality.  
3. Simplify the resulting expression to find an upper bound for $mx + ny$.  
4. Recognize that equality in Cauchy-Schwarz is achieved when the vectors $(m, n)$ and $(x, y)$ are proportional, confirming the maximum value is attainable.
```
**gen(Z_B)**:
```
1. Use the Cauchy-Schwarz Inequality to relate $ x + y $ and $ xy $.  
2. Express $ x + y $ and $ xy $ in terms of $ a $ and $ b $ using the given equations.  
3. Substitute these expressions into the inequality $ (x + y)^2 \leq a^2 + b^2 $.  
4. Simplify the resulting inequality to find the maximum value of $ a^2 + b^2 $.
```

### s1val-00344-ba8a76c3

**P_A**:
```
1. Rewrite the equation as $2^{x+1} + x^2 - 2 = 0$ to express it as a function $f(x)$.  
2. Identify the domain of the function based on the expression $2^{x+1} + x^2 - 2$.  
3. Choose a value of $x$ and evaluate the function to determine if it equals zero.  
4. Solve the equation for $x$ by testing rational or integer values to find a solution.  
5. Confirm that the solution satisfies the original equation.
```
**gen(Z_A)**:
```
1. Substitute $ x = 0 $ into the equation to check for potential solutions.  
2. Substitute $ x = 1 $ into the equation to check for potential solutions.  
3. Substitute $ x = -1 $ into the equation to check for potential solutions.  
4. Analyze the behavior of the function $ f(x) = x^3 - 2x + 1 $ to determine if it ever equals 0.  
5. Use numerical methods or graphing to approximate the real root.
```
**P_B**:
```
1. Rewrite the equation as two separate functions: $ f(x) = 2^{x+1} $ and $ g(x) = -x^2 + 2 $.  
2. Graph both functions on the same coordinate plane.  
3. Identify the x-values where the two graphs intersect.  
4. List the intersection points as the solutions to the original equation.
```
**gen(Z_B)**:
```
1. Set the two equations equal to each other to find the points of intersection.  
2. Solve the resulting equation for $ x $, then substitute back to find the corresponding $ y $-value.  
3. Identify the coordinates of the intersection points.
```

### s1val-00660-ef03132b

**P_A**:
```
1. Factor the left-hand side of the equation by grouping terms.  
2. Add and subtract a term to complete the square for the expression involving $ x $.  
3. Simplify the equation to a perfect square form.  
4. Express the equation as a difference of squares.  
5. Solve for $ (x + y) $ and $ (x - y - 1) $ by taking square roots.  
6. Set up the two equations derived from the square roots.  
7. Solve the system of equations to find values of $ x $ and $ y $.  
8. Check if the solutions satisfy the original equation.
```
**gen(Z_A)**:
```
1. Factor the left-hand side of the equation to simplify the expression.  
2. Recognize that the equation can be rewritten as a perfect square.  
3. Set the factored form equal to zero and solve for $ x $, considering both positive and negative roots.  
4. Substitute each value of $ x $ into the original equation to check for solutions.  
5. Solve the resulting equations to find valid solutions.  
6. Check for extraneous solutions by substituting back into the original equation.
```
**P_B**:
```
1. Factor the equation by grouping terms to identify a common structure.  
2. Introduce a substitution $ z = y - 1 $ to simplify the equation and express it in terms of $ x $ and $ z $.  
3. Solve for $ x $ in terms of $ z $ and identify the condition that $ z(z + 1) $ must divide 95.  
4. Find all positive integer factor pairs of 95 and test them to determine which pair results in a natural number $ x $.  
5. Use the valid pair to calculate $ y $ and $ x $, ensuring all values are natural numbers.
```
**gen(Z_B)**:
```
1. Factor the numerator $ n^2 + 1 $ as $ (n + i)(n - i) $, where $ i $ is the imaginary unit.  
2. Express the denominator $ n^2 + 1 $ as $ (n + i)(n - i) $, and rewrite the fraction with the factored form.  
3. Multiply both sides of the equation by the denominator to eliminate the fraction.  
4. Simplify the resulting equation to obtain a polynomial in $ n $.  
5. Test small integer values of $ n $ to find solutions that satisfy the equation.  
6. Identify all integer solutions and check if they are perfect cubes.
```

### s1val-00368-f1e220c8

**P_A**:
```
1. Identify the hyperbola parameters: determine $ a $, $ b $, $ c $, and the coordinates of the right focus $ F $.  
2. Write the equation of the line passing through $ P(3,0) $ with slope $ k $: $ y = k(x - 3) $.  
3. Substitute the line equation into the hyperbola equation to form a quadratic in $ x $.  
4. Ensure the discriminant of the quadratic is positive to confirm the line intersects the right branch of the hyperbola at two points.  
5. Use the coordinates of $ A $ and $ B $ to compute $ |AF| + |BF| $, then set this equal to 16 and solve for $ k $.  
6. Determine the valid value of $ k $ that satisfies the condition $ |AF| + |BF| = 16 $.
```
**gen(Z_A)**:
```
1. Identify the foci of the hyperbola and the directrices, then determine the condition for the line $ y = kx + c $ to be tangent to the hyperbola.  
2. Use the condition for tangency to set up an equation involving $ k $ and $ c $.  
3. Substitute $ c = 1 $ (since the directrix is $ x = 1 $) into the equation and simplify to find the value of $ k $.  
4. Determine the range of $ k $ for which the line is tangent to the hyperbola.  
5. Check which of the given options correspond to the valid range of $ k $.
```
**P_B**:
```
1. Identify the hyperbola equation and determine its parameters: $ a = 1 $, $ b^2 = 3 $, $ c^2 = a^2 + b^2 = 4 $, so $ c = 2 $.  
2. Use the hyperbola’s definition: the difference of distances from any point on the hyperbola to the two foci is $ 2a = 2 $.  
3. Express the condition $ |AF| + |BF| = 16 $ using the relationship between distances and the hyperbola’s foci.  
4. Simplify the equation using algebraic manipulation to relate $ |AF| $ and $ |BF| $ to the coordinates of points $ A $ and $ B $.  
5. Substitute the line equation into the hyperbola equation to find the coordinates of points $ A $ and $ B $.  
6. Calculate $ |AF| + |BF| $ using the distance formula and set it equal to 16.  
7. Solve the resulting equation to find the value of $ k $.
```
**gen(Z_B)**:
```
1. Identify the foci of the hyperbola and the directrices, noting that $ c = 1 $ and $ a = 1 $.  
2. Use the given condition $ |PF_1| + |PF_2| = 2 $ to set up an equation involving the coordinates of point $ P $ on the hyperbola.  
3. Express the coordinates of point $ P $ in terms of $ x $ using the hyperbola equation $ x^2 - y^2 = 1 $.  
4. Substitute the coordinates of $ P $ into the equation $ |PF_1| + |PF_2| = 2 $ and simplify.  
5. Solve the resulting equation to find the value of $ x $.  
6. Use the value of $ x $ to determine the coordinates of point $ P $, then compute $ |PF_1| $ and $ |PF_2| $.
```

### s1val-00535-02813895

**P_A**:
```
1. Use the formula for area of a triangle in terms of two sides and the included angle: $\text{Area} = \frac{1}{2}ab\sin\theta$.  
2. Identify the known values: area = 30, side $a = 10$, median $b = 9$, and substitute into the formula.  
3. Solve for $\sin\theta$ using the equation $\frac{1}{2} \cdot 10 \cdot 9 \cdot \sin\theta = 30$.
```
**gen(Z_A)**:
```
1. Use the formula for the area of a triangle with given sides and included angle: $ \text{Area} = \frac{1}{2}ab\sin C $.  
2. Identify the sides and included angle from the given information.  
3. Substitute the values into the formula and solve for the area.
```
**P_B**:
```
1. Place the triangle on a coordinate plane with the side of length 10 along the x-axis and its midpoint at the origin.  
2. Use the area formula to express the height of the triangle in terms of $ \sin\theta $.  
3. Apply the formula for the length of the median in terms of $ \sin\theta $ and solve for $ \sin\theta $.
```
**gen(Z_B)**:
```
1. Place the triangle in a coordinate system with the base $ BC $ on the x-axis and the vertex $ A $ at the top.  
2. Assign coordinates to points $ B $, $ C $, and $ A $ based on the given side lengths.  
3. Use the area formula for a triangle with base and height to find the height from $ A $ to the base $ BC $.  
4. Apply the Law of Cosines to find the angle $ \angle BAC $ using the known side lengths.
```

### s1val-00491-740fb9ec

**P_A**:
```
1. Place the triangle in a coordinate system with $ D $ at the origin and $ B $ on the x-axis.  
2. Express vectors $ \vec{AB} $, $ \vec{AC} $, and $ \vec{AD} $ in terms of coordinates.  
3. Use the midpoint formula to find the coordinates of $ D $.  
4. Express vector $ \vec{AO} $ in terms of vectors $ \vec{AB} $ and $ \vec{AC} $.  
5. Compute the dot product $ \vec{AO} \cdot \vec{AD} $ using the expression for $ \vec{AO} $.  
6. Use the given dot product value and the length of $ \vec{AD} $ to solve for $ AD $.
```
**gen(Z_A)**:
```
1. Assign coordinates to points $ A, B, C, D $ based on the given distances and angles.  
2. Express vectors $ \vec{AB}, \vec{AC}, \vec{AD} $ in terms of coordinates.  
3. Use the dot product formula to find the cosine of the angle $ \angle ACB $.  
4. Calculate the magnitudes of the vectors $ \vec{AB} $ and $ \vec{AC} $.  
5. Substitute the cosine value into the dot product formula to find the cosine of the angle $ \angle ACB $.
```
**P_B**:
```
1. Recognize that $ D $ is the midpoint of $ BC $, so $ \vec{AD} = \frac{1}{2} \vec{AB} + \frac{1}{2} \vec{AC} $ and $ \vec{OD} = \frac{1}{2} \vec{BC} $.  
2. Express $ \vec{AO} $ in terms of $ \vec{OD} $ and $ \vec{AD} $, then simplify using vector properties.  
3. Compute the dot product $ \vec{AO} \cdot \vec{AD} $ using the simplified expressions and the given value $ \vec{AO} \cdot \vec{AD} = 4 $.  
4. Solve for $ |\vec{AD}| $ using the dot product identity $ |\vec{AO}||\vec{AD}| \cos \theta = 4 $, and substitute the known value of $ |\vec{BC}| $.  
5. Simplify the expression to find the value of $ AD $.
```
**gen(Z_B)**:
```
1. Recognize that $ \vec{OA} + \vec{OB} + \vec{OC} = \vec{O} $, where $ O $ is the origin.  
2. Express $ \vec{OA} $, $ \vec{OB} $, and $ \vec{OC} $ in terms of position vectors of points $ A $, $ B $, and $ C $ relative to the origin.  
3. Use the given condition $ \vec{OA} \cdot \vec{OB} = \vec{OC} \cdot \vec{OA} $ to relate the dot products of the position vectors.  
4. Simplify the dot product expressions using vector identities and properties of dot products.  
5. Substitute the simplified expressions into the given equation and solve for $ \vec{OC} \cdot \vec{OA} $.  
6. Use the result to find $ \vec{OC} \cdot \vec{OA} $, then compute $ \vec{OC} \cdot \vec{OA} + \vec{OA} \cdot \vec{OC} $.
```

### s1val-00653-bc3c49e7

**P_A**:
```
1. Use the Pythagorean Theorem $ a^2 + b^2 = c^2 $ with $ a = 11 $ to express the equation in terms of $ b $ and $ c $.  
2. Rearrange the equation to $ c^2 - b^2 = 121 $ and factor it as $ (c - b)(c + b) = 121 $.  
3. List the positive integer factor pairs of 121 and test them to find valid values for $ c - b $ and $ c + b $.  
4. Solve the system of equations for each valid pair to find corresponding $ b $ and $ c $.  
5. Check if the resulting $ b $ and $ c $ are natural numbers and verify that the triangle is right-angled.  
6. Calculate the perimeter $ a + b + c $ for valid solutions.  
7. Compare the result with the given options to determine the correct answer.
```
**gen(Z_A)**:
```
1. Use the Pythagorean Theorem to express $ b $ in terms of $ a $ and $ c $, then substitute into the given equation.  
2. Simplify the resulting equation to form a quadratic in $ a $.  
3. Solve the quadratic equation to find possible integer values for $ a $.  
4. For each valid $ a $, calculate $ b $ and $ c $ using the Pythagorean Theorem.  
5. Check which of the given options match the calculated values of $ b $ and $ c $.  
6. Identify the correct option based on the matches.
```
**P_B**:
```
1. Identify the right-angled triangle with one leg = 11 and the other two sides as natural numbers.  
2. Use the Pythagorean Theorem $ a^2 + b^2 = c^2 $, where $ a = 11 $, and $ b $ and $ c $ are natural numbers.  
3. Test integer values for $ b $ to find a corresponding integer $ c $ that satisfies the equation.  
4. Calculate the perimeter using $ 11 + b + c $ for the valid case.  
5. Compare the calculated perimeter with the given options.
```
**gen(Z_B)**:
```
1. Identify the three integers as the legs of a right triangle and the hypotenuse.  
2. List all possible triplets of three consecutive natural numbers that sum to 20.  
3. For each triplet, calculate the sum of the squares of the two smaller numbers and check if the result is a perfect square.  
4. If a valid triplet is found, calculate the hypotenuse using the Pythagorean theorem.  
5. Compare the hypotenuse with the given options to find the correct one.  
6. Select the triplet that matches the given options.
```

### s1val-00281-3cd770ea

**P_A**:
```
1. Use the given equation $ a^2 + ab + b^2 = 1 $ to express $ t $ in terms of the equation.  
2. Recognize that $ t = -1 $ is directly obtainable by substituting the given equation into the expression for $ t $.  
3. Substitute $ a = r \cos \theta $ and $ b = r \sin \theta $ into the constraint equation to find the range of $ r $.  
4. Express $ t $ in terms of $ r $ and $ \theta $, then simplify using the constraint equation.  
5. Use the range of $ \theta $ and the found range of $ r $ to determine the maximum and minimum values of $ t $.
```
**gen(Z_A)**:
```
1. Use the given condition $ a + b + c = 1 $ to express $ c $ in terms of $ a $ and $ b $, then substitute into the equation $ a^3 + b^3 + c^3 = 1 $.  
2. Simplify the equation by substituting $ c = 1 - a - b $ and using the identity $ a^3 + b^3 = (a + b)^3 - 3ab(a + b) $.  
3. Express the equation in terms of $ a + b $ and $ ab $, then use the constraint $ a + b = 1 - c $ to find the range of possible values for $ ab $.  
4. Determine the range of possible values for $ c $ by analyzing the resulting expressions.
```
**P_B**:
```
1. Use the given equation $ a^2 + ab + b^2 = 1 $ to express $ a^2 + b^2 $ as $ 1 - ab $.  
2. Substitute $ a^2 + b^2 $ into the expression for $ t $ to get $ t = -2ab + 1 $.  
3. Let $ x = ab $, then express $ t $ in terms of $ x $ as $ t = -2x + 1 $.  
4. Determine the possible values of $ x = ab $ under the constraint $ a^2 + ab + b^2 = 1 $.  
5. Solve the quadratic inequality in $ x $ to find the valid range of $ x $.  
6. Substitute the bounds of $ x $ into $ t = -2x + 1 $ to find the corresponding range of $ t $.
```
**gen(Z_B)**:
```
1. Use the given equation $ x^2 + y^2 = 1 $ to express $ y^2 $ in terms of $ x $.  
2. Substitute $ y^2 = 1 - x^2 $ into the equation $ x^2 + y^2 = 1 $ to eliminate $ y^2 $.  
3. Simplify the resulting equation to find $ x^2 $.  
4. Determine the range of $ x $ by analyzing the simplified equation.  
5. Substitute the range of $ x $ back into the expression for $ y^2 $ to find the corresponding range of $ y^2 $.
```

### s1val-00439-e679617d

**P_A**:
```
1. Assign coordinates to the square vertices based on a unit square with side length 1.  
2. Determine the coordinates of point $ K $ on side $ CD $ using the given ratio $ CK:KD = 1:2 $.  
3. Calculate the vector $ \vec{AK} $ and use the formula for the distance from a point to a line in vector form.  
4. Substitute the coordinates of point $ C $ and the vector $ \vec{AK} $ into the distance formula.  
5. Simplify the expression to find the final distance.
```
**gen(Z_A)**:
```
1. Assign coordinates to points $ A, B, C, D $ based on the given ratios.  
2. Determine the coordinates of point $ P $ on $ AD $ such that $ \frac{AP}{PD} = \frac{1}{2} $.  
3. Use the determinant formula for the area of a triangle with vertices $ (x_1, y_1), (x_2, y_2), (x_3, y_3) $ to compute the area of $ \triangle PBC $.  
4. Simplify the determinant expression to find the area of $ \triangle PBC $.
```
**P_B**:
```
1. Assign coordinates to the square vertices based on the side length of 1.  
2. Determine the coordinates of point $ K $ on side $ CD $ using the given ratio $ CK:KD = 1:2 $.  
3. Find the equation of line $ AK $ by calculating its slope and using the point-slope form.  
4. Use the point-to-line distance formula to calculate the perpendicular distance from point $ C $ to line $ AK $.
```
**gen(Z_B)**:
```
1. Assign coordinates to points $ A, B, C, D $ based on the given proportions and place $ A $ at the origin.  
2. Determine the coordinates of point $ E $, the intersection of $ AD $ and $ BC $.  
3. Use the coordinates of points $ A, B, C, D, E $ to calculate the area of triangle $ ABC $ using the determinant method.  
4. Use the coordinates of points $ A, B, C, E $ to calculate the area of triangle $ ABE $ using the determinant method.
```

### s1val-00498-50ca6d71

**P_A**:
```
1. Recognize the sequence follows the Fibonacci recurrence relation with initial terms $ a_1 = 1 $, $ a_2 = x $, and $ a_{n} = a_{n-1} + a_{n-2} $.  
2. Calculate the sequence up to the 10th term in terms of $ x $, and check when 463 appears as one of the terms.  
3. Identify all values of $ x $ for which 463 is a term in the sequence.  
4. Sum all valid $ x $ values.
```
**gen(Z_A)**:
```
1. Recognize the sequence as a geometric progression with first term 1 and common ratio 2.  
2. Use the formula for the sum of the first $ n $ terms of a geometric series: $ S_n = \frac{1 - 2^n}{1 - 2} = 2^n - 1 $.  
3. Set $ S_n = 2013 $ and solve for $ n $, then check if $ n $ is an integer.  
4. Identify all valid integer values of $ n $ that satisfy the equation.  
5. For each valid $ n $, compute $ a_n $ using the formula $ a_n = \frac{1}{2^n} $.  
6. Collect all valid values of $ a_n $ and sum them.
```
**P_B**:
```
1. Define the sequence with $ a_1 = 1 $, $ a_2 = x $, and $ a_n = a_{n-1} + a_{n-2} $ for $ n \geq 3 $.  
2. Express the general term $ a_n $ in terms of $ x $ and powers of the golden ratio $ \phi $ using the closed-form formula.  
3. Set up the equation $ a_n = 463 $ and solve for $ x $ in terms of $ \phi $.  
4. Use the matrix exponentiation method to compute $ a_n $ for $ n = 1 $ to $ 10 $ and solve for $ x $ in each case.  
5. Collect all positive integer values of $ x $ for which 463 appears in the sequence.  
6. Sum all such values of $ x $.
```
**gen(Z_B)**:
```
1. Express the sequence $ a_n $ in terms of $ a_1 $ and $ a_2 $ using the recurrence relation.  
2. Use the given $ a_1 = 1 $ and $ a_2 = 2 $ to compute the first few terms of the sequence.  
3. For each term, compute the sum of the squares of the previous two terms.  
4. Set up the equation $ a_n = \frac{1}{2} \left( a_{n-1}^2 + a_{n-2}^2 \right) $ and solve for $ a_n $ in terms of $ a_1 $ and $ a_2 $.  
5. Identify all integer values of $ a_n $ that satisfy the equation for some integer $ n \geq 3 $.  
6. List all such integer values of $ a_n $.
```

### s1val-00313-ab57b867

**P_A**:
```
1. Recognize the constraint $ x^2 + y^2 + z^2 = 1 $ and the function to minimize $ \frac{x}{1 - x^2} + \frac{y}{1 - y^2} + \frac{z}{1 - z^2} $.  
2. Assume symmetry by setting $ x = y = z $ and verify it satisfies the constraint.  
3. Substitute $ x = y = z = \frac{1}{\sqrt{3}} $ into the function and compute the value.  
4. Use the inequality $ \frac{t}{1 - t^2} \geq 3t $ for $ t \in (0, 1) $ to bound the function below by $ 3(x + y + z) $.  
5. Apply the Cauchy-Schwarz inequality to bound $ x + y + z $ under the given constraint.  
6. Conclude that the minimum value is achieved when $ x = y = z = \frac{1}{\sqrt{3}} $, and the minimum value is $ \sqrt{3} $.
```
**gen(Z_A)**:
```
1. Recognize the constraint $ x^3 + y^3 + z^3 = 1 $ and the target function $ f(x, y, z) = \frac{1}{x} + \frac{1}{y} + \frac{1}{z} $.  
2. Use the method of Lagrange multipliers to find critical points by setting up the system of equations $ \nabla f = \lambda \nabla g $, where $ g(x, y, z) = x^3 + y^3 + z^3 - 1 $.  
3. Simplify the Lagrange equations to find that the critical point occurs when $ x = y = z = 1 $.  
4. Evaluate the function $ f(x, y, z) $ at the critical point $ (1, 1, 1) $.  
5. Consider the behavior of the function when one variable approaches 0 and the other two adjust to maintain the constraint, and evaluate the limit as one variable approaches 0.  
6. Conclude that the maximum value of $ f(x, y, z) $ is achieved at $ (1, 1, 1) $ and equals 3.
```
**P_B**:
```
1. Recognize that the constraint is $ x^2 + y^2 + z^2 = 1 $, which bounds the values of $ x, y, z $ in the unit sphere.  
2. Analyze the function $ f(a) = \frac{a}{1 - a^2} $ for $ a \in (0, 1) $ and note its monotonicity and concavity.  
3. Determine that increasing $ x, y, z $ while keeping their squares sum to 1 decreases the overall value of the expression, suggesting a minimum occurs when one variable is as large as possible.  
4. Test the symmetric case $ x = y = z = \frac{1}{\sqrt{3}} $ to compute the value of the expression.  
5. Compare with the case where one variable is maximized (e.g., $ x = 1, y = z = 0 $) to confirm that the expression achieves its minimum in the symmetric configuration.
```
**gen(Z_B)**:
```
1. Recognize that the function $ f(x, y) = \frac{x^3 + y^3}{x^2 + y^2} $ is symmetric in $ x $ and $ y $, so assume $ x = y $ without loss of generality.  
2. Substitute $ x = y $ into the function to simplify it to $ \frac{2x^3}{2x^2} = x $, and find the minimum value of $ x $ under the constraint $ x^2 + y^2 \leq 1 $.  
3. Use Lagrange multipliers to find the critical point of the function $ f(x, y) $ under the constraint $ x^2 + y^2 = 1 $.  
4. Evaluate the function at the critical point to determine the minimum value.
```

### s1val-00471-2af60d16

**P_A**:
```
1. Recognize that the circle is tangent to the y-axis at (0, 2), so its center has an x-coordinate equal to the radius $ r $, and the y-coordinate is 2.  
2. Use the given x-intercept (8, 0) and the center (r, 2) to apply the distance formula for radius.  
3. Solve the resulting equation to find the value of $ r $.
```
**gen(Z_A)**:
```
1. Recognize that the circle is tangent to the x-axis at (0, 0) and has center (h, k).  
2. Use the condition for tangency to the x-axis: the y-coordinate of the center must be equal to the radius.  
3. Set up the equation $ k = r $ and substitute $ r = h^2 + k^2 $ to form a quadratic equation in $ h $.  
4. Solve the quadratic equation to find the value of $ h $.
```
**P_B**:
```
1. Recognize that the circle is tangent to the y-axis at (0, 2), so the center is at (r, 2) where r is the radius.  
2. Use the fact that the larger x-intercept is 8 to write the equation of the circle in standard form.  
3. Substitute the point (8, 0) into the circle's equation and simplify to form a quadratic equation in terms of r.  
4. Solve the quadratic equation using the quadratic formula and select the positive solution for r.
```
**gen(Z_B)**:
```
1. Recognize that the circle is tangent to the x-axis and the line $ y = x $, so its center lies on the line $ y = x $ and its radius is equal to the y-coordinate of the center.  
2. Use the distance from the center to the line $ y = x $ to set up an equation and solve for the radius.  
3. Substitute the radius into the equation of the circle and solve the resulting quadratic equation.  
4. Solve the quadratic equation to find the possible values of $ x $, then determine the corresponding $ y $ values.
```

