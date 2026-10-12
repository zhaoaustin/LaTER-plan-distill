### s1val-00626_A

```
1. Use the formula for the sum of numbers less than $ n $ that are coprime with $ n $: $ \frac{n \cdot \phi(n)}{2} $.  
2. Calculate $ \phi(200) $ by factoring 200 and applying the Euler's totient function formula.  
3. Compute $ \phi(200) = 80 $.  
4. Substitute $ \phi(200) $ into the formula to find the sum: $ \frac{200 \cdot 80}{2} $.
```

### s1val-00626_A'

```
1. Recall that the integers below $n$ that are relatively prime to $n$ add up to $\frac{n\,\phi(n)}{2}$.  
2. Find Euler's totient $\phi(200)$ from the prime factorization of 200 together with the totient product formula.  
3. This gives $\phi(200) = 80$.  
4. Plug this value into the sum formula and evaluate $\frac{200 \cdot 80}{2}$.
```

### s1val-00626_B

```
1. Factorize 200 to identify its prime factors: $200 = 2^3 \cdot 5^2$.  
2. Apply the inclusion-exclusion principle to count numbers less than 200 that are not coprime with 200.  
3. Subtract the count of non-coprime numbers from 199 to find the count of coprime numbers.  
4. Use the formula for the sum of numbers less than $n$ and coprime with $n$: $\sum_{k=1}^{n-1} \gcd(k, n) = \frac{n \cdot \phi(n)}{2}$.  
5. Calculate $\phi(200)$ using Euler's totient function.  
6. Substitute the value of $\phi(200)$ into the formula to compute the required sum.
```

### s1val-00626_B'

```
1. Write 200 as a product of primes: $200 = 2^3 \cdot 5^2$.  
2. Count how many integers below 200 share a common factor with 200, using the inclusion-exclusion principle.  
3. Remove that count from 199 to get how many integers below 200 are relatively prime to 200.  
4. Then apply the sum formula for integers below $n$ that are coprime to $n$: $\sum_{k=1}^{n-1} \gcd(k, n) = \frac{n \cdot \phi(n)}{2}$.  
5. Evaluate Euler's totient $\phi(200)$.  
6. Insert $\phi(200)$ into this formula to obtain the requested sum.
```

### s1val-00496_A

```
1. Recognize the expression $ mx + ny $ as a dot product of vectors $ \vec{u} = (m, n) $ and $ \vec{v} = (x, y) $.  
2. Use the identity that the maximum value of the dot product $ \vec{u} \cdot \vec{v} $ is $ \|\vec{u}\| \|\vec{v}\| $ when the vectors are in the same direction.  
3. Substitute $ \|\vec{u}\| = \sqrt{a} $ and $ \|\vec{v}\| = \sqrt{b} $ into the identity to find the maximum value $ \sqrt{ab} $.  
4. Compare the result with the given options and select the correct one.
```

### s1val-00496_A'

```
1. View $mx + ny$ as the inner product of the vectors $\vec{u} = (m, n)$ and $\vec{v} = (x, y)$.  
2. An inner product $\vec{u} \cdot \vec{v}$ is largest, equal to $\|\vec{u}\| \|\vec{v}\|$, when the two vectors point in the same direction.  
3. Since $\|\vec{u}\| = \sqrt{a}$ and $\|\vec{v}\| = \sqrt{b}$, this largest value is $\sqrt{ab}$.  
4. Match this value against the answer choices and pick the right option.
```

### s1val-00496_B

```
1. Use the Cauchy-Schwarz inequality in the form $(mx + ny)^2 \leq (m^2 + n^2)(x^2 + y^2)$.  
2. Substitute the given values $m^2 + n^2 = a$ and $x^2 + y^2 = b$ into the inequality.  
3. Simplify the resulting expression to find an upper bound for $mx + ny$.  
4. Recognize that equality in Cauchy-Schwarz is achieved when the vectors $(m, n)$ and $(x, y)$ are proportional, confirming the maximum value is attainable.
```

### s1val-00496_B'

```
1. Apply the Cauchy-Schwarz inequality $(mx + ny)^2 \leq (m^2 + n^2)(x^2 + y^2)$.  
2. Replace $m^2 + n^2$ by $a$ and $x^2 + y^2$ by $b$ in this inequality.  
3. Simplify to obtain an upper bound on $mx + ny$.  
4. Note that Cauchy-Schwarz becomes an equality when $(m, n)$ and $(x, y)$ are proportional, so the bound is actually reached.
```

