:::: columns 300 1fr
::: column
::: rope_dial
:::
:::

::: column
::: lead
Attention 有一个奇怪的缺陷：它不知道词的顺序。旋转位置编码（Rotary Position Embedding，RoPE）就是用来补这个缺陷的。下面用这一页的例子，一步步说明它怎么做。
:::

### 一、问题：attention 不认识顺序

先看一个例子。这一页的输入是 “the cat sat”。轮到 “sat” 打分时，它要和 “the”、“cat”、自己分别算一次点积 $\mathbf{q}\cdot\mathbf{k}$。

点积只看两个向量的内容。也就是说，把 “the” 和 “cat” 对调位置，“sat” 给它们的分数不会有任何变化。

这显然不对。离得近的词和离得远的词，往往应该区别对待。因此，必须想办法把位置告诉模型。

### 二、思路：不加，而是转

最早的 Transformer 给每个位置加一个位置向量（position embedding）。RoPE 换了一个办法：不加，而是转。

具体来说，位置 $t$ 的 q 转 $t\omega$ 的角度，位置 $s$ 的 k 转 $s\omega$ 的角度。

看到这里，你可能会问：转一下有什么用？关键在于两个向量都转了。q 转了 $t\omega$，k 转了 $s\omega$，它们之间的夹角只改变 $(t-s)\omega$。而点积只取决于长度和夹角，所以分数只跟距离 $t-s$ 有关。

左图可以验证这一点。把 $t$ 和 $s$ 同时加 1，两个箭头一起转，点积不变。

### 三、计算：两两配对，各转各的

二维向量很容易转。但 q 有 $d_h$ 维，怎么转？

办法是两两配对。第 $c$ 个和第 $c+d_h/2$ 个分量组成一对，每一对在自己的平面里转。对一对来说，就是一个 $2\times2$ 的旋转矩阵。

$$\begin{pmatrix}q'_c\\ q'_{c+d_h/2}\end{pmatrix}=\begin{pmatrix}\cos t\omega_c & -\sin t\omega_c\\ \sin t\omega_c & \cos t\omega_c\end{pmatrix}\begin{pmatrix}\bar q_c\\ \bar q_{c+d_h/2}\end{pmatrix}$$

代码里却没有这个矩阵，写的是 $\bar{\mathbf{q}}\odot\cos+\mathrm{rotate\_half}(\bar{\mathbf{q}})\odot\sin$。其实两者是一回事。rotate_half 把一对 $(\bar q_c,\bar q_{c+d_h/2})$ 变成 $(-\bar q_{c+d_h/2},\bar q_c)$，代进去展开，正好是矩阵乘法的两行。

$$q'_c=\bar q_c\cos t\omega_c-\bar q_{c+d_h/2}\sin t\omega_c,\qquad q'_{c+d_h/2}=\bar q_{c+d_h/2}\cos t\omega_c+\bar q_c\sin t\omega_c$$

因此，3D 图里的四步（乘 cos、rotate_half、乘 sin、相加），合起来就是一次旋转。
:::
::::

:::: columns 340 1fr
::: column
::: rope_curve
:::
:::

::: column
### 四、为什么要多个频率

旋转有一个问题：它是周期的。转满一圈，就回到原处。

比如 $\omega=1$ 时，周期约 6.3 个位置。距离为 0 和距离为 6 的两个词，在这一对分量上几乎分不开。

这就需要多个频率。每一对用不同的转速，$\omega_i=\theta^{-2i/d_h}$。转得快的对，对近距离敏感；转得慢的对，能分辨远距离。几组合起来，就像同时看秒针、分针和时针。

左图是整个 head 的点积随 $t-s$ 的变化。两组频率叠在一起，曲线就不再是简单的周期。

真实的 Qwen3 有 64 组频率（$d_h=128$），$\theta=10^6$，最慢的一组周期远远超过上下文长度。这一页只有 2 组频率、3 个位置，照搬 $10^6$ 看不出任何变化，所以改用 $\theta=10$。
:::
::::

### 五、为什么 v 不转

你可能会问：q 和 k 都转了，v 为什么不转？

这是因为三者的分工不同。q 和 k 负责打分，决定“看谁、看多少”；v 是被取回来的内容。

位置只该影响前者。如果 v 也按位置转，输出 $\sum_s P_{t,s}\mathbf{v}_s$ 就是把转了不同角度的向量加在一起。这样一来，同一个词放在不同位置，取回的内容就不同了，而且输出会带上绝对位置，RoPE “只看距离”的好处也就没了。

所以 v 不转。一句话总结：位置决定看谁，内容原样取回。

### 六、反向传播

RoPE 没有参数。cos 和 sin 只由位置决定，不需要梯度。

梯度只是穿过它。旋转矩阵的逆就是它的转置，所以反向时，梯度按 $-t\omega$ 转回去。

---

### 七、回到这一页的 Qwen3

在 Qwen3 的 attention 里，RoPE 的位置是固定的：q_proj、k_proj 之后做 q_norm、k_norm，然后是 RoPE，最后才打分。

这一步很特殊。embedding、RMSNorm、MLP 都是逐个位置计算的，不知道自己在第几位。整个模型里，只有这里把位置告诉了网络。

那么，它到底改变了什么？下面用同一组 $\bar{\mathbf{q}}$、$\bar{\mathbf{k}}$，对比加和不加 RoPE 的结果。

::: buttons
- { label: 在图上看 RoPE 这几步, go: layer.self_attn.apply_rotary_pos_emb.qc }
- { label: '看打分 $\mathbf{q}\mathbf{k}^{\top}$', go: 'layer.self_attn.S[1]' }
:::

::: rope_compare
scores: 'layer.self_attn.Ss[1..H]'
weights: 'layer.self_attn.P[1..H]'
ids: ids
:::
