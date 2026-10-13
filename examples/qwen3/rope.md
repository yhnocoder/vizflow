:::: columns 300 1fr
::: column
::: rope_dial
:::
:::

::: column
::: lead
注意力的分数是 query 与 key 的点积，点积里没有位置，所以要另外把位置加进 q、k。旋转位置编码（Rotary Position Embedding，RoPE）就是这样做的，并且让分数只依赖两个位置的差。下面从这个要求出发，推出 RoPE 的公式，再对照 HF 的代码。
:::

### 一、点积里没有位置

位置 $t$ 的 query 对位置 $s$ 的 key 打分 $\mathbf{q}_t\cdot\mathbf{k}_s$。$\mathbf{q}_t$ 和 $\mathbf{k}_s$ 都只由各自位置的 token 经过逐位置的计算得到（embedding、RMSNorm、线性投影都不看位置），所以分数只取决于两个 token 的内容。

因此，把可见的几个 key 交换位置，每个 key 得到的分数不变，加权平均 $\sum_sP_{t,s}\mathbf{v}_s$ 也不变。causal mask 决定哪些位置可见，让输出与顺序有关，但它不提供两个位置之间的距离，所以需要把位置显式地放进 q、k。问题是怎么放？

### 二、要求：分数只依赖位置差

设给 q、k 加位置的操作是 $f$：位置 $t$ 的 query 变成 $f(\mathbf{q},t)$，位置 $s$ 的 key 变成 $f(\mathbf{k},s)$。我们希望分数里位置只以差的形式出现，这样同样的两个 token 相隔同样的距离，无论出现在句子的哪里，分数都一样：

$$\langle f(\mathbf{q},t),\,f(\mathbf{k},s)\rangle=g(\mathbf{q},\mathbf{k},t-s)$$

同时要求位置 0 不改变向量：$f(\mathbf{q},0)=\mathbf{q}$。

### 三、二维情形的解

先看 $\mathbf{q}$ 只有两个分量的情形。把 $(q_0,q_1)$ 看成复数 $\tilde q=q_0+\mathrm{i}\,q_1$，二维内积可以写成 $\langle\mathbf{q},\mathbf{k}\rangle=\mathrm{Re}[\tilde q\,\tilde k^{*}]$。把 $f$ 写成模长和辐角：$f(\mathbf{q},t)=R(\mathbf{q},t)\,e^{\mathrm{i}\Theta(\mathbf{q},t)}$。

由上面的内积写法，只要 $\tilde f(\mathbf{q},t)\tilde f^{*}(\mathbf{k},s)$ 只依赖 $t-s$，分数就只依赖 $t-s$。这要求模长和辐角分别满足：

$$R(\mathbf{q},t)R(\mathbf{k},s)=R_g(t-s),\qquad \Theta(\mathbf{q},t)-\Theta(\mathbf{k},s)=\Theta_g(t-s)$$

取 $s=t$，右边是 $t-s=0$ 时的值，由 $f(\cdot,0)$ 不变可知它等于 $\|\mathbf{q}\|\|\mathbf{k}\|$ 和 $\arg\tilde q-\arg\tilde k$。于是最简单的解是：模长不随位置变，$R(\mathbf{q},t)=\|\mathbf{q}\|$；辐角是原辐角加上一个只和位置有关的量，$\Theta(\mathbf{q},t)=\arg\tilde q+\varphi(t)$。再取 $s=t-1$，得到 $\varphi(t)-\varphi(t-1)$ 是与 $t$ 无关的常数，记为 $\omega$，所以 $\varphi(t)=t\,\omega$。合起来：

$$f(\mathbf{q},t)=\tilde q\,e^{\mathrm{i}t\omega}$$

乘 $e^{\mathrm{i}t\omega}$ 就是把向量转过角度 $t\omega$。验证：$\mathrm{Re}[\tilde q e^{\mathrm{i}t\omega}\,(\tilde k e^{\mathrm{i}s\omega})^{*}]=\mathrm{Re}[\tilde q\tilde k^{*}e^{\mathrm{i}(t-s)\omega}]$，只依赖 $t-s$。也就是说，$t$ 和 $s$ 同时加 1 时两个向量一起转，点积不变。
:::
::::

### 四、推广到 $d_h$ 维

内积可以按分量拆开：$\mathbf{q}\cdot\mathbf{k}=\sum_c$（每一对分量的二维内积）。所以把 $d_h$ 个分量分成 $d_h/2$ 对，每一对按第三节的方式各自旋转，每一对的贡献只依赖 $t-s$，于是总和也只依赖 $t-s$。

HF 的配对方式是第 $c$ 个和第 $c+d_h/2$ 个分量一对，$c=0,\dots,d_h/2-1$。一对的旋转是

$$\begin{pmatrix}q'_c\\ q'_{c+d_h/2}\end{pmatrix}=\begin{pmatrix}\cos t\omega_c & -\sin t\omega_c\\ \sin t\omega_c & \cos t\omega_c\end{pmatrix}\begin{pmatrix}q_c\\ q_{c+d_h/2}\end{pmatrix}$$

而 RoPE 的原论文把相邻的第 $2i$ 和第 $2i+1$ 个分量配对。两种配对只差一个固定的分量重排。对 q 和 k 用同一个重排，点积不变，而这个重排可以并进 $W_q$、$W_k$ 的行顺序里，所以两种写法的模型能力相同，只是训练得到的权重不能直接互换。

整个 head 的变换写成矩阵是 $R_t$，它把 $d_h/2$ 对分量（第 $c$ 和第 $c+d_h/2$ 个分量为一对）各转过一个角度，是正交矩阵，并且 $R_t^{\top}R_s=R_{s-t}$。于是

$$(R_t\mathbf{q})^{\top}(R_s\mathbf{k})=\mathbf{q}^{\top}R_{s-t}\,\mathbf{k}$$

### 五、rotate_half 就是这个旋转

代码里没有旋转矩阵，写的是 $\mathbf{q}\odot\cos+\mathrm{rotate\_half}(\mathbf{q})\odot\sin$。rotate_half 把一对 $(q_c,q_{c+d_h/2})$ 变成 $(-q_{c+d_h/2},q_c)$，也就是转过 $90^\circ$。cos、sin 表的第 $c$ 列和第 $c+d_h/2$ 列是同一个角度 $t\omega_c$（rotary_emb 里 <code>torch.cat((freqs, freqs), dim=-1)</code> 的作用），所以同一对的两个分量乘的是同一组 cos、sin。代入展开：

$$q'_c=q_c\cos t\omega_c-q_{c+d_h/2}\sin t\omega_c,\qquad q'_{c+d_h/2}=q_{c+d_h/2}\cos t\omega_c+q_c\sin t\omega_c$$

正好是上面矩阵乘法的两行，所以乘 cos、rotate_half、乘 sin、相加这几步合起来就是一次旋转。代码不写旋转矩阵，是因为它每行只有 2 个非零元素（第 $c$ 行在第 $c$ 和第 $c+d_h/2$ 列，把分量重排成相邻的一对一对以后才是对角上的 $2\times2$ 块）：按矩阵乘要做 $d_h^2$ 次乘法，其中大部分在乘 0，而按这个写法只要 $2d_h$ 次。

:::: columns 340 1fr
::: column
::: rope_curve
:::
:::

::: column
### 六、频率怎么选

一对分量的贡献是 $\mathrm{Re}[\tilde q_c\tilde k_c^{*}e^{\mathrm{i}(t-s)\omega_c}]$，它是 $t-s$ 的周期函数，周期 $2\pi/\omega_c$。只用一个频率时，相差一个周期的两个距离无法区分。

所以每一对用不同的频率：

$$\omega_c=\theta^{-2c/d_h},\qquad c=0,\dots,d_h/2-1$$

$c=0$ 时 $\omega=1$，周期约 6.3 个位置；$c$ 越大转得越慢，最慢一组的周期是 $2\pi\,\theta^{(d_h-2)/d_h}$。快的几组区分相邻的位置，慢的几组区分远处的位置，这和钟表用秒针、分针、时针一起表示时间是同一个道理。于是各组的贡献相加后，分数作为 $t-s$ 的函数不再有短周期。

这个取法沿用了 Sinusoidal 位置编码，它还带来一个性质：苏剑林在 RoFormer 的推导中用分部求和证明，$q$、$k$ 固定时，各组之和的上界大致随 $|t-s|$ 增大而减小，所以距离越远，分数的可能范围越小。

这一页的 $d_h$ 很小，只有 $d_h/2$ 组频率，最慢一组的频率是 $\theta^{-(d_h-2)/d_h}$。$\theta$ 大时这一组在几个位置内几乎不转，所以这一页把 $\theta$ 也取得很小，让慢的那组也能看出转动。真实规模的 $\theta$ 和 $d_h$ 见「位置编码表」那一步的说明。
:::
::::

### 七、v 为什么不转

q 和 k 只用来算权重 $P_{t,s}$，v 是被加权求和的内容。位置差通过分数进入 $P$，已经影响了输出，所以 v 不需要再加位置。

进一步，如果 v 也按位置旋转，结果反而有害：输出 $\sum_sP_{t,s}R_s\mathbf{v}_s$ 是转过不同角度的向量之和，同一个 token 出现在不同位置，取回的内容不同，而且输出里出现的是绝对位置 $s$，不再只依赖位置差。所以 HF 只对 q、k 调用 <code>apply_rotary_pos_emb</code>。

### 八、反向传播

RoPE 没有参数，cos、sin 也只由位置决定，所以都不需要梯度，rotary_emb 在 <code>torch.no_grad()</code> 下算出它们。

反向也是旋转。$\mathbf{q}'=R_t\mathbf{q}$ 的局部导数是 $R_t$，所以 $\pfl{\mathbf{q}}=R_t^{\top}\pfl{\mathbf{q}'}=R_{-t}\pfl{\mathbf{q}'}$：梯度按 $-t\omega_c$ 转回去，模长不变。

---

### 九、在 Qwen3 里的位置

Qwen3 的 attention 里，顺序是 q_proj、k_proj 之后 <code>view</code> 成 head，做 q_norm、k_norm，<code>transpose(1, 2)</code>，然后 RoPE，最后打分。cos、sin 由模型开头的 <code>rotary_emb</code> 算一次，每一层用同一组。

q_norm、k_norm 在 RoPE 之前，旋转不改变模长，所以 norm 给出的分数上界在 RoPE 之后仍然成立。

在模型里，RoPE 是位置信息唯一的显式来源：embedding、RMSNorm、MLP 都逐位置计算，不知道自己在第几位。所以加和不加 RoPE 的差别就是位置带来的全部差别。下面用第 0 层同一组 $\bar{\mathbf{q}}$、$\bar{\mathbf{k}}$ 对比两者的分数与注意力权重。

::: buttons
- { label: 在图上看 RoPE 这几步, go: 'layers[0].self_attn.apply_rotary_pos_emb.qc' }
- { label: '看打分 $\mathbf{q}\mathbf{k}^{\top}$', go: 'layers[0].self_attn.eager_attention_forward.S' }
:::

::: rope_compare
scores: 'layers[0].self_attn.eager_attention_forward.Ss'
weights: 'layers[0].self_attn.eager_attention_forward.P'
ids: ids
:::
