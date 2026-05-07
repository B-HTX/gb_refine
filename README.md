我建议优先改 **“标签置信度构造 + 近邻粒球使用方式”**，不要先动复杂优化。

你现在的版本里，每个粒球的标签向量主要来自候选标签的均匀归一化；候选标签多时，粒球标签向量会比较模糊。后面虽然用了 `quadprog`，但目标基本是把这些模糊目标投影回候选标签约束里，区分度提升有限。`CalAccuracy` 最终只看每行最大值，所以初始化矩阵需要更“尖锐”一点。 

我最推荐的简单改法是：

1. **先做一次 kNN 候选标签平滑**
   不再把候选标签都初始化为完全相同，而是让相似样本中更常出现的候选标签得到更高初值。

### 原来的做法

你原来的代码先做：

```matlab
train_p_conf = normalize_candidate_label_rows(train_p_target);
```

也就是如果第 `i` 个样本有 3 个候选标签，那么这 3 个标签一开始都是：

[
\frac{1}{3},\frac{1}{3},\frac{1}{3}
]

这个做法太平均了。后面粒球里再平均一次，就会更“糊”。原代码确实是先把候选标签矩阵行归一化成 `train_p_conf`，再用它生成粒球标签向量。

---

### 改后的思路

假设：

[
Y \in {0,1}^{p \times q}
]

是候选标签矩阵，`Y(i,l)=1` 表示样本 (x_i) 的候选标签里包含第 (l) 类。

先得到原始均匀置信度：

[
Y^{(0)}_{il}=
\begin{cases}
\frac{1}{|S_i|}, & l \in S_i \
0, & l \notin S_i
\end{cases}
]

其中 (S_i) 是样本 (x_i) 的候选标签集合。

然后对每个样本找 k 个最近邻。因为代码里已经做了 L2 归一化：

```matlab
train_data = normalize_rows_l2(train_data);
```

所以相似度可以直接用余弦相似度：

[
s_{ij}=x_i^\top x_j
]

取 (k) 个最近邻：

[
\mathcal{N}_k(i)
]

邻居权重为：

[
a_{ij}=
\frac{\max(s_{ij},0)}
{\sum_{t\in \mathcal{N}*k(i)} \max(s*{it},0)}
]

如果所有相似度都小于等于 0，就退化成均匀权重。

然后用邻居的候选标签置信度来修正当前样本：

[
\tilde{Y}_{i}
=============

\sum_{j\in \mathcal{N}*k(i)}
a*{ij}Y^{(0)}_j
]

但注意，代码里会把非候选标签置 0：

[
\tilde{Y}_{il}=0,\quad l\notin S_i
]

这是为了保证最后不会把分数分给非候选标签。

---

### 先验修正

代码里还有一个小细节：

```matlab
label_prior = sum(Y, 1) / p;
prior_corr = 1 ./ ((label_prior + eps) .^ opts.prior_power);
```

公式是：

[
\pi_l=\frac{1}{p}\sum_{i=1}^{p}Y_{il}
]

[
c_l=\frac{(\pi_l+\varepsilon)^{-\beta}}
{\text{mean}_r((\pi_r+\varepsilon)^{-\beta})}
]

其中 (\beta=\texttt{opts.prior_power})，默认是 0.5。

然后：

[
\tilde{Y}*{il}\leftarrow \tilde{Y}*{il}c_l
]

这个作用是：如果某个标签在候选标签里出现得特别频繁，它很容易在 kNN 平滑中占优势。先验修正会稍微压一压频繁标签，稍微抬一抬稀有标签，避免大类标签把小类标签淹没。

---

### 最后融合原始候选标签和邻域标签

代码里：

```matlab
row = (1 - opts.alpha_knn) * Y0(i, :) + opts.alpha_knn * neigh;
```

公式是：

[
Y^{conf}_i
==========

(1-\alpha)Y^{(0)}_i+\alpha \tilde{Y}_i
]

默认：

[
\alpha=0.6
]

所以不是完全相信邻居，而是 40% 保留原始均匀候选标签，60% 使用邻域结构。

---

### 举个例子

假设样本 (x_i) 的候选标签是：

[
S_i={A,B}
]

原始置信度是：

[
Y^{(0)}_i=[0.5,0.5]
]

它的 3 个近邻给出的加权候选标签统计结果是：

[
\tilde{Y}_i=[0.85,0.15]
]

也就是说，邻居更支持 A。

如果 (\alpha=0.6)，那么：

[
Y^{conf}_i
==========

0.4[0.5,0.5]+0.6[0.85,0.15]
]

[
Y^{conf}_i=[0.71,0.29]
]

这就比原来的 `[0.5, 0.5]` 更有区分度。

**直观理解：**

原来是：

[
A 和 B 一样可疑
]

现在是：

[
A 更像真标签，但 B 仍然保留一定可能性
]

这对后面生成粒球标签向量非常重要，因为粒球内部再聚合时，不会一直聚合一堆完全均匀的 `[0.5,0.5]`、`[1/3,1/3,1/3]`。

---





   

3. **近邻粒球不要取 `lambda * m_i` 那么多**
   当前如果某类粒球很多，取 30% 可能太多，会把局部结构平均掉。改成
   `ceil(lambda * sqrt(m_i))`，并限制在 3 到 10 个附近。

4. **每个候选标签只取自己的 label-specific score**
   原代码对每个候选标签都累加完整的 q 维粒球标签向量，容易产生“交叉标签泄漏”。我改成：标签 `lab` 的粒球只给 `score(j, lab)` 提供证据。

这一点是我觉得最关键的结构性修改。

---

### 原来的做法

原代码中，对于样本 (x_j) 和它的某个候选标签 `lab`，会找到该标签下最近的粒球，然后重构出一个完整的 q 维向量：

```matlab
h = w' * gvecs(nb, :);
h_sum(j, :) = h_sum(j, :) + h;
```

也就是说，**第 lab 类粒球不仅给 lab 加分，还会给所有类别加分**。

假设当前处理的是候选标签 A，但是 A 标签下的附近粒球标签向量是：

[
g_A=[0.4,0.4,0.2]
]

那么原代码会把 `[0.4,0.4,0.2]` 整个加到当前样本的 `h_sum` 里。

这会带来一个问题：
**A 标签的粒球证据，可能会给 B、C 标签也加分。**

如果每个候选标签都这样贡献完整向量，最后不同标签之间会互相“串分”。

---

### 改后的做法

refined 代码改成：

```matlab
base = sum(w .* gvecs(nb, lab));
score(j, lab) = base;
```

也就是：
当我正在判断标签 `lab` 是否可信时，我只取附近粒球标签向量里的第 `lab` 个分量。

公式如下。

对于第 (l) 个标签，先用所有候选标签包含 (l) 的样本生成粒球。第 (l) 类的第 (b) 个粒球中心是：

[
c_{lb}
]

粒球标签向量是：

[
g_{lb}\in \mathbb{R}^{q}
]

其中 (g_{lb,r}) 表示这个粒球对第 (r) 个标签的支持程度。

对于样本 (x_i)，如果 (l\in S_i)，找到第 (l) 类下最近的若干个粒球：

[
\mathcal{B}_l(i)
]

距离为：

[
d_{ilb}=|x_i-c_{lb}|^2
]

然后用高斯核距离权重：

[
\rho_{ilb}
==========

\frac{
\exp\left(-\frac{d_{ilb}}{2\sigma_i^2+\varepsilon}\right)
}{
\sum_{t\in \mathcal{B}*l(i)}
\exp\left(-\frac{d*{ilt}}{2\sigma_i^2+\varepsilon}\right)
}
]

其中代码里 (\sigma_i^2) 近似用这些距离的中位数：

[
\sigma_i^2=\text{median}(d_{ilb})
]

最后，第 (l) 个标签的分数是：

[
score_{il}
==========

\sum_{b\in \mathcal{B}*l(i)}
\rho*{ilb}g_{lb,l}
]

注意这里是：

[
g_{lb,l}
]

不是完整的：

[
g_{lb,:}
]

---

### 距离可靠性因子

代码里还有一项：

```matlab
rel = exp(-min(near_d2) / (2 * (median(d2) + eps)));
base = base * rel;
```

公式是：

[
rel_{il}
========

\exp
\left(
-\frac{\min_b d_{ilb}}
{2\cdot \text{median}*b(d*{ilb})+\varepsilon}
\right)
]

然后：

[
score_{il}\leftarrow score_{il}\cdot rel_{il}
]

它的作用是：
如果样本 (x_i) 离第 (l) 类粒球整体比较近，那么这个标签更可信；如果最近粒球都很远，那么这个标签虽然是候选标签，但可信度要稍微降一点。

这个因子不会特别激进，因为指数分母用了整体距离中位数，所以通常只是温和调整。

---

### 举个例子

假设样本 (x_i) 的候选标签是：

[
S_i={A,B,C}
]

现在分别看 A、B、C 三个标签的粒球。

假设找到的最近粒球加权后得到：

[
h_A=[0.40,0.40,0.20]
]

[
h_B=[0.35,0.45,0.20]
]

[
h_C=[0.70,0.10,0.20]
]

原代码会把三个完整向量加起来：

[
h_{sum}
=======

h_A+h_B+h_C
]

[
h_{sum}
=======

[1.45,0.95,0.60]
]

归一化后 A 很容易赢。

但是这里有个问题：
第三项 (h_C=[0.70,0.10,0.20]) 是在判断 “C 这个候选标签是否可信” 时得到的结果。它虽然说明 C 粒球附近其实更像 A，但它不应该直接给 A 加很多分，否则就变成 **C 标签的判断过程反过来给 A 加分**。

改后的做法是只取对角项：

[
score_A=h_{A,A}=0.40
]

[
score_B=h_{B,B}=0.45
]

[
score_C=h_{C,C}=0.20
]

于是：

[
score=[0.40,0.45,0.20]
]

归一化后：

[
[0.381,0.429,0.190]
]

B 会略微领先。

**直观理解：**

原来是：

> 我判断 A 时，顺便给 B、C 加分；我判断 B 时，也顺便给 A、C 加分。

改后是：

> 判断 A 的过程只负责给 A 打分；判断 B 的过程只负责给 B 打分。

这样更符合“每个候选标签单独评估可信度”的逻辑。

---



   

6. **去掉二次规划，改成距离核加权 + 行归一化 + 轻微 sharpening**
   更简单、更快，也更适合初始化。原粒球生成函数可以继续用，不需要大改。

### 原来的最终优化

原代码最后构造了一个二次规划：

[
\min_F
\sum_i
\sum_{l\in S_i}
|f_i-h_{i,l}|^2
]

并且约束：

[
0\leq f_{il}\leq Y_{il}
]

[
\sum_{l\in S_i}f_{il}=1
]

代码里用的是 `quadprog`。

这个形式看起来比较正式，但这里有个问题：如果前面的 (h) 本身比较模糊，那么二次规划也只是把模糊分数投影回可行域，未必能让最大标签更明显。

而最后算准确率的时候，本质是：

[
\hat{y}*i=\arg\max_l F*{il}
]

`CalAccuracy` 也是取 `test_outputs` 每行最大值和真实标签最大值比较。

所以初始化矩阵不一定需要复杂 QP，更重要的是：
**正确标签的分数要尽量排到第一。**

---

### 改后的最终流程

经过第 3 步，我们得到：

[
score_i=[score_{i1},score_{i2},...,score_{iq}]
]

先把非候选标签置 0：

[
score_{il}=0,\quad l\notin S_i
]

然后候选标签内归一化：

[
z_{il}
======

\frac{score_{il}}
{\sum_{r\in S_i}score_{ir}}
]

如果这一行全是 0，就退回 kNN 平滑后的：

[
Y_i^{conf}
]

---

### 再融合一点 kNN 结果

代码里：

```matlab
row = (1 - opts.knn_blend_final) * row + opts.knn_blend_final * Y_conf(i, :);
```

公式是：

[
\hat{f}_i
=========

(1-\gamma)z_i+\gamma Y_i^{conf}
]

默认：

[
\gamma=0.15
]

也就是说，85% 相信粒球打分，15% 保留 kNN 候选标签平滑结果。

为什么要这样？

因为粒球有时会分得不好，尤其是小数据集或者噪声比较大的数据集。如果完全相信粒球，容易被局部错误粒球带偏。保留一点 kNN 结果，相当于给最终分数加一个稳定项。

---

### 最后 sharpening

代码里：

```matlab
row(cand) = row(cand) .^ opts.sharpen;
row = row / sum(row);
```

公式是：

[
F_{il}
======

\frac{
\hat{f}*{il}^{\tau}
}{
\sum*{r\in S_i}\hat{f}_{ir}^{\tau}
}
]

其中：

[
\tau=\texttt{opts.sharpen}
]

默认：

[
\tau=1.4
]

如果 (\tau>1)，大的值会变得更大，小的值会变得更小。

---

### 举个例子

假设融合之后：

[
\hat{f}_i=[0.60,0.40]
]

取：

[
\tau=1.4
]

则：

[
F_i=
\frac{[0.60^{1.4},0.40^{1.4}]}
{0.60^{1.4}+0.40^{1.4}}
]

计算后大约是：

[
F_i=[0.638,0.362]
]

差距从：

[
0.60-0.40=0.20
]

变成：

[
0.638-0.362=0.276
]

最大标签更明显。

如果是：

[
\hat{f}_i=[0.71,0.29]
]

sharpening 后大约变成：

[
F_i=[0.778,0.222]
]

这对 `argmax` 型准确率有利。

---

## 三个修改连起来看

整体流程可以理解成：

[
Y^{(0)}
\rightarrow
Y^{conf}
\rightarrow
score
\rightarrow
Outputs
]

分别对应：

1. **kNN 平滑：**

[
Y^{conf}_i=(1-\alpha)Y^{(0)}_i+\alpha \tilde{Y}_i
]

作用：
让候选标签初值不再完全均匀。

2. **标签专属粒球打分：**

[
score_{il}
==========

\sum_{b\in \mathcal{B}*l(i)}
\rho*{ilb}g_{lb,l}
]

作用：
判断第 (l) 个候选标签时，只给第 (l) 个标签打分，减少交叉标签泄漏。

3. **最终融合和锐化：**

[
\hat{f}_i=(1-\gamma)z_i+\gamma Y_i^{conf}
]

[
F_{il}
======

\frac{
\hat{f}*{il}^{\tau}
}{
\sum*{r\in S_i}\hat{f}_{ir}^{\tau}
}
]

作用：
保留稳定性，同时让最大标签更清楚。

---




   

参数：

```matlab
opts.k_nn = 10;
opts.alpha_knn = 0.6;
opts.sharpen = 1.4;
opts.knn_blend_final = 0.15;
opts.min_near_balls = 3;
opts.max_near_balls = 10;

[F, acc, info] = build_label_manifold_gb_refined(data, partial_target, target, 0.3, 10, opts);
```

建议你先试这三组：

| 参数组                                   | 适用情况                        |
| ------------------------------------- | --------------------------- |
| `k_nn=10, alpha_knn=0.6, sharpen=1.4` | 默认推荐，先跑这个                   |
| `k_nn=15, alpha_knn=0.7, sharpen=1.5` | Lost、Msrcv2 这种当前 gb 明显偏低的可试 |
| `k_nn=5, alpha_knn=0.4, sharpen=1.2`  | Fg-net 这种本来很低、容易过锐化的数据可试    |

从代码结构看，这个版本最可能提升的是 **Lost 和 Msrcv2**，因为它们当前 gb 比 Leaf 低很多，更像是初始化过于模糊和近邻粒球过度平均导致的。
