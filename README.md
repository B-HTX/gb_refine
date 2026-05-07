我建议优先改 **“标签置信度构造 + 近邻粒球使用方式”**，不要先动复杂优化。

你现在的版本里，每个粒球的标签向量主要来自候选标签的均匀归一化；候选标签多时，粒球标签向量会比较模糊。后面虽然用了 `quadprog`，但目标基本是把这些模糊目标投影回候选标签约束里，区分度提升有限。`CalAccuracy` 最终只看每行最大值，所以初始化矩阵需要更“尖锐”一点。 

我最推荐的简单改法是：

1. **先做一次 kNN 候选标签平滑**
   不再把候选标签都初始化为完全相同，而是让相似样本中更常出现的候选标签得到更高初值。

2. **近邻粒球不要取 `lambda * m_i` 那么多**
   当前如果某类粒球很多，取 30% 可能太多，会把局部结构平均掉。改成
   `ceil(lambda * sqrt(m_i))`，并限制在 3 到 10 个附近。

3. **每个候选标签只取自己的 label-specific score**
   原代码对每个候选标签都累加完整的 q 维粒球标签向量，容易产生“交叉标签泄漏”。我改成：标签 `lab` 的粒球只给 `score(j, lab)` 提供证据。

4. **去掉二次规划，改成距离核加权 + 行归一化 + 轻微 sharpening**
   更简单、更快，也更适合初始化。原粒球生成函数可以继续用，不需要大改。

我已经把这个版本写成 MATLAB 文件了：

[下载 build_label_manifold_gb_refined.m](sandbox:/mnt/data/build_label_manifold_gb_refined.m)

用法：

```matlab
[F, acc, info] = build_label_manifold_gb_refined(data, partial_target, target);
```

或者手动调参：

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

我没有你的四个 `.mat` 数据集，所以不能直接替你跑出新准确率；但从代码结构看，这个版本最可能提升的是 **Lost 和 Msrcv2**，因为它们当前 gb 比 Leaf 低很多，更像是初始化过于模糊和近邻粒球过度平均导致的。
