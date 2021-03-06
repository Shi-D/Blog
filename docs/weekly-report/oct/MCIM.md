# Multi-Community Influence Maximization in Device-to-Device social networks

> 多社区网络的影响力最大化



## 0 论文信息

**Author**: Xiaofei Wang, Xu Tong, Hao Fan, Chenyang Wang, Jianxin Li, Xin Wang

**Conference**: Knowledge-Based Systems 2021




## 1 背景

重复下载流行内容严重浪费了通信网络资源。

目前主流的解决办法是部署 WiFi 接入点 (AP) 或在基站缓存流行内容，从而减少重复流量。 

除了通过 WiFi 网络和小基站进行流量分流之外，还可以使用 D2D 机会主义共享机制来鼓励用户从周围设备中获取他们需要的内容。

???+ example
    初始用户 Alice 将多媒体文件下载到她的移动设备，然后通过 D2D 链接与她的朋友 Bob 共享该文件，而无需访问蜂窝网络，而 Bob 可能会与其他用户共享该文件。进行这种线下短距离分享的人很可能是朋友，可以形成 D2D 移动社交网络。 

 <img src="../MCIM-1.png" alt="MCIM-1" style="zoom:20%;" />



## 2 贡献

1. 对 D2D 社交网络中的多社区影响最大化 (MCIM) 问题进行建模，并证明它是 NP-hard 问题。我们将这个问题转化为两个子问题，包括单一社区影响最大化（SCIM）问题和多社区预算分配（MCBA）问题。
2. 对 SCIM 问题，提出了加权 LeaderRank with Neighbors (WLRN) 算法，来对一个社区中的用户进行排名，通过考虑链接的权重和用户的多跳邻居，可以提高传播覆盖率。
3. 对 MCBA 问题，提出了状态值迭代的优化预算分配（OBA）算法，将多个社区的分配过程建模为马尔可夫决策过程 (MDP)，并成功找到了最佳预算分配方案。



## 3 模型

### 3.1 目标函数构建

节点 $u$ 的直接覆盖用户表示为，$O(u) = \{u\} ∪ \{v|e_{uv} ∈ E\}$

节点 $u$ 的所有覆盖用户表示为，$C(u) = O(u) ∪ (∪v∈O(u)C(v))$

集合 $S$ 的覆盖用户表示为，$C(S) = ∪_{u∈S}C(u)$

则，目标函数即可定义为集合 $S$ 中覆盖用户的总和，$f(S) = |C(S)| = |∪_{u∈S}C(u)|$

 <img src="../MCIM-2.png" alt="MCIM-2" style="zoom:20%;" />

??? info
    离群用户：随着时间的推移，一些用户可能不再与种子用户或其他受种子用户影响的用户进行任何交互。在社区建成后不会直接或间接受到种子用户的影响。

### 3.2 多社区覆盖优化模型

将用户的重要性分为两类，即全局重要性和局部重要性。

一个 D2D 移动社交网络由 n 个社区组成，表示为 $G = {G_1, G_2, ..., G_n}$。每个社区 $G_i = (V_i , E_i )$ 是一个连通组件，其中 $V_i$ 和 $E_i$ 是用户集，边集为 $i = 1,2,...,n$。我们假设 n 个社区不重叠，即 $V_i \bigcap V_j =∅ \quad for\ i \neq j,m_i =|V_i|\ and\ M=|V|$。

对于 G~i~ 中的用户 u，全局重要性是指它对整个社交网络 G 的重要性，局部重要性是指它对社区 G~i~ 的重要性。例如，可以通过在 G 和 G~i~ 上运行 PageRank 来获得全局重要性和局部重要性。

定义 $K = (k_1 , k_2 , ... , k_n )$，$S_i$ 即为从 $G_i$ 中选择的 $k_i$ 个种子的集合。

$f (S) = f (∪^n_{i=1}S_i) = |C(∪^n_{i=1}S_i)| = |∪^n_{i=1}C(S_i)|$

定义 $S_i = g(G_i,k_i)$，则影响力最大化问题可以总结如下，

$$
\begin{align}
\max_{g,K} f(S) = |∪_{i=1}^n C(g(G_i,k_i))| \xlongequal {def} f_G(g,K) \\
s.t.∑_{i=1}^n k_i = k; \  k_i ≤ m_i, \ i = 1,2,...,n; \\
g :V_i × N → 2V_i, i=1,2,...,n
\end{align}
$$

为了在可接受的时间内解决这个问题，分两步解决：

1. 在单个社区中进行用户排名
2. 在多个社区中进行预算分配

若用 WLRN 算法来衡量用户的本地重要性，多社区中的影响最大化问题就可以建模如下，

$$
\begin{align}
\max_{\alpha,L,K} f(S)=|∪_{i=1}^n C(\arg\max_{\{u_{ij}\}}) \sum_{j=1}^{k_i}WLRN(u_{ij}; α, L))| \\
s.t. \sum _{i=1}^n k_i = k, \ k_i ≤ m_i,\ 1≤k_i ≤m_i; \\
L∈N; \quad α∈[0,1]
\end{align}
$$

其中，WLRN(u~ij~; α, L) 是用户 u~ij~ 的得分，α 和 L 是两个参数。

## 4 单社区的 WLRN

该阶段的目标为，对单个社区内的节点进行排序。

### 4.1 LR

> 原始 LeaderRank

R^∗^(u) 是从 u 接收到内容的一组用户。 S^∗^(v) 是一组向 v 发送内容的用户。LR(t,v) 是用户 v 在第 t 次迭代中的得分。

$$
LR(t+1,u)= \sum_{v∈R^∗(u)}\frac {LR(t,v)}{|S^∗(v)|}
$$

在 V 中有 n 个用户，在 E 中有 m 条边的社交图 G = (V,E)，我们向 G 中添加一个虚拟用户 u*，如下图，

<img src="../MCIM-3.png" alt="MCIM-3" style="zoom:18%;" />



### 4.2 WLR

> 带权重的 LeaderRank

在 D2D 社区中，外链多于内链的用户会更积极地发送文件而不是接收文件。 因此，具有更多外链的用户将获得更高的分数。

在D2D社区中，边的权重是指分享频率。 并且虚拟用户和其他用户之间的权重都设置为1，这对图中流动的分数没有影响。 WLR(t,v) 是 v 在第 t 次迭代中的得分,

$$
WLR(t+1,u)= \sum_{v∈R^∗(u)} \frac{W(u,v)WLR(t,v)}{\sum_{s\in S^*(v)}W(s,v)}
$$

矩阵形式为，

$$
L(t + 1) = AL(t)
$$

其中 L(t) 是一个 n+1 维向量，表示用户在第 t 次迭代后的得分，A 是一个 n+1 维矩阵，其值为：

$$
a_{uv}=
 \begin{cases}
 \frac {W(u,v)}{\sum_{s\in S^*(v)}W(s,v)},\,\,v\in R^*(u)\\
 0,\,\,otherwise\\
 \end{cases}
$$

当式(4)的迭代收敛，我们将虚拟用户 u^∗^ 的分数平均分配给其他 n 个用户，如下所示,

$$
WLR(u) = WLR(t_c , u) + \frac {WLR(t_c , u^∗)} {n}
$$


其中 WLR(u) 是用户 u 的最终得分，tc 表示迭代收敛的时间。



### 4.3 WLRN

> 带邻居和权重的 LeaderRank

用户的重要性不仅取决于自己，还取决于其多跳邻居，因此设计了带邻居和权重的 LeaderRank，即 WLRN

$$
WLRN(u)=WLR(u)+∑^L_{l=1} α^l ∑_{v \in N^l(u)} WLR(v)
$$

其中 WLR(u) 是用户 u 的收敛加权 LeaderRank 值，α 表示邻居影响的可调因子，其中 α∈[0, 1]。$N^l(u)$ 是用户 $u$ 的邻居的第 $l$ 跳。$l$ 是为用户邻居考虑的跳数。

 <img src="../MCIM-4.png" alt="MCIM-4" style="zoom:25%;" />



## 5 多社区的预算分配

该阶段的目标是将 K 个预算分配到不同的社区。

### 5.1 多社区预算分配优化模型

每个社区得到的种子集表示为，$g(G_i , k_i )$

则，种子集 $S_i$ 的覆盖大小（影响力延展度）为，$C(S_i) = C(g(G_i, k_i)) \xlongequal {def} C_g (G_i, k_i),$

则，优化问题变为，$\max_k f(S) = |∪_{i=1}^n C_g(G_i,k_i)| \xlongequal {def} f_{g,G}(K)$

由于，==来自不同社区的用户相互交流的可能性比同一社区的用户低得多==，因此作者假设 $C(S_1) ∩ C(S_2) = ∅$。

在上述假设下，优化问题变为，

$$
\max_K f(S) = \sum_{i=1}^{n} |C_g(G_i, k_i)| \\
s.t. \sum_{i=1}^{n} k_i=k; \ k_i \in N; \ k_i \leqslant m_i
$$

### 5.2 将分配过程建模为 MDP

<u>State</u>:  $s = (i, q)$ 表示前 $i-1$ 个社区 $\{G_1, G_2, . . . , G_{i−1}\}$ 已经分配完毕，还剩余 $q$ 个预算用于分配后面的社区  $\{G_i, G_{i+1}, . . . , G_n\}$。

<u>Action</u>:  $a = j$ 表示在状态 $s = (i, q)$ 下，在社区 $G_i$ 中，函数 $S_i =g(S_i,k_i)$ 选择了 $j$ 个节点作为该社区的种子集。

<u>Reward</u>:  $R(s, a) = f (g(G_i, j)) = |Cg (G_i, j)|$，即为 action a=j 在社区 G~i~ 中的影响力延展度，作为每个 action 的奖励。

<u>Policy</u>:   Policy 是用来指导下一步 action，在状态 s 下，动作 $a = π(s)$ 。

<u>State-Value function</u>:  状态价值函数 $V_π(s) = E_π[∑_{i=0}^{\infty} γ^i r_i|s_0 = s]$，在策略 π 下，采取下一个 action 得到的奖励。也即，$V_π (s)$  为从 $\{G_i , G_{i+1} , ...\}$ 中选出的 $q$ 个种子用户所覆盖的预期用户数，$V_π (s = (i, q)) = f (∪^n_{j=i}S_j)$

<img src="../MCIM-5.png" alt="MCIM-5" style="zoom:25%;" />



### 5.3 基于值迭代的分配方法

状态价值函数：$V_π(s)=E_π[r(s'|s,a)+γV_π(s')|s_0 =s] = \sum_{s' \in s} p(s' |s, a)[r (s, a) + γ V_π (s')].$

又，

$$
p(s_j|s_i, a) =
\begin{cases}
 1,\,\,j=i+1\\
 0,\,\,j \neq i+1\\
 \end{cases}
$$

则，$V_π(s)=r(s,a)+γV_π(s')$

最优状态值函数 V*(s)满足贝尔曼最优方程:  $V^∗(s) = \max[r(s,a)+γV_π(s′)]$

相应地，最优策略为，$π^∗(s) = \arg \max_a [R(s, a) + γV^∗(s′)]$

应用一种称为值迭代的基于模型的方法来找出最佳状态价值函数 V* 和最佳策略 π* (s) 。将此算法命名为最佳预算分配 (OBA)。

 <img src="../MCIM-6.png" alt="MCIM-6" style="zoom:25%;" />


## 6 最终算法

 <img src="../MCIM-7.png" alt="MCIM-7" style="zoom:20%;" />

==bug==

 <img src="../MCIM-15.png" alt="MCIM-15" style="zoom:30%;" />




## 7 实验

### 7.1 数据集和实验设置

数据集：使用现实的离线 D2D 共享数据集 Xen-der 和在线网络 Bitcoin-Alpha

![MCIM-8](../MCIM-8.png){width=500}

参数设置：

![MCIM-9](../MCIM-9.png){width=500}



### 7.2  WLRN 解决 SCIM 问题

对照算法：

1. PageRank
2. Closeness Centrality:u的贴近度中心性到达其他 n1 的平均路径长度的倒数
3. HITS : HITS 是一种链接分析算法，对社区中的用户进行排名，使用其权限更新规则对用户进行排名
4. Greedy:逐步寻找边际影响最大的用户
5. SeedRank : SeedRank 将用户的分享频率作为用户间链接的权重来考虑
6. k-core : 图的 k-core 是其所有顶点的度都大于 k 的子图

实验结果：

<img src="../MCIM-10.png" alt="MCIM-10" style="zoom:20%;" />

==2-hop 和 $\alpha = 0.1$ 时，能取得最优的传播覆盖率。== 

==考虑链接的权重和邻居的影响会导致更高的传播覆盖率。==

<img src="../MCIM-11.png" alt="MCIM-11" style="zoom:20%;" />

==和对照算法对比下，WLRN 能取得更好的解。==

<img src="../MCIM-12.png" alt="MCIM-11" style="zoom:20%;" />



1. ==WLRN 算法在平均覆盖率上最优。==
2. ==随着种子用户数量的增加，一个种子用户平均能够覆盖的用户数量也在下降。==
3. ==有影响力的用户在社区中只占很小的比例，但在传播中起着重要的作用。==

### 7.3  WLRN + OBA 解决 MCIM 问题

**OBA 的对照算法：**

1. Random : 分配给每个社区的预算是随机生成的
2. Average : 预算平均分配给每个社区
3. Proportion : 分配给每个社区的预算与社区规模成正比，越大的社区获得的预算越多
4. Without allocation (WOA) : 将 n 个社区视为一个大社区，不做任何分配，选择排名得分最高的用户

**WLRN + OBA 的对照算法：**

1. CIM : 社区检测和候选社区生成；第二步将规模大于平均社区规模的社区视为重要社区。种子用户从重要的 社区中选择
2. CoFIM : 它将社交网络中的信息扩散视为两个阶段:节点扩展和社区内传播，基于这样的扩散模型，定义了一 个影响评价函数，并使用贪婪方法来选择边际效应最大的用户
3. MMIC : 它是一个在多社区多层网络中竞争的多重影响最大化扩散模型，模型还包含两个阶段: 找到 k 个最有 影响力的种子，并将 k 个种子公平地分配给 t 个不同的用户;使用 RRE 算法选择种子用户

**实验结果：**

 <img src="../MCIM-13.png" alt="MCIM-13" style="zoom:25%;" />

==上表展示了不同算法组合的覆盖率，其中总预算，即种子用户数为 50。表明无论使用何种排序算法，所提出的 OBA 算法都达到了在所有比较的分配算法中覆盖率最高。==

 <img src="../MCIM-14.png" alt="MCIM-14" style="zoom:20%;" />

==由于 D2D 传播要求参与传播的双方在一定范围内，所以 D2D 传播具有很强的地域限制，这导致 D2D 社交网络社 区数量普遍较少，当种子节点数量达到一定数量(35个种子节点)时，算法的最大传播能力将受到限制。==

