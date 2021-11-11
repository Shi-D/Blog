# S2V-DQN vs GCOMB



## 1 训练流程

**GCOMB**

1. Probabilistic Greedy，修剪出候选节点

2. GCN，预测单个节点质量 score‘

3. Q-learning，学习 Q 函数 (DQN)

**S2V-DQN**

1. struc2vec，图嵌入获取节点标签向量 μ
2. Q-learning，学习 Q 函数 (DQN)



## 2 节点标签向量

**GCOMB**（参数数量 4）

- 每个节点的**标签**向量 ==$\mu_v$==，即为 $[score'(v), loc(v, S_t)]$
  - **节点自身质量**，即 score‘ 值
  - **节点的位置**，即 2 个有较多重合邻居的节点都加入种子集大概率不是个好的选择，loc 值，$loc(v, S_t) = |N(v) \diagdown \bigcup_{\forall u \in S_t} N(u) |$
- 候选节点集的**标签**为 ==$\mu_{C_t}$== $= MAXPOOL\{ \mu_v | v \in C_t\}$
- 种子集的**标签**为 ==$\mu_{S_t}$== $= MAXPOOL\{\mu_v|v \in S_t\}$
- $\mu_{C_t, S_t, v} = CONCAT(\theta_1·\mu_{C_t}, \theta_2·\mu_{S_t}, \theta_3·\mu_v)$
- $Q'_n(S_t, v; \theta_Q) = \theta_4 · \mu_{C_t, S_t, v}$

**S2V-DQN**（参数数量 7）

- 由 S2V 训练得到的 p 维向量 $\mu_v^{(t+1)} \leftarrow relu(\theta_1 x_v + \theta_2 \sum_{u \in N(v)} \mu_u^{t} + \theta_3 \sum_{u \in N(v)} relu(\theta_4w(v, u)))$

- concat 邻居节点和自身节点的 $\mu$ 值：$Q'_n(h(S), v; \theta_Q) = \theta_5 relu([\theta_6 \sum_{v \in V} \mu_u^{(T)}, \theta_7 \mu_v^{(T)}])$

## 3 DQN

**GCOMB**

1. 学习参数使用 Adam 

2. exploration 和 exploitation 均采用 ϵ-greedy，但与采样次数有关。

**S2V-DQN**

1. 学习参数使用 SDG 

2. exploration 和 exploitation 均采用 ϵ-greedy，但为定值。