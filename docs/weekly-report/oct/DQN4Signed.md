# Deep RL for IM in Signed Social Network



![](DQN-1.png)



## 1 Background

**Goal**：最大化总奖励数

**Question**：如何得到每个状态下的最优动作，使得奖励最大化？

**Challenge**：$Q^*$ 能得到每个动作带来的平均回报，即 $a^*=\arg \max_a Q^*(s,a)$

**Solution**：学习一个 $Q$ 函数来近似 $Q^*$，$Q$ 函数即为动作价值函数



## 2 Solution

**传统强化学习**：得到一个 table，通过训练的方式，记录并更新每种动作下得到的回报 (奖励和)

**DQN**：Deep Q Network，用神经网络 $Q(s,a;w)$ 来近似 $Q^*(s,a)$



## 3 DQN

1. 观察状态 $s_t$ 和动作 $a_t$
2. 预测该动作下的 $Q$ 函数的值：==$q_t = Q(s_t, a_t; W_t)$==
3. 环境更新状态，给出动作 $a_t$ 的 reward $r_t$
4. 计算 TD target: ==$y_t = r_t + \gamma ·\max_a Q(s_{t+1},a_{t+1}; W_t)$== 
5. 进行梯度下降，更新参数：$w_{t+1}=w_t-(q_t-y_t)·d_t$



## 4 剪枝

将符号社交网络拆分成 G+ 和 G-，分别剪枝后，再合并。



## 5 IM in Signed Social Network

**影响力传播**

$f_{S \rightarrow j}^+ = \alpha^+(\sum_{i,j\in E^+} w_{i,j}f^+_{S \rightarrow i} \ + \ \sum_{i,j\in E^-} w_{i,j}f^-_{S \rightarrow i})$

$f_{S \rightarrow j}^- = \alpha^-(\sum_{i,j\in E^+} w_{i,j}f^-_{S \rightarrow i} \ + \ \sum_{i,j\in E^-} w_{i,j}f^+_{S \rightarrow i})$

$f_{S \rightarrow j} = f_{S \rightarrow j}^+ + f_{S \rightarrow i}^-$

**抽象成马尔可夫决策过程**

<u>State</u>:  $s = S$ ，小 s 代表当前状态，大 S 即为当前选为种子的节点。

<u>Action</u>:  $a = j$ 表示在状态 $s$ 下，选择了节点 $j$ 作为种子节点。

<u>Reward</u>:  奖励函数设计如下，

$$
\begin{align}
R_{a=j}&=f(S \bigcup j) - f(S)\\
&= f_{\{S \bigcup j\} \rightarrow V}^+ + f_{\{S \bigcup j \} \rightarrow V}^- - f_{S \rightarrow V}^+ - f_{S \rightarrow V}^-
\end{align}
$$

<u>Discounted Return</u>：折扣回报函数，

$$
\begin{align}
U_t &= R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + ... \\
&=R_t + \gamma U_{t+1}
\end{align}
$$

<u>Action-Value function</u>:  动作价值函数，
$$
Q(s_t, a_t)= E[U_t|s_t,a_t]
$$

$$
\underbrace{ Q(s_t,a_t;\theta)}_{predict}= \underbrace{ \underbrace{r_t}_{真实奖励}+ \underbrace{\gamma · Q(s_{t+1},a_{t+1};\theta)}_{predict}}_{TD \ target}
$$



## 6 Q-function 的设计

### 6.1 S2V-DQN

NIPS '19
$$
\mu_v^{(t+1)} \leftarrow relu(\theta_1 x_v + \theta_2 \sum_{u \in N(v)} \mu_u^{t} + \theta_3 \sum_{u \in N(v)} relu(\theta_4w(v, u)))\\
\Q'_n(h(S), v; \theta_Q) = \theta_5 relu([\theta_6 \sum_{v \in V} \mu_u^{(T)}, \theta_7 \mu_v^{(T)}])\\
$$
DIEM\DISCO

### 6.2 GCOMB

NIPS '20
$$
\mu_v = [score'(v), \ loc(v, S_t)]\\
\mu_{C_t}=MAXPOOL\{\mu_v | v \in C_t\}\\
\mu_{S_t}=MAXPOOL\{\mu_v | v \in S_t\}\\
\mu_{C_t, S_t, v}=CONCAT(\theta_1·\mu_{C_t}, \theta_2·\mu_{S_t}, \theta_3·\mu_v)\\
\Q'_n(S_t, v; \theta_Q) = \theta_4 ·\mu_{C_t, S_t, v}\\
$$

### 6.3 GRIM

$\widehat{\sigma}_S$ 使用 GNN 预测的当前种子集 S 的影响力大小

$\widehat{\sigma}_s$ 使用 GNN 预测的节点 s 在初始时的影响力大小

$O(S,s) = \sum_{i=0}^n 1\{L_s^i - L_S^i \geqslant 0 \}$ 表示节点 s 加入后的边际影响力大小

$Q(u,S,G)=ReLU(ReLU([\widehat{\sigma}_S, \widehat{\sigma}_s, O(S, s)] W_k)W_q)$



## 7 节点特征向量

### 7.1 方案一

模型变为两阶段，

第一阶段，先通过 GNN 监督学习，学习每个节点的 score；

第二阶段，第一阶段的 score 将作为该阶段的节点特征之一。

#### 7.1.1 第一阶段

第一阶段节点的特征向量为：

1. 度

2. 接近中心性

3. 介数中心性
4. 聚类系数

GNN 监督学习节点 score 阶段，还可以有更多的设计，如粗粒度、细粒度预测。

#### 7.1.2 第二阶段

第二阶段节点的特征向量为：

1. score
2. 种子节点

$$
x_0(v) = 
\begin{cases}
1, \quad v \in S \\
0, \quad v \notin S \\
\end{cases}
$$



### 7.2 方案二

直接进行强化学习

1. d 跳邻居

$(𝑃^𝑇 )^𝑑$ 表示从任意种子到任意目标节点的所有长度为 𝑑 的路径

$(𝑃^𝑇 )^𝑑 x$ 表示所有节点的 $d$ 步影响上界

因此可以定义节点的特征矩阵为，$𝑋 = [𝒙,𝑃^𝑇𝒙,(𝑃^𝑇)^2𝒙...,(𝑃^𝑇)^𝑑𝒙]$

2. 种子节点

$$
x_0(v) = 
\begin{cases}
1, \quad v \in S \\
0, \quad v \notin S \\
\end{cases}
$$

综上，

节点的特征矩阵，$𝑋 = [x_0, x,𝑃^𝑇x,(𝑃^𝑇)^2x...,(𝑃^𝑇)^𝑑x]$

