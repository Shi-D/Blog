# Adaptive Influence Maximization in Dynamic Social Networks

## 0 论文信息

**Author**：Guangmo Tong

**Conference**：TON' 17



## 1 论文背景

现实社会网络中扩散过程具有不确定性: 

1. 种子用户不一定能成功激活；
2. 社交网络的拓扑结构是动态变化的。 

这篇论文研究的是具有上述特征的动态社交网络中的影响最大化问题。



## 2 DIC模型

将 IC 模型的 $G(V, E)$ 变为 $G(V, E, F_V, F_E)$ 即为 DIC (Dynamic IC) 模型。

其中，$ F_V$ 是 $f_v$ 的集合，$f_v$ 表示若将节点 $v$ 作为种子节点其被成功激活的概率；$F_E$ 是 $f_e$ 的集合，$f_e$ 表示边 e 被激活的概率 $D_e$ 的概率（概率的概率）。

$F_V、F_E$ 的加入，分别是 “论文背景” 中第1、2点在模型中的表现。 



## 3 辅助图 auxiliary graph

DIC 模型的一个辅助图，记作 $c-G = (V_c, E_c)$

辅助图 c-G 中，有 $N · B + N$ 个节点，















