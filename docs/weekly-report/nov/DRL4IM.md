# DRL for IM



## *1 Learning combinatorial optimization algorithms over graphs

**Conference**：==NeurIPS‘ 17==

**Author**：Elias Khalil, Hanjun Dai, Yuyu Zhang, Bistra Dilkina, and Le Song

**Model**：S2V-DQN

**Link**：[S2V-DQN](https://shiyingdan.top/subject/IMwithDRL/S2V-DQN/)



## *2 Combinatorial Optimization with Graph Convolutional Networks and Guided Tree Search

**Conference**：==NeurIPS‘ 18==

**Author**：Zhuwen Li, Qifeng Chen, and Vladlen Koltun

**Abstract**：用 GCN 得到解的质量，再用传统的 Guided Tree Search 进行解的选择。

**Model**：GCN-TreeSearch



## 3 Influence maximization in unknown social networks: Learning policies for effective graph sampling

**Conference**：AAMAS‘ 20

**Author**：Harshavardhan Kamarthi, Priyesh Vijayan, Bryan Wilder, Balaraman Ravindran, and Milind Tambe

**Abstract**：应用 RL 在影响最大化的背景下探索**未知图**，这项工作其实不太一样，因为他们在是用 RL 来探索图结构而不是选择种子。



## 4 Monstor: An inductive approach for estimating and maximizing influence over unseen social networks

**Conference**：ASONAM‘ 2020.

**Author**：Jihoon Ko, Kyuhan Lee, Kijung Shin, and Noseong Park

**Abstract**：提出了一种归纳 ML 方法来估计看不见的网络的影响传播。



## *5 Disco: Influence maximization meets network embedding and deep learning

**arXiv preprint**：2019

**Author**：Hui Li, Mengting Xu, Sourav S Bhowmick, Changsheng Sun, Zhongyuan Jiang, and Jiangtao Cui

**Abstract**：强化学习 + Structure2Vec 解决影响力最大化问题。

**创新点**： 对大型社交网络的预处理、对动态网络的预处理、测试阶段直接取前 k 个节点而无需迭代选取。

**Model**：Disco

**Link**：[Disco](https://shiyingdan.top/weekly-report/sept/DISCO/)



## *6 Deep reinforcement learning-based approach to tackle topic-aware influence maximization

**Journal**：Data Science and Engineering, 2020

**Author**：Shan Tian, Songsong Mo, Liwei Wang, and Zhiyong Peng

**Abstract**：用图嵌入+强化学习解决主题感知 (topic-aware) 影响力最大化

**Model**：DIEM

**Link**：[DIEM](https://shiyingdan.top/weekly-report/sept/DIEM/)



## *7 Gcomb: Learning budget-constrained combinatorial algorithms over billion-sized graphs

**Conference**：==NeurIPS‘ 20==

**Author**：Sahil Manchanda, Akash Mittal, Anuj Dhawan, Sourav Medya, Sayan Ranu, and Ambuj Singh

**Abstract**：用概率贪婪进行剪枝，用 GNN 监督学习每个节点的 score 作为深度强化学习中 Q 函数的构成之一。

**存在的问题**：旨在解决具有数百万个节点的问题实例。它使用监督学习作为预测节点的个体质量的初步步骤，这引入了大量额外的计算开销和手工制作学习管道的工作。因此，它无法扩展到大量训练图。

**Model**：GCOMB

**Link**：[GCOMB](https://shiyingdan.top/subject/IMwithDRL/GCOMB/)



## *8 Contingency-Aware Influence Maximization: A Reinforcement Learning Approach

**Conference**：UAI‘ 20

**Author**：Haipeng Chen, Wei Qiu, Han-Ching Ou, Bo An, Milind Tambe

**Model**：RL4IM

**Link**：[RL4IM](https://shiyingdan.top/subject/IMwithDRL/Contingency/)



## *9 Learning to Maximize Influence

**arXiv**：2021.10.6

**Author**：George Panagopoulo, Nikolaos Tziortziotis, Fragkiskos D. Malliaros, Michalis Vazirgiannis

**Abstract**：用 GNN 对节点的影响力大小进行预测，对排好序的节点用 RL 进行节点选择（决策过程）

**Model**：GRIM

**Link**：[GRIM](https://shiyingdan.top/weekly-report/nov/GLIE/)

