# Survey

## 1 不同动态行为分类

- 拓扑演化（关于节点和边的添加或移除）

- 特征演化（关于节点/边特征或标签随时间的变化）

- 网络过程（节点的扩散和全局角色及其演化）



## 2 嵌入技术分类

- Factorization-based Methods 基于分解的方法
- ==Approaches based on Deep Learning 基于深度学习的方法==
  - **Encoder-Decoder Architecture 编码解码器**
    - (i) 传统自动编码器，其表示重建每个图快照，因此遵循**无损编码范式**；
      - DynGEM [34] 为每个图快照构建了一个完全连接的自动编码器，使用迁移学习范式在两个连续的自动编码器之间共享参数，以及一种允许自动编码器网络加宽其层并插入新层以处理越来越多的节点的策略。
      - LDANE [106] 遵循类似的策略，通过在损失函数中添加一个基于边际的排名损失项来处理节点属性，以确保两个相似节点的嵌入比两个非相似节点的嵌入更接近。
      - [95] 开发了 DyGGNN，它利用 GGNN 捕获图拓扑并将其与 LSTM 编码器耦合以处理图动态，并与 LSTM 解码器耦合以在每个时间戳重建动态图的结构。
      - DySAT [87] 和 DGNN [65]
    - (ii) 动态自编码器，其表示不重建每个图快照，而是重建未来时间戳中的快照，从而预测网络结构；
    - (iii) 鉴别器网络，其嵌入根本不重构网络拓扑，而是旨在学习节点标签、节点集群或全局网络属性，并且损失函数通常是应用驱动的。
  - Generative Models 生成模型
    - Based on Variational Autoencoders 基于变分自编码器
    - Based on Generative Adversarial Networks 基于生成对抗网络
- ==Random Walk Approaches 基于随机游走的方法==
  - **random walk on snapshots**
    - 为每个图快照生成对应的随机游走序列

  - evolving random walks
    - 增量更新节点表示

  - temporal random walks
    - 允许跨连续时间戳的随机游走并考虑时间排序限制来定义时间相关的上下文矩阵




## 3 基于随机游走的方法

### 3.1 random walk on snapshots

在动态图的每个快照上执行随机游走，通过考虑时间依赖性优化联合问题来获得向量表示。

时间依赖性是在嵌入生成的后期定义的。例如，可以使用包括 node2vec 和 DeepWalk 在内的方法独立地为每个图快照学习嵌入，然后，可以使用操作组合表示，从简单的向量连接到动态嵌入和正交变换，以在连续时间戳对齐嵌入向量。

![Survey-1](/Users/shiyingdan/GitHub/Blog/docs/subject/DyEmbedding/Survey-1.png)



### 3.2 Temporal Random Walk Methods

上一个方法考虑到随机游走，但它的更新是分别对每个快照进行的。

而这种方法将时间依赖性直接包含在随机游走生成的节点序列中，构建一种方法来创建随时间推移的游走语料库，并尊重时间流。在文献中，这些游走被视为时间游走。

扩散预测问题与时间随机游走有关，遵循这一目标的模型包括（i）DeepCas [57]，它使用 GRU 和注意力机制来预测级联的未来大小，（ii）DAN [104]，它利用前馈输出下一个受感染节点的概率分布神经网络和注意力机制，以及 (iii) Topo-LSTM [102]，使用 LSTM 来处理对扩散的时间依赖性。此外，杨等人。 [116] 使用 GRU、GCN 和 GraphSAGE 来预测下一个受影响的节点，并使用强化学习框架来预测级联大小。
时间随机游走代表了一种更自然的处理动态连续图 [21, 73] 的方法，因为它不需要将图任何时间离散化为快照。正如我们所展示的，这也是解决扩散问题的理想方法。下表列出了时间随机游走方法，并指出了应用于每个快照的静态嵌入方法以及它们**如何更新随机游走和向量表示**。

![Survey-1](/Users/shiyingdan/GitHub/Blog/docs/subject/DyEmbedding/Survey-2.png)



## 4 动态图嵌入应用

- 节点相关，包括节点分类、推荐系统和轨迹分析；
- 边缘相关，包括链路预测和事件时间预测；
- 图相关，包括随时间推移的图分类、网络可视化和**扩散预测**

### 4.1 图相关应用

(i) 序列预测问题 (即微观扩散问题) [53, 102, 104 , 116, 125];

(ii) 回归问题，它预测网络的未来数值特性 (例如，宏观扩散问题中受感染节点的总数) [57, 116]。





# 参考资料

> [DyREP](https://blog.csdn.net/CSDNTianJi/article/details/103844015?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~default-1.highlightwordscore&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~default-1.highlightwordscore)
>
> [DySAT](https://blog.csdn.net/Miha_Singh/article/details/114288854)





# paper list

- [ ] Combining tempo- ral aspects of dynamic networks with node2vec for a more efficient dynamic link prediction
  - Sam De Winter, Tim Decuypere, Sandra Mitrović, Bart Baesens, and Jochen De Weerdt
  - ASONAM‘ 2018
  - 为每个快照应用 node2vec，随着时间的推移将向量连接应用于节点表示

- [ ] Dyn2Vec: Exploiting dynamic behaviour using difference networks- based node embeddings for classification
  - Sandra Mitrovic and Jochen De Weerdt
  - In Proceedings of the International Conference on Data Science
  - 为每个快照应用 DeepWalk，随着时间的推移将向量连接应用于节点表示

- [ ] Dynamic network embeddings for network evolution analysis
  - Chuanchang Chen, Yubo Tao, and Hai Lin
  - 2019
  - 通过使用具有对角协方差的高斯先验来初始化节点嵌入，并使用动态伯努利嵌入随着时间的推移学习表示，将节点序列矩阵的行视为每个节点的上下文
  - [paper](https://arXiv:1906.09860)

- [ ] Dynamic network embedding by semantic evolution
  - Yujing Zhou, Weile Liu, Yang Pei, Lei Wang, Daren Zha, and Tianshu Fu
  - IJCNN' 19
  - DynSEM 使用 node2vec 为每个时间戳训练节点嵌入，使用正交 Procrustes 将节点嵌入对齐到公共空间，并在考虑时间平滑度的情况下优化联合损失函数。

- [ ] Continuous-time dynamic network embeddings
  - Giang Hoang Nguyen, John Boaz Lee, Ryan A. Rossi, Nesreen K. Ahmed, Eunyee Koh, and Sungchul Kim
  - In Proceedings of the Web Conference. International World Wide Web Conferences Steering Committee, 2018
  - 将 Skip-gram 架构推广到处理连续时间动态网络。提出了一个称为 CTDNE 的通用框架，用于学习时间保持嵌入，并提出了几种从起始节点选择后续节点的方法，从而执行时间随机游走：（i）无偏时间邻居选择； (ii) 有偏选择，这可能基于时间指数加权衰减（即，较旧的时间戳对选择的贡献呈指数降低）或时间线性加权衰减。
