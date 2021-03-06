# DyREP: Learing Representations over Dynamic Graphs

## 1 背景

提出了动态图上的的表示学习算法 DyREP，当前大部分算法都是直推式学习，即针对特定的网络环境所学习出的网络表示，一旦网络结构发生改变，直推式学习就需要进行重新训练。而本文提出的是归纳式学习的算法，不再学习节点的固定表示，而是学习节点的表示方法，这样即便结构改变，也能很容易得到新的节点表示。

学习节点的表示方法，即学习节点信息如何通过其邻居节点聚合而来的，学习到了相关的聚合函数，本身又知道节点的特征和邻居关系，就可以很容易得到新节点的表示。

该算法将网络中的演变视为两种情况:

- 一种是关联演变(association)，即拓扑结构发生变化；

- 另一种是社交演变(communication)，即节点发生交互行为。

该算法认为一个节点的表示由三个部分组成：

- 一是 Localized Embedding Propagaion，**局部嵌入传播**，即节点邻居的信息通过节点聚合后传播给其它节点，这一聚合的过程会对节点表示的形成产生影响；
- 二是 Self-Propagation，**自传播**，即节点在其所参与的两个事件发生的时间间隔中，可能会受之前表示的影响发生新的改变；
- 三是 Exogenous Drive，**外因驱动**，节点可能会受到全局的影响发生改变。

通过训练对应的参数，就能够得到相关节点表示，且这种表示是可以动态变化的。算法引入了注意力机制来不断更新节点邻居所占的权重比例，从而形成了不断更新的局部嵌入传播部分的嵌入，这是最为重要的一部分。因此，模型训练完成后，不仅可以根据时间的变化动态更新节点的表示，还可以对新的节点进行预测。

## 2 模型

### 2.1 动态图设定

$G_{t_0} = (V_{t_0} , E_{t_0} ) $

将图的演化观察为事件流 $O=\{(u,v,t,k)\}^{p}_{p=1}$

其中 {u,v} 代表参与事件的两个基点，t 代表时刻，p 是时间窗口 t 中排序好的 p 个观察到的事件集合，k 是一个尺度，k=0 代表拓扑结构演变(association)，k=1 代表交互行为(communication)。

只支持网络的增长

假设图中的一条边没有类型，节点没有属性

