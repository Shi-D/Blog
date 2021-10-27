# RW Sample

## 背景

随机游走采样是一种经典的图采样技术。

**算法思想**：随机选择一个节点 $v$ 进行随机游走，以 $p$ 的概率返回游走序列的前一个节点、以 $1-p$ 的概率继续随机游走。游走终止条件为设置的最长游走序列长度。



## 实验设置

### 1 数据集

 <img src="../RW_Sample_1.png" style="zoom:25%;" />

Epinions 和 Slashdot 为符号社交网络中最常用的两个数据集。



### 2 参数设置

$p = 0.15$

$RW\_Len=100$



### 3 评价指标

==采样前和采样后的图的属性比较：==

入度 In_degree

出度 Out_degree

正负边占比



## 实验结果

### 实验一 在 Slashdot 数据集上的效果

剪枝比设置为 0.7

入度：

 <img src="../RW_Sample_2.PNG" alt="RW_Sample_2" style="zoom:75%;" />

出度：

 <img src="../RW_Sample_3.PNG" alt="RW_Sample_3" style="zoom:75%;" />

正负边占比：

  <img src="../RW_Sample_6.PNG" alt="RW_Sample_6" />

 



### 实验二 在 Epinions 数据集上的效果

剪枝比设置为 0.5

入度：

 <img src="../RW_Sample_4.PNG" alt="RW_Sample_4" style="zoom:75%;" />

出度：

 <img src="../RW_Sample_5.PNG" alt="RW_Sample_5" style="zoom:75%;" />

正负边占比 

  <img src="../RW_Sample_7.PNG" alt="RW_Sample_7" />

## 实验结论

传统的随机游走图采样，在正负边占比上，采样后仍能保留原图的属性。