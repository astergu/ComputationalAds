### 协同过滤 Collaborative Filtering

Amazon在2001年提出的用于推荐领域的一个算法，是推荐领域最经典的算法之一。

![CF图片](http://kubicode.me/img/Collaborative-Filtering-Meet-to-Deep-Learning/cf.png)

在实际场景中可以将用户对于Item的评分/购买/点击等行为 形成一张user-item的矩阵，单个的User或者Item可以通过对于有交互的Item和User来表示(最简单的就是One-Hot向量)，通过各种相似度算法可以计算到User2User、Item2Item以及User2Item的最近邻，先就假设按User2Item来说:

1. 和你购买相似宝贝的用户,其实和你相近的，也同时认为你们的习惯是相似的，
2. 因此他们买的其他的宝贝你也是可能会去够买的，这批宝贝就可以认为和你相似的

但是传统的CF会存在这两个问题:

1. 往往这个矩阵会非常稀疏，大部分稀疏程度在95%以上，甚至会超时99%，这样在计算相似度时就非常不准确了（置信度很低）
2. 整个求最近邻过程中会引入很多Trick，比如平滑、各种阈值等,最终将CF拿到效果还是比较难的。
3. 另外还有一个就是冷启动的问题，新用户或者新的item没法直接使用这种方式来计算。