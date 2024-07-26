- [Recommender Systems](#recommender-systems)
  - [Candidate Generation](#candidate-generation)
  - [Ranking](#ranking)
  - [Re-ranking](#re-ranking)
  - [Calibration](#calibration)
- [经典论文阅读](#经典论文阅读)
  - [Deep Neural Networks for YouTube Recommendations](#deep-neural-networks-for-youtube-recommendations)
- [参考](#参考)

# Recommender Systems

Here is a common architecture for a recommendation systems model:

![](https://aman.ai/recsys/assets/recsys/1.jpg)

- Candidate Generation: This is where the model starts from. It has a huge corpus and generates a smaller subset of candidates.
For e.g., Netflix has thousands of movies, but it has to pick which ones to show you on your home page!
The model needs to evaluate the queries at a rapid speed on a large corpus and keep in mind, a model may provide multiple candidate generators which each having different candidate subsets.
- Scoring: This is the next step in our architecture. Here, we simply rank the candidates in order to select a set of documents.
Here the model runs on a smaller subset of documents, so we can aim for higher precision in terms of ranking.
- Re-ranking: The final step in our model is to re-rank the recommendations.
For e.g., if the user explicitly dislikes some movies, Netflix will need to move those out of the main page.

## Candidate Generation

## Ranking

## Re-ranking

## Calibration


# 经典论文阅读

| Paper | Affiliation | Key Takeaways |
| ---- | ---- | ---- |
| [Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf) (RecSys 2016) | YouTube | |
| [Neural Collaborative Filtering](https://arxiv.org/pdf/1708.05031) (WWW 2017) | | |
| [Collaborative Memory Network for Recommendation Systems](https://arxiv.org/pdf/1804.10862) (SIGIR 2018) | | |
| [Real-time Personalization using Embeddings for Search Ranking at Airbnb](https://dl.acm.org/doi/pdf/10.1145/3219819.3219885) (KDD 2018) | Airbnb | |

## Deep Neural Networks for YouTube Recommendations


- **主要贡献点**
- **背景**
  - Youtube的视频推荐面临三个主要的挑战：
    - `体量(Scale)大`：视频量大，用户多
    - `新鲜度(Freshness)的需求`：需要平衡高质量视频和新近更新的视频之间的推荐频率
    - `噪音（Noise）`：用户行为稀疏且多样，而且推荐的结果大部分都不能得到显式的反馈，而是充满噪音的隐式反馈（比如观看时长，点赞等）
  - 推荐可以分为两个阶段：
    - **候选生成（candidate generation）**
      - `Goal`：一定的个性化推荐
      - `Input`: 用户行为历史
      - `Output`：一个小的视频集合 (hundreds)
      - `Features`：用户视频浏览历史，搜索词，人口统计学特征等
    - **排序（ranking）**
      - `Goal`：对候选生成阶段获取的视频集合进行打分
      - `Input`：候选生成获取的视频集合
      - `Output`：带分数的视频集合，一般输出最高分的商品作为结果返回
      - `Features`：用户特征，商品特征
- **线下评估**
  - Precision
  - Recall
  - Ranking loss
- **线上评估**
  - A/B testing
  - click-through rate, watch time, etc.

![architecture](../image/dnn_youtube_architecture.png)

- **候选生成 Candidate Generation**
  - 这里把推荐问题看做是一个多分类问题（multiclass classification），预测在时间$t$基于当前用户$U$和上下文$C$对从视频库$V$里选出的一个特定视频的观看时长$w_t$。其中，$u\in \mathbb{R}^N$表示用户和上下文的embedding，而$v_j\in \mathbb{R}^N$表示每个候选视频的embedding。
    - $P(w_t=i|U,C)=\frac{e^{v_i u}}{\sum_{j\in V}e^{v_j u}}$
- **候选采样 candidate sampling**
  - Sample negative classes from the background distribution and then correct for this sampling via importance weighting.
  - For each example, the cross-entropy loss is minimized for the true label and the sampled negative classes.
  - Serveral thousand negatives are sampled, corresponding to more than 100 times speedup over traditional softmax.
  - At serving time, compute the most likely $N$ classes (videos) in order to choose the top $N$ to represent to the user. 
  - Since calibrated likelihoods from the softmax output layer are not needed at serving time, the scoring problem reduces to a nearest neighbor search in the dot product space.

![candidate generation](../image/youtube_candidate_generation.png)
- **模型**
  - 简单来说，把用户的观看历史（一堆稀疏的视频ID）映射到稠密的embeddings，平均以后得到用户的观看embedding。同样的，用户的搜索历史在经过tokenizer之后，平均得到用户的搜索embedding。另外，还可以考虑用户的人口统计学特征（地区，设备，性别，登录状态，年龄等）。把所有特征连接起来，形成一个第一层（wide first layer），然后再经过若干层ReLU（Rectified Linear Units）。
  - 样本年龄 Example Age
    - 一般来说，机器学习模型会天然地偏向很早以前上传的视频，因为有更多的互动历史，但是作为一个视频网站，YouTube需要推荐新上传的内容。
    - 因此，在训练时，把样本（视频）的年龄也作为一个特征送入模型。在预测时，这个特征设置为0，或者是一个很小的负数。
  -  排序 Ranking
     -  特征提取 Feature Representation
        -  特征分为categorical和continuous/ordinal特征。同时，把特征分类为impression特征（有关item的）和query特征（有关用户的）。query特征只需要在per request计算一次，而impression特征需要每个item计算一次。
        -  据观察，最有用的特征是描述该用户与当前item或者类似item的历史交互信息，比如用户看过该item所在频道的多少个视频，上一次用户看相关主题的视频是什么时候。另外，历史impression的频率也是个有效地特征，比如用户之前被推荐过某视频，但是他没有点积，那么之后会降低这个视频的推荐概率。
![ranking](../image/youtube_ranking.png)
        - 建模类别特征（Embedding Categorical Features）
        - 正则化连续特征（normalizing Continuous Features）
        - 神经网络对输入的scaling和distribution比较敏感，因此需要对连续特征做适当的正则化
        - 连续特征最终会被映射到[0,1)的范围
        - 除了常规正则化$\tilde{x}$，还会引入$\tilde{x}^2$和$\sqrt{\tilde{x}}$
        - 建模预期观看时长（Modeling Expected Watch Time）
     - 排序的目标是为了预测预期观看时长，我们通过`weighted logistic regression`模型来预测时长

- **实验结果**
  - 结果显示增加隐层的宽度和深度都有助于提升效果，但是考虑到线上serving的CPU时间，实际配置采用1024-wide ReLU + 512-wide ReLU + 256-wide ReLU。

![experiments](../image/youtube_experiments.png)

# 参考

- [Aman.ai Recommendation Systems](https://aman.ai/recsys/index.html)
- [工业界的推荐系统](https://github.com/wangshusen/RecommenderSystem)