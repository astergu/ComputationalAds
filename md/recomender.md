- [推荐系统链路](#推荐系统链路)
- [召回](#召回)
  - [基于协同过滤的召回](#基于协同过滤的召回)
    - [相似性度量方法](#相似性度量方法)
      - [杰卡德（Jaccard）相似系数](#杰卡德jaccard相似系数)
      - [余弦相似度 (Cosine Similarity)](#余弦相似度-cosine-similarity)
      - [皮尔逊相关系数 (Pearson Correlation)](#皮尔逊相关系数-pearson-correlation)
      - [适用场景](#适用场景)
    - [UserCF](#usercf)
      - [基本思想](#基本思想)
    - [ItemCF](#itemcf)
      - [基本思想](#基本思想-1)
    - [协同过滤算法的问题分析](#协同过滤算法的问题分析)
    - [Swing](#swing)
      - [Surprise算法](#surprise算法)
    - [矩阵分解](#矩阵分解)
      - [隐语义模型](#隐语义模型)
  - [基于向量的召回](#基于向量的召回)
    - [FM召回](#fm召回)
    - [item2vec召回系列](#item2vec召回系列)
      - [word2vec原理](#word2vec原理)
        - [负采样 Negative Sampling](#负采样-negative-sampling)
      - [item2vec召回](#item2vec召回)
      - [Airbnb召回](#airbnb召回)
    - [Youtube DNN召回](#youtube-dnn召回)
- [粗排](#粗排)
- [精排](#精排)
  - [多目标排序模型](#多目标排序模型)
  - [Multi-gate Mixture-of-Experts (MMOE)](#multi-gate-mixture-of-experts-mmoe)
- [重排](#重排)
- [混排](#混排)
- [经典论文阅读](#经典论文阅读)
  - [Deep Neural Networks for YouTube Recommendations](#deep-neural-networks-for-youtube-recommendations)
- [参考](#参考)


# 推荐系统链路

- 召回：从推荐池中选取几千上万的item，送给后续的排序模块。由于召回面对的候选集十分大，且一般需要在线输出，故召回模块必须轻量快速低延迟。由于后续还有排序模块作为保障，召回不需要十分准确，但不可遗漏（特别是搜索系统中的召回模块）。目前基本上采用多路召回解决范式，分为非个性化召回和个性化召回。个性化召回又有content-based、behavior-based、feature-based等多种方式。
- 粗排：粗拍的原因是有时候召回的结果还是太多，精排层速度还是跟不上，所以加入粗排。粗排可以理解为精排前的一轮过滤机制，减轻精排模块的压力。粗排介于召回和精排之间，要同时兼顾精准性和低延迟。一般模型也不能过于复杂
- 精排：获取粗排模块的结果，对候选集进行打分和排序。精排需要在最大时延允许的情况下，保证打分的精准性，是整个系统中至关重要的一个模块，也是最复杂，研究最多的一个模块。精排系统构建一般需要涉及样本、特征、模型三部分。
- 重排：获取精排的排序结果，基于运营策略、多样性、context上下文等，重新进行一个微调。比如三八节对美妆类目商品提权，类目打散、同图打散、同卖家打散等保证用户体验措施。重排中规则比较多，但目前也有不少基于模型来提升重排效果的方案。
- 混排：多个业务线都想在Feeds流中获取曝光，则需要对它们的结果进行混排。比如推荐流中插入广告、视频流中插入图文和banner等。可以基于规则策略（如广告定坑）和强化学习来实现。


| 模块 | 输入输出 | 数据 | 策略 | 评价指标 | 
| ---- | ---- | ---- | ---- | ---- | 
| 召回 |  | | | |
| 粗排 | | | | |
| 精排 | | | | |
| 重排 | | | | |


# 召回

召回层的主要目标时从推荐池中选取几千上万的item，送给后续的排序模块。由于召回面对的候选集十分大，且一般需要在线输出，故召回模块必须轻量快速低延迟。

目前基本上采用多路召回解决范式，分为非个性化召回和个性化召回。个性化召回又有content-based、behavior-based、feature-based等多种方式。

召回主要考虑的内容有：

- 考虑用户层面：用户兴趣的多元化，用户需求与场景的多元化：例如：新闻需求，重大要闻，相关内容沉浸阅读等等
- 考虑系统层面：增强系统的鲁棒性；部分召回失效，其余召回队列兜底不会导致整个召回层失效；排序层失效，召回队列兜底不会导致整个推荐系统失效
- 系统多样性内容分发：图文、视频、小视频；精准、试探、时效一定比例；召回目标的多元化，例如：相关性，沉浸时长，时效性，特色内容等等
- 可解释性推荐一部分召回是有明确推荐理由的：很好的解决产品性数据的引入；

## 基于协同过滤的召回

协同过滤（Collaborative Filtering）推荐算法是最经典、最常用的推荐算法。基本思想是：

根据用户之前的喜好以及其他兴趣相近的用户的选择来给用户推荐物品。

- 基于对用户历史行为数据的挖掘发现用户的喜好偏向， 并预测用户可能喜好的产品进行推荐。
- 一般是仅仅基于用户的行为数据（评价、购买、下载等）, 而不依赖于项的任何附加信息（物品自身特征）或者用户的任何附加信息（年龄， 性别等）。

目前应用比较广泛的协同过滤算法是基于邻域的方法，主要有：

- 基于用户的协同过滤算法（UserCF）：给用户推荐和他兴趣相似的其他用户喜欢的产品。
- 基于物品的协同过滤算法（ItemCF）：给用户推荐和他之前喜欢的物品相似的物品。

不管是 UserCF 还是 ItemCF 算法， 重点是计算用户之间（或物品之间）的相似度。

### 相似性度量方法

#### 杰卡德（Jaccard）相似系数

Jaccard 系数是衡量两个集合的相似度一种指标，计算公式如下： $sim_{uv}=\frac{|N(u) \cap N(v)|}{|N(u)| \cup|N(v)|}$

- 其中，$N(u)$和$N(v)$ 分别表示用户$u$和用户$v$交互物品的集合。
- 对于用户$u$和$v$，该公式反映了两个交互物品交集的数量占这两个用户交互物品并集的数量的比例。

由于杰卡德相似系数一般无法反映具体用户的评分喜好信息，所以常用来评估用户是否会对某物品进行打分， 而不是预估用户会对某物品打多少分。

#### 余弦相似度 (Cosine Similarity)

余弦相似度衡量了两个向量的夹角，夹角越小越相似。余弦相似度的计算如下，其与杰卡德（Jaccard）相似系数只是在分母上存在差异： $sim_{uv}=\frac{|N(u) \cap N(v)|}{\sqrt{|N(u)|\cdot|N(v)|}}$ 从向量的角度进行描述，令矩阵$A$为用户-物品交互矩阵，矩阵的行表示用户，列表示物品。

#### 皮尔逊相关系数 (Pearson Correlation)

在用户之间的余弦相似度计算时，将用户向量的内积展开为各元素乘积和： $sim_{uv} = \frac{\sum_i r_{ui}*r_{vi}}{\sqrt{\sum_i r_{ui}^2}\sqrt{\sum_i r_{vi}^2}}$

- 其中，$r_{ui},r_{vi}$分别表示用户$u$和用户$v$对物品是否有交互(或具体评分值)。

相较于余弦相似度，皮尔逊相关系数通过使用用户的平均分对各独立评分进行修正，减小了用户评分偏置的影响。

#### 适用场景

- $Jaccard$相似度表示两个集合的交集元素个数在并集中所占的比例 ，所以适用于隐式反馈数据（0-1）。
- 余弦相似度在度量文本相似度、用户相似度、物品相似度的时候都较为常用。
- 皮尔逊相关度，实际上也是一种余弦相似度。不过先对向量做了中心化，范围在-1到1。
  - 相关度量的是两个变量的变化趋势是否一致，两个随机变量是不是同增同减。
  - 不适合用作计算布尔值向量（0-1）之间相关度。

### UserCF

User-based算法存在两个重大问题：

- 数据稀疏性
  - 一个大型的电子商务推荐系统一般有非常多的物品，用户可能买的其中不到1%的物品，不同用户之间买的物品重叠性较低，导致算法无法找到一个用户的邻居，即偏好相似的用户。
  - 这导致UserCF不适用于那些正反馈获取较困难的应用场景(如酒店预订， 大件物品购买等低频应用)。
- 算法扩展性
  - 基于用户的协同过滤需要维护用户相似度矩阵以便快速的找出$TopN$相似用户， 该矩阵的存储开销非常大，存储空间随着用户数量的增加而增加。
  - 故不适合用户数据量大的情况使用。

由于UserCF技术上的两点缺陷， 导致很多电商平台并没有采用这种算法， 而是采用了ItemCF算法实现最初的推荐系统。

#### 基本思想

基于用户的协同过滤（UserCF）：

- 例如，我们要对用户$A$进行物品推荐，可以先找到和他有相似兴趣的其他用户。
- 然后，将共同兴趣用户喜欢的，但用户$A$未交互过的物品推荐给$A$。

### ItemCF

#### 基本思想

基于物品的协同过滤（ItemCF）：

- 预先根据所有用户的历史行为数据，计算物品之间的相似性。
- 然后，把与用户喜欢的物品相类似的物品推荐给用户。

举例来说，如果用户1喜欢物品$A$，而物品$A$和$C$非常相似，则可以将物品$C$推荐给用户1。ItemCF算法并不利用物品的内容属性计算物品之间的相似度， 主要通过分析用户的行为记录计算物品之间的相似度， 该算法认为， 物品$A$和物品$C$具有很大的相似度是因为喜欢物品$A$的用户极可能喜欢物品$C$。


### 协同过滤算法的问题分析

协同过滤算法存在的问题之一就是泛化能力弱：

- 即协同过滤无法将两个物品相似的信息推广到其他物品的相似性上。
- 导致的问题是热门物品具有很强的头部效应， 容易跟大量物品产生相似， 而尾部物品由于特征向量稀疏， 导致很少被推荐。

协同过滤的天然缺陷：推荐系统头部效应明显， 处理稀疏向量的能力弱。

为了解决这个问题， 同时增加模型的泛化能力。2006年，矩阵分解技术(Matrix Factorization, MF)被提出：

- 该方法在协同过滤共现矩阵的基础上， 使用更稠密的隐向量表示用户和物品， 挖掘用户和物品的隐含兴趣和隐含特征。
- 在一定程度上弥补协同过滤模型处理稀疏矩阵能力不足的问题。


### Swing

**之前方法局限性**

- 基于 Cosine, Jaccard, 皮尔逊相关性等相似度计算的协同过滤算法，在计算邻居关联强度的时候只关注于 Item-based (常用，因为item相比于用户变化的慢，且新Item特征比较容易获得)，Item-based CF 只关注于 Item-User-Item 的路径，把所有的User-Item交互都平等得看待，从而忽视了 User-Item 交互中的大量噪声，推荐精度存在局限性。
- 对互补性产品的建模不足，可能会导致用户购买过手机之后还继续推荐手机，但用户短时间内不会再继续购买手机，因此产生无效曝光。

Swing 通过利用 User-Item-User 路径中所包含的信息，考虑 User-Item 二部图中的鲁棒内部子结构计算相似性。

$s(i,j)=\sum\limits_{u\in U_i\cap U_j} \sum\limits_{v \in U_i\cap U_j}w_uw_v \frac{1}{\alpha+|I_u \cap I_v|}$

其中$U_i$ 是点击过商品$i$的用户集合，$I_u$是用户$u$点击过的商品集合，$\alpha$是平滑系数。$w_u=\frac{1}{\sqrt{|I_u|}}$, $w_v=\frac{1}{\sqrt{|I_v|}}$是用户权重参数，来降低活跃用户的影响。

#### Surprise算法

首先在行为相关性中引入连续时间衰减因子，然后引入基于交互数据的聚类方法解决数据稀疏的问题，旨在帮助用户找到互补商品。互补相关性主要从三个层面考虑，类别层面，商品层面和聚类层面

### 矩阵分解

为了使得协同过滤更好处理稀疏矩阵问题， 增强泛化能力。从协同过滤中衍生出矩阵分解模型(Matrix Factorization, MF)或者叫隐语义模型：

- 在协同过滤共现矩阵的基础上， 使用更稠密的隐向量表示用户和物品。
- 通过挖掘用户和物品的隐含兴趣和隐含特征， 在一定程度上弥补协同过滤模型处理稀疏矩阵能力不足的问题。

#### 隐语义模型

隐语义模型最早在文本领域被提出，用于找到文本的隐含语义。在2006年， 被用于推荐中， 它的核心思想是通过隐含特征（latent factor）联系用户兴趣和物品（item）， 基于用户的行为找出潜在的主题和分类， 然后对物品进行自动聚类，划分到不同类别/主题(用户的兴趣)。

隐语义模型和协同过滤的不同主要体现在隐含特征上， 比如书籍的话它的内容， 作者， 年份， 主题等都可以算隐含特征。

## 基于向量的召回

### FM召回

### item2vec召回系列

#### word2vec原理

Word2vec(Mikolov et al. 2013)是一个用来学习dense word vector的算法：

- 我们使用大量的文本语料库
- 词汇表中的每个单词都由一个词向量dense word vector表示
- 遍历文本中的每个位置 t，都有一个中心词$c$（center） 和上下文词$o$（“outside”）
- 在整个语料库上使用数学方法最大化单词$o$在单词$c$周围出现了这一事实，从而得到单词表中每一个单词的dense vector
- 不断调整词向量dense word vector以达到最好的效果

Word2vec包含两个模型，Skip-gram与CBOW。

##### 负采样 Negative Sampling



#### item2vec召回

带有负采样（SGNS，Skip-gram with negative sampling）的 Skip-Gram 神经词向量模型在当时被证明是最先进的方法之一。

在论文[Item2Vec：Neural Item Embedding for Collaborative Filtering](https://arxiv.org/pdf/1603.04259)中，作者受到 SGNS 的启发，提出了名为Item2Vec的方法来生成物品的向量表示，然后将其用于基于物品的协同过滤。

Item2Vec 的原理很简单，就是基于 Word2Vec 的 Skip-Gram 模型，并且还丢弃了时间、空间信息。
- 基于 Item2Vec 得到物品的向量表示后，物品之间的相似度可由二者之间的余弦相似度计算得到。
- 可以看出，Item2Vec 在计算物品之间相似度时，仍然依赖于不同物品之间的共现。因为，其无法解决物品的冷启动问题。
  - 一种解决方法：取出与冷启物品类别相同的非冷启物品，将它们向量的均值作为冷启动物品的向量表示。

#### Airbnb召回

### Youtube DNN召回



# 粗排

粗排的原因是有时候召回的结果还是太多，精排层速度还是跟不上，所以加入粗排。粗排可以理解为精排前的一轮过滤机制，减轻精排模块的压力。粗排介于召回和精排之间，要同时兼顾精准性和低延迟。目前粗排一般也都模型化了，其训练样本类似于精排，选取曝光点击为正样本，曝光未点击为负样本。但由于粗排一般面向上万的候选集，而精排只有几百上千，其解空间大很多。

粗排阶段的架构设计主要是考虑三个方面，一个是根据精排模型中的重要特征，来做候选集的截断，另一部分是有一些召回设计，比如热度或者语义相关的这些结果，仅考虑了item侧的特征，可以用粗排模型来排序跟当前User之间的相关性，据此来做截断，这样是比单独的按照item侧的倒排分数截断得到更加个性化的结果，最后是算法的选型要在在线服务的性能上有保证，因为这个阶段在pipeline中完成从召回到精排的截断工作，在延迟允许的范围内能处理更多的召回候选集理论上与精排效果正相关。


# 精排

精排需要在最大时延允许的情况下，保证打分的精准性，是整个系统中至关重要的一个模块，也是最复杂，研究最多的一个模块。

精排是推荐系统各层级中最纯粹的一层，他的目标比较单一且集中，一门心思的实现目标的调优即可。最开始的时候精排模型的常见目标是ctr,后续逐渐发展了cvr等多类目标。精排和粗排层的基本目标是一致的，都是对商品集合进行排序，但是和粗排不同的是，精排只需要对少量的商品(即粗排输出的商品集合的topN)进行排序即可。因此，精排中可以使用比粗排更多的特征，更复杂的模型和更精细的策略（用户的特征和行为在该层的大量使用和参与也是基于这个原因）。

## 多目标排序模型

## Multi-gate Mixture-of-Experts (MMOE) 

# 重排

常见的有三种优化目标：Point Wise、Pair Wise 和 List Wise。重排序阶段对精排生成的Top-N个物品的序列进行重新排序，生成一个Top-K个物品的序列，作为排序系统最后的结果，直接展现给用户。重排序的原因是因为多个物品之间往往是相互影响的，而精排序是根据PointWise得分，容易造成推荐结果同质化严重，有很多冗余信息。而重排序面对的挑战就是海量状态空间如何求解的问题，一般在精排层我们使用AUC作为指标，但是在重排序更多关注NDCG等指标。

重排序在业务中，获取精排的排序结果，还会根据一些策略、运营规则参与排序，比如强制去重、间隔排序、流量扶持等、运营策略、多样性、context上下文等，重新进行一个微调。重排序更多的是List Wise作为优化目标的，它关注的是列表中商品顺序的问题来优化模型，但是一般List Wise因为状态空间大，存在训练速度慢的问题。

由于精排模型一般比较复杂，基于系统时延考虑，一般采用point-wise方式，并行对每个item进行打分。这就使得打分时缺少了上下文感知能力。用户最终是否会点击购买一个商品，除了和它自身有关外，和它周围其他的item也息息相关。重排一般比较轻量，可以加入上下文感知能力，提升推荐整体算法效率。比如三八节对美妆类目商品提权，类目打散、同图打散、同卖家打散等保证用户体验措施。重排中规则比较多，但目前也有不少基于模型来提升重排效果的方案。

# 混排

多个业务线都想在Feeds流中获取曝光，则需要对它们的结果进行混排。比如推荐流中插入广告、视频流中插入图文和banner等。可以基于规则策略（如广告定坑）和强化学习来实现。


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

- [工业界的推荐系统](https://github.com/wangshusen/RecommenderSystem)