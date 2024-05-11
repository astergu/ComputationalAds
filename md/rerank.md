- [Rerank 重排](#rerank-重排)
  - [重排的目标](#重排的目标)
  - [Blending Advertising with Organic Content in E-Commerce via Virtual Bids](#blending-advertising-with-organic-content-in-e-commerce-via-virtual-bids)


# Rerank 重排

排序系统一般包括召回（亿级）、粗排（十万级）、精排（千级）、重排序（百级）等多个阶段。精排阶段完成对物品进行pointwise的打分，计算出精排排序分数，然后按分数降序排列，生成Top-N个物品的序列。

重排序阶段对精排生成的Top-N个物品的序列进行重新排序，生成一个Top-K个物品的序列，作为推荐系统最后的结果，直接展现给用户。

## 重排的目标
- 短期目标
  - 提升结果的效率（点击、购买、GMV等）；提升结果的多样性、发现性和用户体验；降低负反馈（结果同质化严重、看/点/买了还推）。
- 长期目标
  - 提升用户流量深度、停留时长；提升用户复访、留存、长期用户价值。
- 面临的挑战
  - 重排序面临的主要挑战包括：
    - 解空间大。重排序问题是从Top-N个商品的排列，生成Top-K个商品的排列，解空间（$A_N^K$）为指数级，生成最优解为NP-hard问题。
    - 每种Top-K个商品的排列的reward难以预测。
    - 海量的状态空间（用户状态）和解空间，导致难以有效探索和学习。

## Blending Advertising with Organic Content in E-Commerce via Virtual Bids

- 目标
  - 兼顾平衡广告收入与点击，以及其他商业指标（i.e.organic CTR, diversity）
    - $max\sum_i bid_i\times pCTR_i^{ads}+\sum_i VB^{ads}\times pCTR_i^{ads}+...\sum_i VB^{orgs}\times pCTR_i^{orgs}$
- Golden Search 黄金搜索法：适合搜索一个虚拟出价
- SPSA：适合同时搜索多个虚拟出价
- 混排（Rerank）大体分为四个过程：
  - 序列生成
    - 满足多样性策略（包含类目以及素材）的前提下，生成待评估的序列
    - 生成策略
      - 排列组合策略：从原始广告序列的前$N$个结果选出前$K$个（$K$为广告坑位数，$N$可配置且$N\ge K$）排列组合，生成序列
      - 启发式生成策略（确保符合多样性打散策略的原则）
        - 贪心策略：基于不同的超参逐坑位挑选广告，生成序列
          - 超参分别为$\lambda$和$t$，公式如下：
            - $\lambda\times pctr + pctr \times bid$
            - $pctr^t \times bid$
        - 随机采样策略：基于广告质量分（综合考量了点击、消费、转化、多样性等）逐坑位进行采样，生成序列
  - 序列评估
    - 目标是对所有待评估的序列，根据序列内的排列顺序，对序列内的各item进行pctr预估
  - 序列选优
    - 目标是从所有评估完毕的序列结合中选取最优的序列，作为最终的展示序列。评优结果综合考虑广告和推荐的指标
      - $max\sum_i bid_i\times pCTR_i^{ads} + \sum_i VB^{ads}\times pCTR_i^{ads} + ...\sum_i VB^{orgs}\times pCTR_i^{orgs}$
  - 计费调整
    - 由于序列评估方案是期望序列整体最优，并不能保证严格的GSP规则，因此采用VCG算法对广告序列整体进行计费