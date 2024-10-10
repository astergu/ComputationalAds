ctr, cvr, click/install/purchase

# 基础概念

- Bidding 计费模式
  - CPA (Cost Per Action)
    - 按照行为（Action）来计费
  - CPC（Cost Per Click）
    - 按照点击次数（Click）来计费
- Evaluation 衡量指标
  - CTR 点击率 $=\frac{No.Clicks}{No.Impressions}$
  - CVR 转化率 $=\frac{No.Conversions}{No.Clicks}$


# 经典论文

| Model | Paper | Affiliation |	Key Takeaways |
| ---- | ---- | ---- | ---- |
| LR+GBDT | [Practical lessons from predicting clicks on ads at facebook](https://research.facebook.com/file/273183074306353/practical-lessons-from-predicting-clicks-on-ads-at-facebook.pdf) (ADKDD 2014) | Meta | `LR+GBDT` <br> 1. Data freshness很重要，模型至少每天训练一次; <br> 2. 使用boosted decision tree进行特征转换提高了模型性能; <br> 3. 在线学习：LR+per-coordinate learning rate |
| DNN  | [Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf) (RecSys 2016) | Youtube |  |
| | | | |
| Wide&Deep | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf) (DLRS 2016) | Google | 	1. Wide模型提供记忆能力；<br> 2. Deep模型提供泛化能力； <br> 3. Wide&Deep联合训练 |
| DeepFM | [DeepFM: An End-to-End Wide & Deep Learning Framework for CTR Prediction](http://www.ijcai.org/proceedings/2017/0239.pdf) (IJCAI 2017) | Huawei | |
| | [Real-time Personalization using Embeddings for Search Ranking at Airbnb] (KDD 2018) | Airbnb | |
| | [Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate] | | |
| | [Real-time Personalization using Embeddings for Search Ranking at Airbnb] | | |
| | [Practice on Long Sequential User Behavior Modeling for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09248.pdf) (2019) | Alibaba | |
| SIM | [Search-based User Interest Modeling with Sequential Behavior Data for CTR Prediction](https://arxiv.org/pdf/2006.05639.pdf) (2020) | Alibaba | |
| | [Using deep learning to detect abusive sequences of member activity](https://engineering.linkedin.com/blog/2021/using-deep-learning-to-detect-abusive-sequences-of-member-activi) ([Video](https://exchange.scale.com/public/videos/using-deep-learning-to-detect-abusive-sequences-of-member-activity-on-linkedin)) (2021) | LinkedIn | |
| | | | |
| | | | |
| | | | |
| | | | |