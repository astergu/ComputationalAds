- [计算广告 Computational Advertising](#计算广告-computational-advertising)
  - [书籍 Books](#书籍-books)
  - [论文](#论文)
  - [技术文章](#技术文章)
    - [CTR预估](#ctr预估)


# 计算广告 Computational Advertising

- [点击率预估CTR](./ctr_model.md)
- [重排 Rerank](./rerank.md)



## 书籍 Books 

- **计算广告：互联网商业变现的市场与技术**，刘鹏，王超著

## 论文

1. 计算广告领域的Paper List: [https://github.com/wnzhang/rtb-papers/](https://github.com/wnzhang/rtb-papers/)
2. [Predicting Clicks: Estimating the Click-Through Rate for New Ads](./papers/PredictingClicks.md)
3. [Web-Scale Bayesian Click-Through Rate Prediction for Sponsored Search Advertising in Microsoft’s Bing Search Engine](./papers/BayesianCTR.md)
4. [Practical Lessons from Predicting Clicks on Ads at Facebook](./papers/PracticalFacebook.md)
5. [Greedy function approximation:a gradient boosting machine]
6. [Ad Click Prediction: a View from the Trenches](./papers/AdClickPrediction_Google_KDD2013.md)
7. [Factorization Machines]
8. [An Empirical Evaluation of Thompson Sampling]
9. [Content-based Recommendation Systems]
10. [Click-Through Rate Estimation for Rare Events in Online Advertising]
11. [Wide & Deep Learning for Recommender System]
12. [Deep Neural Networks for YouTube Recommendations]
13. [Predictive Model Performance: Offline and Online Evaluations]
14. [Personalized Click Prediction in Sponsored Search](./papers/coec.md)


## 技术文章 

### CTR预估

1. Google在CTR预估领域的工程实践经验 [Ad Click Prediction: A View From the Trenches](papers/AdClickPrediction_Google_KDD2013.md)
2. Google的Wide&Deep [Wide & Deep Learning for Recommender Systems](papers/Wide_Deep.md)
3. InfoQ：从逻辑回归到深度学习，点击率预测技术面面观 [https://www.infoq.cn/article/click-through-rate-prediction](https://www.infoq.cn/article/click-through-rate-prediction)
4. Collaborative Filtering: [http://kubicode.me/2019/01/16/Deep%20Learning/Collaborative-Filtering-Meet-to-Deep-Learning/#more](http://kubicode.me/2019/01/16/Deep%20Learning/Collaborative-Filtering-Meet-to-Deep-Learning/#more)
5.  Ad Click Prediction: a View from the Trenches笔记[https://blog.csdn.net/fangqingan_java/article/details/51020653](https://blog.csdn.net/fangqingan_java/article/details/51020653)
6. 知乎：广告ctr预估有什么值得推荐的论文？[https://www.zhihu.com/question/26154847](https://www.zhihu.com/question/26154847)
7. 知乎：主流CTR预估模型的演化及对比 [https://zhuanlan.zhihu.com/p/35465875](https://zhuanlan.zhihu.com/p/35465875)
8. Github上关于计算广告的整理
    - [https://github.com/wzhe06/Ad-papers](https://github.com/wzhe06/Ad-papers)
    - [https://github.com/wnzhang/rtb-papers/](https://github.com/wnzhang/rtb-papers/)
9. 代码实现
    - LR, GBDT，RF，NN，PNN等几种CTR预估模型的spark实现 [https://github.com/wzhe06/CTRmodel](https://github.com/wzhe06/CTRmodel)
    - 可以直接使用的Deep CTR模型：DeepFM, DIN, DIEN, DCN, AFM, NFM, AutoInt, DSIN [https://github.com/shenweichen/DeepCTR](https://github.com/shenweichen/DeepCTR)