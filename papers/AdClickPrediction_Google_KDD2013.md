# Ad Click Prediction : a View from the Trenches 工程实践视角下的广告点击率预估

这是Google于2013年在KKD上发表的论文。2013年，也就是这篇论文发表的时候，当时大规模深度学习的环境还没有完全成熟起来，Google的科学家和工程师们选择了逻辑回归，这是一个非常传统但也非常强大的线性分类工具。

## 核心观点 Key Takeaways

- 核心技术点
    - 
- 核心工程点
    - 节省内存
    - 性能评估和可视化方法
    - 预估概率的置信估计
    - 校准方法
    - 特征自动管理
- 没有用的尝试
    - Aggressive Feature Handling
    - Dropout
    - Feature bagging
    - Feature Normalizatiion

## 问题定义

点击率预估