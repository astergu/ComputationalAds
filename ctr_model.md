



# 主流CTR模型演化

CTR预估本质是一个二分类问题，以移动端展示广告推荐为例，依据日志中的用户侧的信息（比如年龄，性别，国籍，手机上安装的app列表）、广告侧的信息（广告id，广告类别，广告标题等）、上下文侧信息（渠道id等），去建模预测用户是否会点击该广告。

![architect](http://aistar.site/image001.png)

![algorithms](./algorithms.jpg)



## 深度CTR模型的基本框架

典型的深度CTR模型可以分成以下四个部分：输入、特征嵌入（Embedding）、特征交互（有时候也称为特征提取）和输出。

- **输入**

输入通常包含若干个<特征ID, 特征值>对，当然也可以One-Hot Encoding展开。

- **特征嵌入（Embedding）**

在CTR任务中数据特征呈现高维、稀疏的特点，假设特征数为N，直接将这些特征进行One-Hot Encoding构造二阶及以上特征时候会产生巨大的参数数量，以FM的二阶项为例子，如一万个特征，两两构造二阶特征时将会产生一亿规模的特征权重参数。

Embedding可以减小模型复杂度，具体过程如下：

通过矩阵乘法将1\*N的离散特征向量通过维度为N\*k的参数矩阵W压缩成1\*k的低维度稠密向量，通常k<<N，参数从N^2降到N\*k。

- **特征交互**

经过特征嵌入可以获得稠密向量，在特征交互模块中设计合理的模型结构将稠密向量变成标量，该模块直接决定模型的质量好坏。

- **输出**

将特征交互模块输出的标量用sigmoid函数映射到[0, 1]，即表示CTR。


## 主流算法

### 逻辑回归 Logistic Regression

LR一直是CTR预估的benchmark模型，具有简单、易于并行化实现、可解释性强等优点，但是LR模型中的特征是默认相互独立的，遇到具有交叉可能性的特征需进行大量的人工特征工程进行交叉(连续特征的离散化、特征交叉)，不能处理目标和特征之间的非线性关系。LR将特征加权求和并经sigmoid即得到CTR值。


### LR + GBDT

GBDT(Gradient Boost Decision Tree)是用来解决LR模型的特征组合问题。GBDT可以用来学习高阶非线性特征组合。对应树的一条路径。通常将一些连续值特征、值空间不大的categorical特征都丢给GBDT模型；空间很大的ID特征留在LR模型中训练，既能做高阶特征组合又可以利用线性模型易于处理大规模稀疏数据的优势。

![formula](https://www.zhihu.com/equation?tex=f%28x%29%3Dlogistics%28gbdt%5C_tree_1%28X%29%2Bgbdt%5C_tree_2%28X%29%2B...%29)

GBDT优势在于处理连续值特征，如用户历史点击率、用户历史浏览次数等连续值。由于树的分裂算法，具有一定组合特征的能力。GBDT根据最优的分裂特征和该特征的最优分裂点，根据特征的分裂次数得到一个特征的重要性排序，GBDT减少了人工特征工程的工作量。

但是大多数推荐系统中出现的是大规模的离散化特征，使用GBDT需要首先统计成连续值特征(embedding)，需要耗费时间，GBDT具有记忆性强的特点，不利于挖掘长尾特征。而且GBDT虽然具有一定组合特征能力，但是组合的能力十分有限，远不能与DNN相比。

### Deep Interest Network (DIN)

用户场景很简单，就是在一个电商网站或APP找那个给用户推荐广告。

注意力机制，就是模型在预测的时候，对用户不同行为的注意力是不一样的。

![注意力机制](https://pic4.zhimg.com/v2-b8251f4d2a41f1a7de359c330a355530_1440w.jpg?source=172ae18b)

#### 贡献点

- 用GAUC代替AUC
- 用Dice方法代替经典的PReLU激活函数
- 介绍一种Adaptive的正则化方法



#### 用户行为



### FM/FFM

#### FM

与LR相比，FM增加了二阶项的信息，通过穷举所有的二阶特征（一阶特征两两组合）并结合特征的有效性（特征权重）来预测点击结果，FM的二阶特征组合过程可拆分成Embedding和内积两个步骤。GBDT虽然可以学习特征交叉组合，但是只适合中低度稀疏数据，容易学到高阶组合。但是对于高度稀疏数据的特征组合，学习效率很低。另外GBDT也不能学习到训练数据中很少或者没有出现的特征组合。但是FM（因子分解机，Factorization Machine）可以通过隐向量的内积提取特征组合，对于很少或没出现的特征组合也可以学习到。

FM的优点就是具有处理二次交叉特征的能力，而且可以实现线性复杂度O(n)，模型训练速度快。

#### FFM (Field-aware Factorization Machine)

FFM引入了field概念，FFM将相同性质的特种归于同一个field。同一个categorical特种经过one-hot编码生成的数值特种都可以放入同一个field。

![ffm image](https://www.zhihu.com/equation?tex=f%28x%29%3Dlogistic%28%5CTheta%5ETX%2B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Csum_%7Bj%3Di%2B1%7D%5E%7Bn%7D%7B%3Cv_%7Bi%2Cf_j%7D%2Cv_%7Bj%2Cf_i%7D%3Ex_ix_j%7D%29)

![ffm model](https://www.zhihu.com/equation?tex=%5Cphi%28W%2CX%29%3D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bw_ix_i%7D%2B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Csum_%7Bj%3Di%2B1%7D%5E%7Bn%7D%7B%28v_%7Bi%2Cf_j%7D%C2%B7v_%7Bj%2Cf_i%7D%29x_ix_j%7D)

FFM模型使用**logistic loss**作为损失函数+L2正则项：

![logistic loss](https://www.zhihu.com/equation?tex=L%3D%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dlog%281%2Bexp%28-y_i%5Cphi%28w%2Cw_i%29%29%29%2B%5Cfrac%7B%5Clambda%7D%7B2%7D%7C%7CW%7C%7C%5E2)

FM是把所有特征都归属于一个field时的FFM模型。

**FFM模型训练时的注意事项**

- **样本归一化**。FFM默认是进行样本数据的归一化的 。若不进行归一化，很容易造成数据inf溢出，进而引起梯度计算的nan错误。因此，样本层面的数据是推荐进行归一化的。

- **特征归一化**。CTR/CVR模型采用了多种类型的源特征，包括数值型和categorical类型等。但是，categorical类编码后的特征取值只有0或1，较大的数值型特征会造成样本归一化后categorical类生成特征的值非常小，没有区分性。例如，一条用户-商品记录，用户为“男”性，商品的销量是5000个（假设其它特征的值为零），那么归一化后特征“sex=male”（性别为男）的值略小于0.0002，而“volume”（销量）的值近似为1。特征“sex=male”在这个样本中的作用几乎可以忽略不计，这是相当不合理的。因此，将源数值型特征的值归一化到[0,1]是非常必要的。

- **省略零值特征**。从FFM模型的表达式可以看出，零值特征对模型完全没有贡献。包含零值特征的一次项和组合项均为零，对于训练模型参数或者目标值预估是没有作用的。因此，可以省去零值特征，提高FFM模型训练和预测的速度，这也是稀疏样本采用FFM的显著优势。

### GBDT+(LR,FM,FFM)

GBDT适合处理连续值特征，而LR、FM、FFM更加适合处理离散化特征。GBDT可以做到一定程度的特征组合，而GBDT的特征组合是多次组合而不仅是与FM和FFM这样的二阶组合而已。GBDT具备一定的特征选择能力（选择最优的特征进行分裂）。

### DNN

在ctr预估场景中，绝大多数特征都是大规模离散化特征，并且交叉类的特征十分重要，如果利用简单的模型如LR的话需要大量的特征工程，即使是GBDT，FM这种具有一定交叉特征能力的模型，交叉能力十分有限，脱离不了特征工程。

DNN具有很强的模型表达能力，有以下优势：

- 模型表达能力强，能够学习出高阶非线性特征。
- 容易扩充其他类别的特征，如特征是图片或文字类时。

### Embedding+MLP

多层感知机MLP因具有学习高阶特征的能力常常被用在各种深度CTR模型中。MLP主要由若干个全连接层和激活层组成。

### Wide&Deep

将LR和MLP并联即可得到Wide&Deep模型，可同时学习一阶特征和高阶特征。

### DeepFM 

DeepFM是为了解决DNN的不足而推出的一种并行结构模型。将LR、MLP和Quadratic Layer并联可得到DeepFM，注意到MLP和Quadratic Layer共享Group Embedding。DeepFM是目前效率和效果上都表现不错的一个模型。

### DCN: Deep & Cross Network

将LR、MLP和Cross Net并联可得到DCN。

Cross Net是一个堆叠型网络，该部分的初始输入是将f个(1,k)的特征组向量concat成一个(1,f\*k)的向量（不同特征组的嵌入维度可以不同，反正拼起来就对了）。

每层计算过程如下：输入向量和初始输入向量做Cartesian product得到(f\*k,f\*k)的矩阵，再重新投影成(1,k)向量，每一层输出都包含输入向量。

### xDeepFM

将LR、MLP和CIN并联可得到xDeepFM。

CIN也是一个堆叠型网络，该部分的初始输入是一个(f,k)的矩阵，

每层计算过程如下：输入矩阵(Hi, k)和初始输入矩阵沿嵌入维度方向做Cartesian product得到(Hi, f, k)的三维矩阵，再重新投影成(Hi+1,k)矩阵。

CIN的最后一层：将CIN中间层的输出矩阵沿嵌入维度方向做sum pooling得到(H1,1),(H2,1)...(Hl,1)的向量，再将这些向量concat起来作为CIN网络的输出。

### DIN

该模型基于对用户历史行为数据的两个观察：1、多样性，一个用户可能对多种品类的东西感兴趣；2、部分对应，只有一部分的历史数据对目前的点击预测有帮助，比如系统向用户推荐泳镜时会与用户点击过的泳衣产生关联，但是跟用户买的书就关系不大。于是，DIN设计了一个attention结构，对用户的历史数据和待估算的广告之间部分匹配，从而得到一个权重值，用来进行embedding间的加权求和。


## 代码实现 

### LR实现

SGD classifier分类器

```python
# Linear classifiers (SVM, logistic regression, etc.) with SGD training.
dict_one_hot_encoder = DictVectorizer(sparse=False)
X_train = dict_one_hot_encoder.fit_transform(X_dict_train)
sgd_log_reg_model = SGDClassifier(loss='log', penalty=None, fit_intercept=True, learning_rate='constant', eta0=0.01)
sgd_log_reg_model.fit(X_train, y_train)
predictions = sgd_log_reg_model.predict_proba(X_test)[:, 1]
score = roc_auc_score(y_test, predictions)
```

LR online learning

```python
sgd_log_reg_model = SGDClassifier(loss='log', penalty=None, fit_intercept=True, learning_rate='constant', eta0=0.01)
X_dict_train, y_train = process_data(100000)
dict_one_hot_encoder = DictVectorizer(sparse=False)
X_train = dict_one_hot_encoder.fit_transform(X_dict_train)
if load_model == True:
	l_reg_file = open('../models/logistic_regression_model_ol.sav', 'rb')
	sgd_log_reg_model = pickle.load(l_reg_file)
	X_dict_test, y_test_next = process_data(10000, (20 + 1) * 200000)  # n_samples, offset
	X_test_next = dict_one_hot_encoder.transform(X_dict_test)
	predict = sgd_log_reg_model.predict_proba(X_test_next)[:, 1]
	score = roc_auc_score(y_test_next, predict)
	return 0

# Train and partially fit on 1 million samples
for i in range(20):
	X_dict_train, y_train_every = process_data(100000, i * 100000)
	X_train_every = dict_one_hot_encoder.transform(X_dict_train)
	sgd_log_reg_model.partial_fit(X_train_every, y_train_every, classes=[0, 1])

X_dict_test, y_test_next = process_data(10000, (i + 1) * 200000)
X_test_next = dict_one_hot_encoder.transform(X_dict_test)

predict = sgd_log_reg_model.predict_proba(X_test_next)[:, 1]
score = roc_auc_score(y_test_next, predict)
l_reg_file = open('../models/logistic_regression_model_ol.sav', "wb")
pickle.dump(sgd_log_reg_model, l_reg_file)
l_reg_file.close()
```

# 工程问题

## 线上serving



# 评价指标

[Predictive Model Performance: Offline and Online Evaluations](https://chbrown.github.io/kdd-2013-usb/kdd/p1294.pdf)

**如何评价CTR预估效果？**

难度：

- 观察到的是点击或没点击的二元数据，但是要预估的是一个[0, 1]上的点击概率，换句话说就是没有绝对的ground truth；
- 各种机器学习模型训练完以后出来的分数，即使是LR，也不见得就是一个好的可以直接拿来当预估结果的概率；
- 观察数据往往是有偏的，观察到的广告展现和点击数据都是赢得竞价的那些。

常用评价指标是logloss和AUC。logloss更关注和观察数据的吻合速度，AUC更关注rank order，这两个指标适合线下评估。线上有更简单直接的评价方式：把线上的impression log按照预测的CTR从小到大排序，然后按照某个特点流量步长（比如每10000个impression）分桶，统计每个分桶的平均预估CTR（pCTR）和实际CTR（aCTR），把他俩的对比关系画出来，理想状态下应该是一条斜率为1的线。

线上的其他业务指标，比如点击率、营收、利润、eCPC等是不能给出CTR预估效果评价的。这些业务指标，受到整个广告系统其他模块，如bid optimization, budget pacing等其他外部竞价环境的综合影响。

CTR预估，需要解决三个问题：
- rank order；
- calibration；
- sample distribution reconstruction;

## COPC

全称click over predicted click，copc=实际的点击率/模型预测的点击率，主要衡量整体预估的偏高和偏低，越接近1越好，一般情况下在1附近波动。


## AUC

- 最好不要把所有流量合起来用AUC评估，因为无法区分广告点击率在哪些流量上预测得好或不好。
- 当AUC指标不好时，可以通过计算max AUC来验证是否是数据的问题，计算的方式是计算每一种特征组合的点击率，将这个点击率作为预测值计算AUC，计算出来的值是max AUC。
- 加入特征后（特征是user/context特征后），AUC的提升，上线后不一定会提升，因为离线评估计算的AUC是针对query与query之间的，而在线时，排序是针对ad与ad之间的。

## Logloss

对数损失（Log loss）亦被称为逻辑回归损失（Logistic regression loss）或交叉熵损失（Cross-entropy loss）。通常把模型关于单个样本预测值与真实值的差称为损失，损失越小，模型越好，而用于计算损失的函数称为损失函数。主要是评估距，但logloss在pCTR和CTR值比较近的时候（比如差个5%），区别比较小。

- Logloss对把正例预测的值很低，或是将负例的值预测得很高都会比较高的值，而将负例预测撑0.01，或者0.011，则区别不大。


## GAUC


# 其他

## 点击率预估 (CTR)

在计算广告系统中，一个可以携带广告请求的用户流量到达后台时，系统需要在较短时间（一般要求不超过 100ms）内返回一个或多个排序好的广告列表；在广告系统中，一般最后一步的排序 *score = bid * pctr <sup>alpha</sup>*；其中 *alpha* 参数控制排序倾向，如果*alpha*<1，则倾向于*pctr*，否则倾向于*bid*；这里的核心因子*pctr*就是通常所说的点击率（predicted click through rate）.

点击率预估是计算广告中非常重要的模块，预估一个用户对广告的点击概率，从而提升广告效果。

### 特征表示 Feature Representation

高维、稀疏、多field是输入给CTR预估模型的特征数据的典型特点。

#### Embedding表示

Embedding表示也叫做Distributed representation，起源于神经网络语言模型（NNLM）对语料库中的word的一种表示方法。相对于高维稀疏的one-hot编码表示，embedding-based的方法，学习一个低维稠密实数向量（low-dimensional dense embedding）。类似于hash方法，embedding方法把位数较多的稀疏数据压缩到位数较少的空间，不可避免会有冲突；然而，embedding学到的是类似主题的语义表示，对于item的“冲突”是希望发生的，这有点像软聚类，这样才能解决稀疏性的问题。

Google公司开源的word2vec工具让embedding表示方法广为人知。Embedding表示通常用神经网络模型来学习，当然也有其他学习方法，比如矩阵分解（MF）、因子分解机（FM)等。



### LR模型

将用户是否点击一个物品看成回归问题以后，使用最广泛的模型当属逻辑回归 Logistic Regression。LR模型是广义线性模型，从其函数形式来看，LR模型可以看作是一个没有隐层的神经网络模型（感知机模型）。

LR模型一直是CTR预估问题的benchmark模型，由于其简单、易于并行化实现、可解释性强等优点而被广泛使用。然而由于线性模型本身的局限，不能处理特征和目标之间的非线性关系，因此模型效果严重依赖于算法工程师的特征工程经验。为了让线性模型能够学习到原始特征与拟合目标之间的非线性关系，通常需要对原始特征做一些非线性转换。常用的转换方法包括：**连续特征离散化**、**特征之间的交叉**等。

- **连续特征离散化**
    - 连续特征离散化的方法一般是把原始连续值的值域范围划分为多个区间，比如等频划分或等间距划分，更好的划分方法是利用监督学习的方式训练一个简单的单特征的决策树桩模型，即用信息增益指标来决定分裂点。特征离散化相当于把线性函数变成了分段线性函数，从而引入了非线性结构。
- **特征之间的交叉**
    - 通常CTR预估涉及到用户、物品、上下文等几方面的特征，往往单个特征对目标判定的贡献是较弱的，而不同类型的特征组合在一起就能够对目标的判定产生较强的贡献。比如用户性别和商品类目交叉就能够刻画例如“女性用户偏爱美妆类目”，“男性用户喜欢男装类目”的知识。

LR模型的不足在于特征工程耗费了大量的精力，而且即使有经验的工程师也很难穷尽所有的特征交叉组合。




#### 在线优化算法

求解该类问题最经典的算法是GD（梯度下降法），即沿着梯度方法逐渐优化模型参数。梯度下降法能够保证精度，要想预防过拟合问题一般会加上正则项，L1或者L2正则。

相比于批量GD算法，OGD能够利用实时产生的正负样本，一定程度上能够优化模型效果。在线优化算法需要特殊关注模型鲁棒性和稀疏性，由于样本是一个一个到来，模型参数可能因为训练样本不足导致过拟合等。因此OGD会着重处理模型稀疏性。

### LR+GBDT

模型级联提供了一种思路，典型的例子就是Facebook 2014年的论文中介绍的通过GBDT（Gradient Boost Decision Tree）模型解决LR模型的特征组合问题。思路很简单，特征工程分为两部分，一部分特征用于训练一个GBDT模型，把GBDT模型每颗树的叶子节点编号作为新的特征，加入到原始特征集中，再用LR模型训练最终的模型。

![LR+GBDT](https://pic3.zhimg.com/80/v2-37df8f79aa024bbca443a7adb1db6b0e_hd.jpg)

GBDT模型能够学习高阶非线性特征组合，对应树的一条路径（用叶子节点来表示）。通常把一些连续值特征、值空间不大的categorical特征都丢给GBDT模型；空间很大的ID特征（比如商品ID）留在LR模型中训练，既能做高阶特征组合又能利用线性模型易于处理大规模稀疏数据的优势。

### FM(Factorization Machine)因子分解机、FFM(Field-aware Factorizatiion Machine)

因子分解机(Factorization Machines, FM)通过特征对之间的隐变量内积来提取特征组合，其函数形式如下：
![FM公式](https://www.zhihu.com/equation?tex=y%3Dw_0+%2B+%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dw_i+x_i+%2B+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Csum_%7Bj%3Di%2B1%7D%5En+%5Clangle+v_i%2Cv_j+%5Crangle+x_i+x_j)

FM和基于树的模型（e.g. GBDT）都能够自动学习特征交叉组合。基于树的模型适合连续中低度稀疏数据，容易学到高阶组合。但是树模型却不适合学习高度稀疏数据的特征组合，一方面高度稀疏数据的特征维度一般很高，这时基于树的模型学习效率很低，甚至不可行；另一方面树模型也不能学习到训练数据中很少或没有出现的特征组合。相反，FM模型因为通过隐向量的内积来提取特征组合，对于训练数据中很少或没有出现的特征组合也能够学习到。例如，特征 *i* 和特征 *j* 在训练数据中从来没有成对出现过，但特征 *i* 经常和特征 *p* 成对出现，特征* *j* 也经常和特征 *p* 成对出现，因而在FM模型中特征 *i* 和特征 *j* 也会有一定的相关性。

在推荐系统中，常用矩阵分解（MF）的方法把User-Item评分矩阵分解为两个低秩矩阵的乘积，这两个低秩矩阵分别为User和Item的隐向量集合。通过User和Item隐向量的点积来预测用户对未见过的物品的兴趣。矩阵分解也是生成embedding表示的一种方法，示例图如下：
![矩阵分解](https://pic4.zhimg.com/80/v2-652d2727bf98f174bd8e2a502656f677_hd.jpg)

MF方法可以看作是FM模型的一种特例，即MF可以看作特征只有userId和itemId的FM模型。FM的优势是能够将更多的特征融入到这个框架中，并且可以同时使用一阶和二阶特征；而MF只能使用两个实体的二阶特征。
![LR vs. MF](https://pic4.zhimg.com/80/v2-d90f9462b10b2638be6e5872e3c6479f_hd.jpg)

在二分类问题中，采用LogLoss损失函数时，FM模型可以看做是LR模型和MF方法的融合，如下图所示：

![](https://pic3.zhimg.com/80/v2-0eb9f4c95e0e6ae96c97d1fb6a2a5832_hd.jpg)

FFM（Field-aware Factorization Machine）模型是对FM模型的扩展，通过引入field的概念，FFM把相同性质的特征归于同一个field。例如，“Day=26/11/15”、 “Day=1/7/14”、 “Day=19/2/15”这三个特征都是代表日期的，可以放到同一个field中。在FFM中，每一维特征 *x<sub>i</sub>* ，针对其它特征的每一种field *f<sub>j</sub>* ，都会学习一个隐向量 *v<sub>i</sub>f<sub>j</sub>* 。因此，隐向量不仅与特征相关，也与field相关。假设样本的*n*个特征属于*f*个field，那么FFM的二次项有*nf*个隐向量。

![](https://www.zhihu.com/equation?tex=y%3Dw_0+%2B+%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dw_i+x_i+%2B+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Csum_%7Bj%3Di%2B1%7D%5En+%5Clangle+v_%7Bi%2Cf_j%7D%2Cv_%7Bj%2Cf_i%7D+%5Crangle+x_i+x_j)

FM可以看作FFM的特例，在FM模型中，每一维特征的隐向量只有一个，即FM是把所有特征都归属到一个field时的FFM模型。


### 混合逻辑回归（MLR）

MLR算法是alibaba在2012年提出并使用的广告点击率预估模型，2017年发表出来。MLR模型是对线性LR模型的推广，它利用分片线性方式对数据进行拟合。基本思路是采用分而治之的策略：如果分类空间本身是非线性的，则按照合适的方式把空间分为多个区域，每个区域里面可以用线性的方式进行拟合，最后MLR的输出就变为了多个子区域预测值的加权平均。如下图(C)所示，就是使用4个分片的MLR模型学到的结果。

![](https://pic3.zhimg.com/80/v2-008b9a77103e53d6e40c0f8b9047c846_hd.jpg)
![](https://www.zhihu.com/equation?tex=f%28x%29%3D%5Csum_%7Bi%3D1%7D%5Em+%5Cpi_i%28x%2C%5Cmu%29%5Ccdot+%5Ceta_i%28x%2Cw%29%3D%5Csum_%7Bi%3D1%7D%5Em+%5Cfrac%7Be%5E%7B%5Cmu_i%5ET+x%7D%7D%7B%5Csum_%7Bj%3D1%7D%5Em+e%5E%7B%5Cmu_j%5ET+x%7D%7D+%5Ccdot+%5Cfrac%7B1%7D%7B1%2Be%5E%7B-w%5ETx%7D%7D)

上式即为MLR的目标函数，其中 m 为分片数（当 m=1 时，MLR退化为LR模型）； ![](https://www.zhihu.com/equation?tex=%5Cpi_i%28x%2C%5Cmu%29%3D+%5Cfrac%7Be%5E%7B%5Cmu_i%5ET+x%7D%7D%7B%5Csum_%7Bj%3D1%7D%5Em+e%5E%7B%5Cmu_j%5ET+x%7D%7D)是聚类参数，决定分片空间的划分，即某个样本属于某个特定分片的概率； ![](https://www.zhihu.com/equation?tex=%5Ceta_i%28x%2Cw%29+%3D+%5Cfrac%7B1%7D%7B1%2Be%5E%7B-w%5ETx%7D%7D)是分类参数，决定分片空间内的预测； ![](https://www.zhihu.com/equation?tex=%5Cmu)和 ![](https://www.zhihu.com/equation?tex=w) 都是待学习的参数。最终模型的预测值为所有分片对应的子模型的预测值的期望。

MLR模型在大规模稀疏数据上探索和实现了非线性拟合能力，在分片数足够多时，有较强的非线性能力；同时模型复杂度可控，有较好泛化能力；同时保留了LR模型的自动特征选择能力。

MLR模型的思路非常简单，难点和挑战在于MLR模型的目标函数是非凸非光滑的，使得传统的梯度下降算法并不适用。相关的细节内容查询论文：Gai et al, “Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction”。

另一方面，MLR模型可以看作带有一个隐层的神经网络。如下图， *x*是大规模的稀疏输入数据，MLR模型第一步是做了一个Embedding操作，分为两个部分，一种叫聚类Embedding（绿色），另一种是分类Embedding（红色）。两个投影都投到低维的空间，维度为*m* ，是MLR模型中的分片数。完成投影之后，通过很简单的内积（Inner Product）操作便可以进行预测，得到输出 *y* 。

![](https://pic1.zhimg.com/80/v2-610b2cdc8238c45c2cbfb5ffbb150f84_hd.jpg)


### Wide & Deep Learning (WDL)

像LR这样的wide模型学习特征与目标之间的直接相关关系，偏重记忆（memorization），如在推荐系统中，wide模型产生的推荐是与用户历史行为的物品直接相关的物品。这样的模型缺乏刻画特征之间的关系的能力，比如模型无法感知到“土豆”和“马铃薯”是相同的实体，在训练样本中没有出现的特征组合自然就无法使用，因此可能模型学习到某种类型的用户喜欢“土豆”，但却会判定该类型的用户不喜欢“马铃薯”。

WDL是Google在2016年的paper中提出的模型，其巧妙地将传统的特征工程与深度模型进行了强强联合。模型结构如下:
![](https://pic2.zhimg.com/80/v2-a317a9fb4bbc943a5bc894924dc997f5_hd.jpg)

WDL分为wide和deep两部分联合训练，单看wide部分与LR模型并没有什么区别；deep部分则是先对不同的ID类型特征做embedding，在embedding层接一个全连接的MLP（多层感知机），用于学习特征之间的高阶交叉组合关系。由于Embedding机制的引入，WDL相对于单纯的wide模型有更强的泛化能力。

### FNN (Factorization-machine supported Neural Network)

除了神经网络模型，FM模型也可以用来学习到特征的隐向量（embedding表示），因此一个自然的想法就是先用FM模型学习到特征的embedding表示，再用学到的embedding向量代替原始特征作为最终模型的特征。这个思路类似于LR+GBDT，整个学习过程分为两个阶段：第一个阶段先用一个模型做特征工程；第二个阶段用第一个阶段学习到新特征训练最终的模型。

FNN模型就是用FM模型学习到的embedding向量初始化MLP，再由MLP完成最终学习，其模型结构如下：
![](https://pic3.zhimg.com/80/v2-8c9c653b6cb47471a19a1265a3a709ea_hd.jpg)

### PNN (Product-based Neural Networks)

MLP中的节点add操作可能不能有效探索到不同类别数据之间的交互关系，虽然MLP理论上可以以任意精度逼近任意函数，但越泛化的表达，拟合到具体数据的特定模式越不容易。PNN主要是在深度学习网络中增加了一个inner/outer product layer，用来建模特征之间的关系。
![](https://pic1.zhimg.com/80/v2-4d8262eddb0e37f8efcf031b48c79e80_hd.jpg)

Embedding Layer和Product Layer之间的权重为常量1，在学习过程中不更新。Product Layer的节点分为两部分，一部分是*z*向量，另一部分是*p*向量。 *z*向量的维数与输入层的Field个数（ *N* ）相同， ![](https://www.zhihu.com/equation?tex=z%3D%28f_1%2Cf_2%2Cf_3%2C%5Ccdots%2Cf_N%29)。 *p*向量的每个元素的值由embedding层的feature向量两两成对并经过Product操作之后生成，![](https://www.zhihu.com/equation?tex=p%3D%5C%7Bg%28f_i%2Cf_j%29%5C%7D%2Ci%3D1+%5Ccdots+N%2C+j%3Di+%5Ccdots+N) ，因此*p*向量的维度为![](https://www.zhihu.com/equation?tex=N%2A%28N-1%29)。这里的*f<sub>i</sub>*是field *i* 的embedding向量，![](https://www.zhihu.com/equation?tex=f_i%3DW_0%5Ei+x%5Bstart_i+%3A+end_i%5D) ，其中*x*是输入向量，![](https://www.zhihu.com/equation?tex=x%5Bstart_i+%3A+end_i%5D)是field *i* 的one-hot编码向量。

这里所说的Product操作有两种：内积和外积；对应的网络结构分别为IPNN和OPNN，两者的区别如下图。
![](https://pic1.zhimg.com/80/v2-cff417b2bd608984001b585390c4af84_hd.jpg)

在IPNN中，由于Product Layer的 p 向量由field两两配对产生，因此维度膨胀很大，给 l_1 Layer的节点计算带来了很大的压力。受FM启发，可以把这个大矩阵转换分解为小矩阵和它的转置相乘，表征到低维度连续向量空间，来减少模型复杂度： 
![](https://www.zhihu.com/equation?tex=W_p%5En+%5Codot+p+%3D+%5Csum_%7Bi%3D1%7D%5EN+%5Csum_%7Bj%3D1%7D%5EN+%5Ctheta_i%5En+%5Ctheta_j%5En+%5Clangle+f_i%2Cf_j+%5Crangle+%3D+%5Clangle+%5Csum_%7Bi%3D1%7D%5EN+%5Cdelta_i%5En%2C+%5Csum_%7Bi%3D1%7D%5EN+%5Cdelta_i%5En+%5Crangle+)

在OPNN中，外积操作带来更多的网络参数，为减少计算量，使得模型更易于学习，采用了多个外积矩阵按元素叠加（element-wise superposition）的技巧来减少复杂度，具体如下： 
![](https://www.zhihu.com/equation?tex=p%3D%5Csum_%7Bi%3D1%7D%5EN+%5Csum_%7Bj%3D1%7D%5EN+f_i+f_j%5ET%3Df_%7B%5CSigma%7D%28f_%7B%5CSigma%7D%29%5ET%2C+f_%7B%5CSigma%7D%3D%5Csum_%7Bj%3D1%7D%5EN+f_i)

### DeepFM

深度神经网络对于学习复杂的特征关系非常有潜力。目前也有很多基于CNN与RNN的用于CTR预估的模型。但是基于CNN的模型比较偏向于相邻的特征组合关系提取，基于RNN的模型更适合有序列依赖的点击数据。

FNN模型首先预训练FM，再将训练好的FM应用到DNN中。PNN网络的embedding层与全连接层之间加了一层Product Layer来完成特征组合。PNN和FNN与其他已有的深度学习模型类似，都很难有效地提取出低阶特征组合。WDL模型混合了宽度模型与深度模型，但是宽度模型的输入依旧依赖于特征工程。上述模型要不然偏向于低阶特征或者高阶特征的提取，要不然依赖于特征工程。而DeepFM模型可以以端对端的方式来学习不同阶的组合特征关系，并且不需要其他特征工程。

DeepFM的结构中包含了因子分解机部分以及深度神经网络部分，分别负责低阶特征的提取和高阶特征的提取。其结构如下：
![](https://pic3.zhimg.com/80/v2-fdfa81ae81648fc44eda422bfbcee572_hd.jpg)

上图中红色箭头所表示的链接权重恒定为1（weight-1 connection），在训练过程中不更新，可以认为是把节点的值直接拷贝到后一层，再参与后一层节点的运算操作。

与Wide&Deep Model不同，DeepFM共享相同的输入与embedding向量。在Wide&Deep Model中，因为在Wide部分包含了人工设计的成对特征组，所以输入向量的长度也会显著增加，这也增加了复杂性。

DeepFM包含两部分：神经网络部分与因子分解机部分。这两部分共享同样的输入。对于给定特征*i*，向量*w<sub>i</sub>*用于表征一阶特征的重要性，隐变量*V<sub>i</sub>*用于表示该特征与其他特征的相互影响。在FM部分，*V<sub>i</sub>*用于表征二阶特征，同时在神经网络部分用于构建高阶特征。所有的参数共同参与训练。DeepFM的预测结果可以写为![](https://www.zhihu.com/equation?tex=%5Chat%7By%7D%3Dsigmoid%28y_%7BFM%7D%2By_%7BDNN%7D%29)
其中![](https://www.zhihu.com/equation?tex=%5Chat%7By%7D%E2%88%88%280%2C1%29)是预测的点击率，![](https://www.zhihu.com/equation?tex=y_%7BFM%7D)与![](https://www.zhihu.com/equation?tex=y_%7BDNN%7D)分别是FM部分与DNN部分。

FM部分的详细结构如下：
![](https://pic1.zhimg.com/80/v2-847aac002cf8f55e718ffea47deea9c8_hd.jpg)

FM的输出如下公式：
![](https://www.zhihu.com/equation?tex=y_%7BFM%7D%3D%5Clangle+w%2Cx+%5Crangle+%2B+%5Csum_%7Bi%3D1%7D%5Ed+%5Csum_%7Bj%3Di%2B1%7D%5Ed+%5Clangle+V_i%2CV_j+%5Crangle+x_i+x_j)
其中![](https://www.zhihu.com/equation?tex=w%E2%88%88R%5Ed%2CV_i%E2%88%88R%5Ek)。加法部分反映了一阶特征的重要性，而内积部分反应了二阶特征的影响。

深度部分详细如下：
![](https://pic1.zhimg.com/80/v2-6462ab78b751e3d7e99e998d9e320ba0_hd.jpg)

深度部分是一个前馈神经网络。与图像或者语音这类输入不同，图像语音的输入一般是连续而且密集的，然而用于CTR的输入一般是及其稀疏的。因此需要设计特定的网络结构，具体实现为，在第一层隐含层之前，引入一个嵌入层来完成将输入向量压缩到低维稠密向量。
![](https://www.zhihu.com/equation?tex=y_%7BDNN%7D%3D%5Csigma%28W%5E%7BH%2B1%7D+%5Ccdot+a%5EH+%2B+b%5E%7BH%2B1%7D%29)

其中*H*是隐层的层数。

### FTRL

FTRL 是从 RDA、FOBOS 等针对 LR 的在线学习算法改进而来，主要是工业界强烈的在线学习的需求驱动而来。

在线学习背后的理念是每个人的兴趣是 non-stationary 的，离线训练的模型在线上可能不能快速对用户最新的行为作出反应。为了解决这个问题，一种做法是我们加快模型的频率，比如原来一天更新一次，现在一个小时更新一次，这种做法有很明显的瓶颈，比如如果我们的时间窗设置的比较长，用一个月或者两个月数据来跑模型，则可能导致模型在更新间隙内完不成训练；如果我们采用增量训练的方式，则增量时间窗的设置是个技术活，太短，很多曝光对应的点击还没有上来，导致训练数据的无效曝光比例偏高，如果太长，可能跟不上节奏；这也是在线学习的一个难点，在线学习一般也不会每一个回流数据立马更新模型，这会导致模型震荡频繁，攒一小段时间是个不错的选择，为此 Facebook 的系统里有一个 online joiner 的组件来做曝光和点击的归约。

从今日头条披露的资料来看，在模型更新方面他们采用了增量更新 + 定时校准的策略；类似于在线学习 + 定时离线校准。这种策略应该也可以用到点击率的场景。

在线学习另外一个要重点解决的问题是学习率；离线训练的时候 sgd 往往使用一个公用的学习率η，但是在线学习这样做会带来问题；因为样本分布不均衡，某些覆盖不是很高的特征对应的权重因为样本少得到的更新次数比较少，如果使用相同的学习率，则这些权重的收敛势必落后于覆盖率高的样本的特征对应的权重，尤其是有做学习率衰减的情况下；因此我们需要针对不同的权重来设置不同的学习率，做法也比较简单，基本思路是统计该维度样本数，多的衰减快点，少的衰减慢点以期能做到基本持平。

FTRL 主要是针对 LR 部分的 online learning；GBDT+LR 是两种不同模型的级联，这两个方案是可以很方便的糅合在一起的变成 GBDT+FTRL-LR；但这里 GBDT 的更新没法做到 online learning；可以做定期更新。理论上这种做法可能会效果更好一点。

### DIN

DIN是阿里17年的论文中提出的深度学习模型，该模型基于对用户历史行为数据的两个观察：1、多样性，一个用户可能对多种品类的东西感兴趣；2、部分对应，只有一部分的历史数据对目前的点击预测有帮助，比如系统向用户推荐泳镜时会与用户点击过的泳衣产生关联，但是跟用户买的书就关系不大。于是，DIN设计了一个attention结构，对用户的历史数据和待估算的广告之间部分匹配，从而得到一个权重值，用来进行embedding间的加权求和。
![](https://pic2.zhimg.com/80/v2-c64cf8730c836d2a709157ce16cc8b7d_hd.jpg)

DIN模型的输入分为2个部分：用户特征和广告(商品)特征。用户特征由用户历史行为的不同实体ID序列组成。在对用户的表示计算上引入了attention network (也即图中的Activation Unit) 。DIN把用户特征、用户历史行为特征进行embedding操作，视为对用户兴趣的表示，之后通过attention network，对每个兴趣表示赋予不同的权值。这个权值是由用户的兴趣和待估算的广告进行匹配计算得到的，如此模型结构符合了之前的两个观察：用户兴趣的多峰分布以及部分对应。Attention network 的计算公式如下，
![](https://www.zhihu.com/equation?tex=V_u%3Df%28V_a%29%3D%5Csum_%7Bi%3D1%7D%5EN+w_i+%5Ccdot+V_i+%3D%5Csum_%7Bi%3D1%7D%5EN+g%28V_i%2CV_a%29+%5Ccdot+V_i)

其中，![](https://www.zhihu.com/equation?tex=V_u)代表用户表示向量， ![](https://www.zhihu.com/equation?tex=V_i)是用户行为![](https://www.zhihu.com/equation?tex=i)的embedding向量，![](https://www.zhihu.com/equation?tex=V_a)代表广告的表示向量。核心在于用户的表示向量不仅仅取决于用户的历史行为，而且还与待评估的广告有直接的关联。

### 评价指标

#### AUC

AUC 是 ROC 曲线下的面积，是一个 [0,1] 之间的值。他的优点是用一个值概括出模型的整体 performance，不依赖于阈值的选取。因此 AUC 使用很广泛，既可以用来衡量不同模型，也可以用来调参。

AUC 指标的不足之处有两点：一是只反映了模型的整体性能，看不出在不同点击率区间上的误差情况；二是只反映了排序能力，没有反映预测精度。 简单说，如果对一个模型的点击率统一乘以 2，AUC 不会变化，但显然模型预测的值和真实值之间的 offset 扩大了。

### RMSE

Netflix 比赛用的 RMSE 指标可以衡量预测的精度，与之类似的指标有 MSE、MAE。

#### RIG

这里要强调的一点是 RIG 指标不仅和模型的质量有关，还和数据集的分布情况有关；因此千万注意不可以使用 RIG 来对比不同数据集上生成的模型，但可以用来对比相同数据集上不同模型的质量差异。这一点尤为重要。


### 总结

主流的CTR预估模型已经从传统的宽度模型向深度模型转变，与之相应的人工特征工程的工作量也逐渐减少。上文提到的深度学习模型，除了DIN对输入数据的处理比较特殊之外，其他几个模型还是比较类似的，它们之间的区别主要在于网络结构的不同，如下图所示: 
![](https://pic2.zhimg.com/80/v2-87e333bad5a7e9c5d5cc738adcc617a5_hd.jpg)

### 新广告：lookalike、相关广告信息挖掘

新广告的点击率预测是另一个比较大的话题，这里我先简单分为两类，一类是临时性投放，比如某个新广告主偶然来试投一下汽车广告。如果广告主能提供一批种子用户，我们可以使用 lookalike 的方法来优化之，可以参考 fandy 的 [15]，我的理解是一个迭代处理，先基于种子用户和采样用户训练一个 model，然后用 model 对采样的用户做一轮 predict，把得分高的用户刨除掉，剩下的用户定义为有效负用户，然后再和种子用户一起训练一个新的 model，将这个 model 作为候选 predict 并圈取用户的指示器。

另一类新广告是广告主或者代理商在广告投放系统里投放的一个新的素材，这个时候虽然广告是全新的，但是我们任然可以从系统里挖掘出大量相关的信息，比如该广告对应的 pushlisher 的历史信息，对应的 advertiser 的信息，对应类比的信息等，具体可以参考 [14]。

### Rare Event：贝叶斯平滑、指数平滑

想法的初衷是我们经常需要使用一些点击率特征，比如曝光两次点击一次我们可以得出 0.5 点击率，另一个广告是曝光一万次，点击五千次，也可以得到 0.5 的点击率，但是这两个 0.5 代表的意义能一样吗？前者随着曝光的增加，有可能会快速下滑，为了解决经验频率和概率之间的这种差异，我们引入平滑的技巧。
