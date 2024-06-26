- [计费](#计费)
  - [计费模式](#计费模式)
  - [计费方式](#计费方式)
    - [广义第一价格（Generalized First Price，GFP）](#广义第一价格generalized-first-pricegfp)
    - [广义第二价格（Generalized Second Price，GSP）](#广义第二价格generalized-second-pricegsp)
    - [Vickrey-Clarke-Groves (VCG)竞价机制](#vickrey-clarke-groves-vcg竞价机制)
    - [实时竞价 RTB (Real-time Bidding)](#实时竞价-rtb-real-time-bidding)

# 计费

广告平台所谓的宏观调控能力，主要是计费模式和计费方式。计费模式包括CPM(千次展现计费），CPC(点击计费），CPS/A(转化/成交计费）等。在广告位资源紧张的时候，广告平台通常会使用竞价的方式，广告主可以根据自己需求和能力出价。

对于互联网广告，广告资源形式多样，位置多样，每个广告主的竞价策略是不同的，互联网的广告通常是以暗拍的方式进行，即拍卖不公布竞价的广告主和它们的出价，由广告系统根据统一算法决定广告的展现。

## 计费模式

![定价方式](https://pic3.zhimg.com/v2-4eb0c156d3f86236be88d6429c13c802_r.jpg)

常用的定价方式，包括展示类和转化类，展示类常用于品牌广告，转化类常用于效果广告，具体介绍如下：

- CPT：Cost Per Time，按时长计费，即按照占据此广告的时长计费，在高价值的广告位上常见，例如开屏广告、跳一跳的广告等
- CPM：Cost Per Mille，按展示量计费，即按照此广告的展示次数计费，以品牌展示类为主
- CPC：Cost Per Click，按点击量计费，即按照此广告的点击次数计费，关键词竞价常用，例如头条的信息流广告
- CPA：Cost Per Action，按行动量计费，即按照某些用户行为次数计费，CPA包括以下CPD、CPI、CPS等
- CPD：Cost Per Download，按下载量计费，即按用户完成APP下载计费，APP、游戏等常用
- CPI：Cost Per Install，按安装量计费，即按用户激活APP计费，这种比较少，一般是广告主内部衡量效果的指标
- CPS：Cost Per Sales，按销售量计费，即按完成订单的用户数量结算，以电商类为主

## 计费方式

### 广义第一价格（Generalized First Price，GFP）

广义第一价格就是按照出价去计费，价格高者排在前面，它的优势就是简单，收入可保证，但是稳定性较差。各个广告主为了获得最佳收益，可以通过频繁修改投放价格而获得。举例来说，一个广告主为了获得展现，它会不断的的增加价格，在获得展现后，它又会开始不断的减少价格而降低成本，这种竞争是相对武断的，而且很容易知晓竞争对手的出价。另外，当出价最高广告主停止投放后，容易对广告平台收入产生较大的波动。在2002年之前，所有的搜索引擎都是第一出价法则。

### 广义第二价格（Generalized Second Price，GSP）

谷歌在2002年，将广义第二价格的方式引入搜索引擎，基本原理就是按照下一位的出价，来实际扣费，为了鼓励广告主提高素材，广告点击率。实际计费的公式变成了

```
收费=下一位价格*（下一位质量分/本位质量分）+ 0.01
```

整个系统相对比较稳定，容易形成局部最优和整体稳定。

### Vickrey-Clarke-Groves (VCG)竞价机制

VCG是一种比第二价格还要晦涩的一种方法，它的基本原理是计算竞价者赢得广告位后，给整个竞价收入带来的收益损失，理论上这种损失就是竞价获胜者应该支付的费用。

### 实时竞价 RTB (Real-time Bidding)

RTB的运作方式并不复杂：当一个用户打开某个网页，这个网页中的广告位信息通过SSP（Supply Side Platform）供应方平台提供给广告交易平台（Ad Exchange），同时，这个用户所用的浏览器获得的Cookies的标签进入DMP（Data Management Platform）管理平台进行分析，将分析所得到的用户属性的标签也传送给Ad Exchange；接下来，Ad Exchange 将这些信息向所有接入到交易平台的广告主或者广告代理商的DSP（Demand Side Platform）需求方平台发出指令，DSP开始向Ad Exchange实时出价，进入RTB模式；经过竞价，用户的属性标签一致，且出价最高的DSP就获得了这次展示广告的机会，广告自动返回到用户的浏览器所打开的这个网页中——这一系列的过程非常快，通常是在80－100毫秒中完成的。