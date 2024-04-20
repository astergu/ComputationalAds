## 计费

广告是一种商业，那就有它的定价模式，有了定价模式才能形成产业链，通过定价模式在上下游间结算。

![定价方式](https://pic3.zhimg.com/v2-4eb0c156d3f86236be88d6429c13c802_r.jpg)

常用的定价方式，包括展示类和转化类，展示类常用于品牌广告，转化类常用于效果广告，具体介绍如下。

CPT：Cost Per Time，按时长计费，即按照占据此广告的时长计费，在高价值的广告位上常见，例如开屏广告、跳一跳的广告等

CPM：Cost Per Mille，按展示量计费，即按照此广告的展示次数计费，以品牌展示类为主

CPC：Cost Per Click，按点击量计费，即按照此广告的点击次数计费，关键词竞价常用，例如头条的信息流广告


CPA：Cost Per Action，按行动量计费，即按照某些用户行为次数计费，CPA包括以下CPD、CPI、CPS等

CPD：Cost Per Download，按下载量计费，即按用户完成APP下载计费，APP、游戏等常用

CPI：Cost Per Install，按安装量计费，即按用户激活APP计费，这种比较少，一般是广告主内部衡量效果的指标

CPS：Cost Per Sales，按销售量计费，即按完成订单的用户数量结算，以电商类为主

### 广义第二价格（Generalized Second Price, GSP）

谷歌在2002年，将广义第二价格的方式引入搜索引擎，基本原理就是按照下一位的出价，来实际扣费，为了鼓励广告主提高素材，广告点击率。实际计费的公式变成了

```
收费=下一位价格*（下一位质量分/本位质量分）+ 0.01
```

整个系统相对比较稳定，容易形成局部最优和整体稳定。

### VCG

VCG是一种比第二价格还要晦涩的一种方法，它的基本原理是计算竞价者赢得广告位后，给整个竞价收入带来的收益损失，理论上这种损失就是竞价获胜者应该支付的费用。

### 实时竞价 RTB (Real-time Bidding)

RTB的运作方式并不复杂：当一个用户打开某个网页，这个网页中的广告位信息通过SSP（Supply Side Platform）供应方平台提供给广告交易平台（Ad Exchange），同时，这个用户所用的浏览器获得的Cookies的标签进入DMP（Data Management Platform）管理平台进行分析，将分析所得到的用户属性的标签也传送给Ad Exchange；接下来，Ad Exchange 将这些信息向所有接入到交易平台的广告主或者广告代理商的DSP（Demand Side Platform）需求方平台发出指令，DSP开始向Ad Exchange实时出价，进入RTB模式；经过竞价，用户的属性标签一致，且出价最高的DSP就获得了这次展示广告的机会，广告自动返回到用户的浏览器所打开的这个网页中——这一系列的过程非常快，通常是在80－100毫秒中完成的。