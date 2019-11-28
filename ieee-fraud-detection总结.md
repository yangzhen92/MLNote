# [ieee-fraud-detection](https://www.kaggle.com/c/ieee-fraud-detection/data )

2019.07.28-2019.08.27。2617/6381，41.13%。公0.945677，私0.915478。

# my solution

* 作为第一个比赛，先尝试用各种原始方法进行特征筛选，用sklearn的lr、rf人工筛选特征，观察roc-auc、pr-auc指标并且绘图
* 使用了imblearn中的几个下采样方法，并学习了算法原理
* 借用论坛里面的EDA以及分享的帖子，观察数据趋势
* 数据为全年交易数据，起始时间为12月，分两个半年，中间间隔1个月

## features

* transactionAmt x card1\card4
* c系列作为类别特征
* d15 x  card1\4\addr1：目的是晒出唯一的uid，因为一张卡有多笔交易
* 邮箱不处理
* id_30操作系统粗粒度归类，移除版本
* id_31浏览器粗粒度归类，检查版本是否最新（实际上没什么用，从top1的方案看，主要是通过d、card、trans找出唯一的uid）
* device粗粒度编码
* card系列计数编码
* card1 x email\device\id19\20
* P x R email
* id19x20\id02x20\id02xd8\d11xdevice
* card1\2\3\5两两组合
* transactionAmt x hourofweek
* ==train、test在做组合特征时，要相互求交集，然后再编码==
* ==用rf找出feature drift（有一点没做到的是，时间序列的drift可以通过一些组合特征消除，而不是直接抛弃）==
* 清理card1中交易<=2的

## 细节

* 用timesplitcv、kfold都有，主要看是否有variant shift的情况，如果做的特征消除了shift，那么kfold好
* 三个boost模型，提交时用平均的效果最好==（ensemble应该是最后用的，首要任务是做好单模）==
* lgb不指定cat反而更好（从源码上看，cat会舍弃1%的小众类，然后默认最多分32bin，粒度太粗）

# top1 solution

## target

从主办方标记fraud的方式来看，赛题实际目的是判断异常客户，而非异常交易==（这是我与top1的本质差异，真正理解赛题的目标非常重要）==。在EDA中，可由一些特征确认一个客户是有多笔交易的。

## data split

确认客户后，可知train中有16.4%客户在test中，有68.2%仅在test中，其余不确定。

要预测大量未出现在train中的客户和交易。

## key point

找出客户并标记UID，在此基础上做agg，但是不用uid作为训练特征。

在top1的分析中，同时fraud=1\0的仅占0.2%。

==强特==：card1、addr1、d1确定UID。

## features

* value normalization with period

```python
df.groupby([period]).agg('min\max\mean\std')[cols,'min\max\mean\std'].to_dict()
df['tmp_xxx'] = df[col].map()
```

* time block frequency encoding: month\week\day
* december十二月单独做一个dummy code，因为有圣诞节的特殊性（类似我国春节）
* 节假日做dummy_code，没有中国节日的包

```python
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
us_holidays = calendar().holiday(start='2017-10-1', end='2019-10-02')
```

* transactionAmt.clip(0, 5000), d4\6\11\12\14\16.clip(0)。我的方案是abs
* ==最近月份的最大值用于clip，而非全局max==
* ==数值、文本提取，用于拼接==

```python
lambda x: ''.join([i for i in x if x.isnumeric()\isalpha()])
```

* ==catboost od_wait 20->500，修改overfit detector的检测迭代次数，应该是担心欠拟合==
* ==数据并行处理psutil.multiprocess.Pool==
* full addr: addr1 + addr2
* d1\2\3\5\10\11\15与dayofyear做差==（d系列属于time delta，随时间推移而增长，需要一个基准做normalize消除time dependency。不做差，会随时间增长，做差之后，若数值相同，表明时间间隔相同，可能来自同一个uid）==
* transactionAmt用np.round(x, 2)，应为显示中交易最小额度为0.01
* v313表示第一次交易的金额，用np.round
* v系列round之后再与transactionAmt_round相加
* 结合card1、uid找出仅有单笔trans的transactionID之后删除
* **lgb对nan赋值，如-999，以避免nan在分列时不被处理**（理论上赋值也一样全分到一侧）
* 对类别特征用frequency encoding，对0.1%的数据用于nan不同的值替换（我的kernel直接换为nan）
* card系列实际由card1加工而得。card1->2->3...。基本上card1、addr1已经提供了全部信息
* dist1用nunique与uid做agg，多个地址有fraud的嫌疑。还可用于monthofyear、cents等
* ==不用V、id系列也有近0.96==



## ==features importance check==

一开始直接梭哈，transactionAmt的fi score最高。top1用random noise替换后，发现score差不多，甚至更高。

参考文章：

* [Beware Default Random Forest Importances](https://explained.ai/rf-importance/ );
  1. gini gain会在各特征scale、类别基数有差异时不可靠，但是比permutation test要节省很多计算量。
  2. 共线性会严重影响fi score，虽然不影响acc。共线性的特征share的fi score差不多，即总分加起来差不多，消除一些特征后总分差不多。
  3. 用spearman's rank correlation coefficient判断两变量之间的单调性，pearson是判断linear的，各有所长。
* "SmoothGrad: removing noise by adding noise"

## ==降维==

V系列有300+特征，存在大量的共线性问题，所以要降维，可大幅提高训练速度。

top1人工整合为sub-group，用PCA也可以，但是无法保留原始特征。

1. 统计nan，找出nan数量相同的特征组
2. pair-plot查看分布，配合corr系数热力图，相关系数大的划为一组（自行设定阈值，看降维程度）
3. 选nunique最大的作为sub-group的代表

## ==pattern shift==

对于时间序列，目的是筛选时间一致性特征（time consistency）：

用1个或1小组的第1个月的特征训练，predict最后1个月的target。若train、validation的roc-auc差异大，则有pattern shift（也有variant shift、drift等别称）。

我的kernel直接train、test进行筛选，粒度相对top1大了。

## early stop

用后几月数据预测前面1月的target，相当于正则化？

## magic feature

uid = card1_addr1_(day - d1)

在此基础上做agg。

test中可以发现train的uid，直接用train的label，提升0.0016。

## ==确定特征时numeric还是categorical==

两种loop，看roc-auc：

1. 依次设定某特征为numeric，其余为categorical
2. 依次设定某特征为categorical，其余为numeric





