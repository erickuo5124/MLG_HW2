# HW2: Link Prediction

contributed by < `erickuo5124` >
###### tags: `MLG`

姓名：郭又宗
學號：F74072269
系級：資訊系111級

### 作業說明
給定三個社群網路資料，預測 A 是否會與 B 建立新的邊：
- train.csv：邊的資料，由 0（沒有邊）跟 1（有邊）組成，沒有方向性

| id | to | from | label |
| -------- | -------- | -------- |-|
| E10311 | 2399 |2339|0|
| E10255|2397|1144|1|

- content.csv：每個點的 binary attributes

| node | attr1 | attr2 | ... |
| -------- | -------- | -------- |-|
| 351 | 0 |0|...
| 1357|0|0|...

- test.csv：要預測的兩個點

| id | to | from | 
| -------- | -------- | -------- |-|
| E10559 | 2323 |2673|
| E4849 |81|1634|

交出的檔案則是根據 test.csv 邊的 id 按照順序輸出邊與機率值的 csv 檔。ex:

| id | prob | 
| -------- | -------- |
| E10559| 0.2064756304 |
|E4849|0.8682585359|

### 評斷標準：
- mAUC：[sklearn.metrics.roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
- mAP：[sklearn.metrics.average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)

==Final = (mAUC+mAP)/2==

## :feet: Random Walk with Restart
[Google Colab](https://colab.research.google.com/drive/1havoUSb0dqIOPR4--zWkYe_vIHDQIKrZ?usp=sharing)

利用 Random walk with Restart 搭配 Enhanced Graph，計算出的值即為邊建立的機率值。

### Hyper Parameter

| $\lambda$ (trade-off between attribute and graph) | $\alpha$ (Restarting Probability) | STEP |
| -------- | -------- | - |
| 0.6 | 0.2 | 10 |


### Enhanced Graph

![](https://i.imgur.com/Q3RVp9R.jpg)

#### 原理

將每個 node feture 以節點的形式放進 graph 中，如果某個節點 $p$ 有該 feature $a$，即 $p$ 與 $a$ 之間有邊，按照這個邏輯下去建完整張圖。

而 Enhanced Graph 的每個邊會有權重，且進出點的計算方式不同，計算方法分別是：

- 從 attribute $a$ 到 person $p$ ($N_p(a)$ 為 $a$ 的鄰居數)
$$
w(a, p) = \frac{1}{|N_p(a)|}
$$

- 從 person $p$ 到  attribute $a$

$$
w(p, a) = 
\begin{cases} 
\frac{\lambda}{|N_a(p)|} & \text{if $|N_a(p)|>0$ and $|N_p(p)|>0$} 
\\ \frac{1}{|N_a(p)|} & \text{if $|N_a(p)|>0$ and $|N_p(p)|=0$}
\\ 0 & \text{otherwise}
\end{cases}
$$

- 從 person $p$ 到  person $p'$

$$
w(p, p') = 
\begin{cases} 
\frac{1-\lambda}{|N_p(p)|} & \text{if $|N_p(p)|>0$ and $|N_a(p)|>0$} 
\\ \frac{1}{|N_a(p)|} & \text{if $|N_p(p)|>0$ and $|N_a(p)|=0$}
\\ 0 & \text{otherwise}
\end{cases}
$$

![](https://i.imgur.com/cdTHgWa.jpg)

#### 實作

我是將原本的 graph 跟 attribute graph 分開計算 degree，如此一來便不會有 **person 跟 attribute edge 混在一起**的問題，算完邊的權重之後將兩個圖合在一起，就完成了 Enhanced Graph。

:::info
每個有邊的 node 所有出去的邊總和應該為 1 或接近 1；若該節點有連接到任一 attribute node，連接到 attribute node 邊的值總和應該接近或等於 $\lambda$ (trade-off between attribute and graph)，沒有的話應該為 0；若該節點有連接到任一 person node，連接到 person node 邊的值總和應該接近或等於 $1-\lambda$，沒有的話應該為 0
:::

### Random Walk with Restart

![](https://i.imgur.com/IOeVxcg.png)

#### 原理

從某個點 $v$ 開始走，每一回合走一步得到每個點的機率值，在經過好幾輪之後每個點的值會收斂到一定範圍內，即為預測的機率。

在決定走哪條路線時，按照剛剛所算出 Enhanced Graph 的值來當作隨機的機率，且每一回合有 $\alpha$ (Restarting Probability) 的機率回到起點，公式大概如下：

$$
ans = 
\left[
    \begin{matrix}
        0 &  0.2 & 0.4 & 0.4 \\
        0.1 & 0 & 0 & 0 \\
        0.9 &  0 & 0 & 0.6 \\
        0 &  0.8 & 0.6 & 0
    \end{matrix}
\right] \times ans + \alpha \times 
\left[
    \begin{matrix}
        0 \\
        1 \\
        0 \\
        0
    \end{matrix}
\right]
$$

其中 $ans$ 的值即為從節點 $v$ 到各點的機率值，右方矩陣則是讓每回合按照 $\alpha$ 的機率回到起始點。

#### 實作

因為我是使用 networkx 的套件來形成 graph，並不是使用矩陣，因此我的算法會稍微再複雜一點點。

```
ans <- 原點=1

for STEP
    for ans 中的點
        for 點的邊
            if 沒有經過
                加進來
            else 
                加上原本的值
    加上 restart probobility
```

STEP 為步數，我試過 5, 10, 15, 20，大概 10 步左右每次的變化就會在 0.01 以下甚至更小，因此我將它設為 10，同時也印證了之前上課 6 步的理論。

:::info
算出來 ans 矩陣中的值**加起來會等於 1**，這樣其實會導致所有預測機率都偏小，最大的反而會是原點，但這不是我們要的，因此我在最後 return 之前做了一次正規化：

```
maxx <- max(ans)
return ans/maxx
```

以機率的最大值為基準，也就是最推薦的點，把它設為 1，其他就根據這個來放大。

:::

### online judge accuracy

- dataset1：88%
- dataset2：92%
- dataset3：93%
:::danger
儘管這個結果並不差，但每個 dataset 運算時間非常長，比較大的 dataset1 需要接近 **2 小時**，較小的 dataset3 也要算 **1 小時**左右。我猜測問題應該出在：

1. 每算一個邊就要對一個點做 BFS，幾乎是算出所有點的機率值，但題目卻只需要其中一個點而已，浪費許多資源在算所有的點上，卻不能因此而不算其他的。
2. 再加上我不是用矩陣運算，而是用迴圈讓他跑，導致沒辦法完全發揮硬體的運算能力。
:::

### Next step...

- GPU：善用平行運算之硬體加速
- Edge Re‐Weighting：依據 attribute $a$ 的熱門程度改變邊的權重
- Global & Local Weighting

## :spider_web: GNN
[Google Colab](https://colab.research.google.com/drive/1p7e1aZdnf1VI2VnM3jsA9-0KLUQAoJ3R?usp=sharing)

利用自己建的 Graph Neural Networks 做訓練，透過 loss function 驗證並做 backpropagation 來提高正確性。

### Hyper parameter

| LR | HIDDEN_LAYER | EMBEDDING_DIM | NUM_EPOCH |
| -------- | -------- | -------- |-|
| 0.001     | 64 | 16 | 3000 |


### Net

![](https://i.imgur.com/InUM1lC.png)


透過 Embedding 將每個點投影到較低維度的向量，再將 Embedded 的點做相似度判斷來計算機率。

#### Embedding

1. Linear([torch.nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html))
2. 三層的 Graph Convolution ([torch_geometric.nn.GCNConv](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gcn_conv.html#GCNConv)) + tanh([torch.tanh](https://pytorch.org/docs/stable/generated/torch.tanh.html))

考量到可能有些 feature 是不重要的，因此先在一開始加一層 Linear 做篩選，再透過三層的 convolution 讓它看附近邊的資料。

#### Similarity

- [cosine similarity](https://zh.wikipedia.org/wiki/%E4%BD%99%E5%BC%A6%E7%9B%B8%E4%BC%BC%E6%80%A7)：我使用 pytorch 的 [CosineSimilarity](https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html)(torch.nn.CosineSimilarity)。

### Train

- loss function：Binary Cross Entropy ([torch.nn.BCELoss](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html))
- optimizer：Adam ([torch.optim.Adam](https://pytorch.org/docs/stable/optim.html))

```
for epoch
    h = embedding(data.feature)
    out = similiarty(pred_edge_pair(h))
    loss = criterion(out, y)
    loss.backward()
```

### online judge accuracy

- dataset1
    - auc1：0.9282
    - ap1：0.9409
- dataset2
    - auc2：0.9439
    - ap2：0.9353
- dataset3
    - auc3：0.9333
    - ap3：0.9345
- mauc：0.9351
- map：0.9369
- final：0.9360

:::success
儘管正確率其實只有高 Random Walk with Restart 一點點，但執行時間可以說是大大的提升了，**訓練時間不會超過一分鐘**，卻能達到如此的效果。
:::

### overfitting

在訓練過程中，其實是有發生 overfitting 導致正確率無法再上升的情況，在訓練集大約在 10 epoch 之後正確率就會達到 0.99，但上傳平台卻只能在 0.9 左右，因此我有做過以下的調整：

- epoch：因為資料集小，訓練太多次反而效果差
- hidden layer：減少神經元數量
- embedding dimension：減少 embedding 維度
- convolution layer：試過 5 層，但沒有比較好
- activation function：relu, leaky_relu, sigmoid, softmax
- normalization

但最後上傳的正確率仍落在 0.93 左右，應該再嘗試更大的改動。

### Next Step...

- 更多 feature
- 改變 network 架構或算法
- 串連不同種的 link prediction model

## :evergreen_tree: Decision Trees
[Google Colab](https://colab.research.google.com/drive/1lswk3nCwt5uaUsmums_QsydngTTbkR9j?usp=sharing)

我有拿 overfitting 的情況跟助教討論過，助教建議我除了加入更多 feature 之外，也可以使用 decision tree 看看，因此嘗試用 scikit learn 的一些套件來訓練看看。

- DecisionTreeClassifier

![](https://i.imgur.com/DVeXsdw.png)

- RandomForestClassifier

![](https://i.imgur.com/pLhTUjN.png)

- XGBClassifier

![](https://i.imgur.com/z3YeqCs.png)

但效果似乎沒有很好，感覺單一使用好像並沒有特別出色，可能得搭配其他模型或是做更深入的調整。

## 心得

![](https://i.imgur.com/KWwghly.png)

在去詢問助教之後，最後在上傳區有達到 95% 左右的正確率，我並沒有新增 feature 或是使用更新的方法了，主要都是在調整模型，我做的更動大概如下：

#### 減少網路的 layer 數量

雖然原本也就只有 3 層，但我發現再拿掉一層正確率會提升，也有試過再多加，但會降低正確率，於是我最終的模型 GCN convolution 僅有兩層而已。

#### 增加神經元數目跟 Embedding 維度

原先我是按照作業一 DrBC 的參數去定網路的架構，我一開始神經元數目跟 embedding 維度分別是 128 跟 64，後來想說 overfitting 所以又將它調小，之後意外調大時發現效果居然比較好，我就一路把它調到 2048 跟 1024，正確率竟然可以上到 94, 95% 呢！

:::info
在調神經元數目跟 embedding 維度時，同時也會將 learning rate 調低，最後大概調在 0.0000004，是一個非常小的數字。

此外，在調 learning rate 時會發現正確率上升大概有三種情況：

##### (1) 一次上升到最高，然後在一個值上下擺動

這個時候就得調低 learning rate，好讓參數可以梯度下降到更低的值。

##### (2) 穩定上升

會有直覺這個情況是最好的，但不知道為什麼讓他穩定上升反而會 overfitting ，得調高 learning rate

##### (3) 一開始上升很快，但增加幅度漸漸減少

這是最理想的情況，正確率可以達到最高，但必須依據情況更改訓練的 epoch 數量，訓練次數太多也會導致正確率下降，以這次的資料為例，因為數量並不多，因此我只訓練 60 次左右，同時根據神經元數目跟 Embedding 維度的增加，得減少訓練數量
:::

#### optimizer

我有試過許多 torch 內建的 optimizer，但最後是用 [Rprop](https://pytorch.org/docs/stable/optim.html#torch.optim.Rprop) 得到最好的結果的，但 Adam, Adamax 的效果也都不錯

雖然上傳結果有達到 95, 96%，但我認爲這個成績有一部份算是運氣，因為是隨機切分訓練集跟測試集的，每次正確率也都有 1％ 左右的浮動，因此我在想有些能達到 96％ 的成績應該也算是剛好讓我遇到。不過每次訓練都會穩定維持在 93％ 以上，所以還蠻有把握模型有 93％ 以上的正確率的。

## 參考資料

- [mlg_03-2_linkpred_attribute(上課投影片)](https://moodle.ncku.edu.tw/pluginfile.php/650542/mod_resource/content/0/mlg_03-2_linkpred_attribute.pdf)