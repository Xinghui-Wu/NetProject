# NetProject

## 环境配置

&emsp;&emsp;建议使用conda或viturlenv创建Python虚拟环境，启动该虚拟环境后，在项目根目录下，输入以下命令来配置环境。

    pip install -r requirements.txt


## 实验数据集

[Stanford Large Network Dataset Collection](http://snap.stanford.edu/data/index.html)

* [Amazon product co-purchasing network and ground-truth communities](http://snap.stanford.edu/data/com-Amazon.html)
* [DBLP collaboration network and ground-truth communities](http://snap.stanford.edu/data/com-DBLP.html)
* [High-energy physics theory citation network](http://snap.stanford.edu/data/cit-HepTh.html)
* [High-energy physics citation network](http://snap.stanford.edu/data/cit-HepPh.html)


## 网络数据特征分析

&emsp;&emsp;对真实网络数据集进行特征分析，包括但不限于：

* 节点级别特征分析：节点度分布（node degree)、节点中心性（node centrality）、节点聚类系数（clustering coefficient）等；

* 网络级别特征分析：网络直径（diameter）、网络连通块（connected components）分析统计等。


## 网络数据机器学习任务

&emsp;&emsp;基于真实网络数据集进行节点分类（node classification）、节点聚类（node clustering）或链接预测（link prediction）等机器学习任务，具体要求：

* 对每个任务选取5+个现有相应算法进行实验分析，所选算法应多样需覆盖多个技术类别。

### 特征提取

&emsp;&emsp;实验选择表示论文之间的相互引用关系的cit-HepTh和cit-HepPh两个数据集。数据集提供了每篇论文的基本元信息，包括论文标题、作者、发表时间、发表期刊和摘要内容等。根据摘要内容，计算每篇论文的文本特征向量；根据引用网络中的被引用量，对论文进行标签标记。

&emsp;&emsp;feature模块下的feature_extraction.py脚本提供了用于特征提取和数据集获取的接口。该脚本文件为每个函数都提供了比较详细的pydoc文档说明。此处，仅提供顶层接口的输入输出和调用说明，省略对后续分类、聚类、预测等机器学习任务透明的其它函数的说明。

&emsp;&emsp;如下所示，get_datasets()接口接受数据集名称（注意大小写）、是否打乱数据集顺序以及训练集和验证集的划分比例元组（其余部分作为测试集）作为输入，直接返回按照指定比例划分好的训练集、验证集和测试集。每个子集均为一个列表，由若干论文样本组成，每个元素对应于一个论文样本的特征向量和标签元组。

```python
def get_datasets(dataset="cit-HepTh", shuffle=True, proportion=(0.7, 0.2)):
    """指定数据集名称，获取指定比例的训练集、验证集和测试集
    每个子集均为一个列表，列表中的每个元素对应于一个论文样本的特征向量和标签元组，即(feature_vector, label)

    Args:
        dataset (str, optional): 数据集名称，cit-HepTh或cit-HepPh（注意大小写）. Defaults to "cit-HepTh".
        shuffle (bool, optional): 是否打乱数据集顺序. Defaults to True.
        proportion (tuple, optional): 训练集和验证集的比例. Defaults to (0.7, 0.2).

    Returns:
        tuple: 训练集、验证集、测试集
    """
```