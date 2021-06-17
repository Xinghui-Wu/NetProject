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

### 图G的获取：

导入dataset下的dataset_mine包，使用其中的函数G_return(path)即可，如

~~~
from dataset import dataset_mine
dblp_path='../dataset/com-dblp.ungraph.txt'
G_dblp=dataset_mine.G_return(dblp_path)#返回的是一个图G
~~~

### 关于图的分析函数

#### 节点度分析

##### - getDegreeDistribution(G)

参数 ：

​	图G

返回值：

 - array, 这个array={'拥有度数1的节点数':度数,'拥有度数2的节点数':度数2,...,'拥有度数n的节点数':度数n}
 - 节点度数的平均值

#### 节点中心性分析

##### - getPRankH(G)

参数 ：

​	图G

返回值：

 - array,  该array是返回所有节点PRank值的集合
 - 节点的平均PRank值

##### - getGraphCenter(G)

参数 ：

​	图G

返回值：

 - array, 这个array是所有节点的度中心性集合
 - 节点的平均度中心性值

##### - getBetweeness(G) 跑不出来

参数 ：

​	图G

返回值：

​    点介数，平均点介数，边介数，平均边介数

#### 节点聚类性分析

##### - getNodeClust(G)

参数 ：

​	图G

返回值：

- array, 这个array是每个节点的聚类系数的集合
- 节点的平均聚类系数

##### - getClusterCof(G)

参数 ：

​	图G

返回值：

- 平均聚类系数
- array，这个array的每个元素为['度数':'该度数节点的平均聚类系数']

### 网络分析

#### 网络直径

##### - getDiameter(G)

参数 ：

​	图G

返回值：

​    网络直径

#### 网络连通块

##### - getCnComV(G)

参数 ：

​	图G

返回值：

​    网络中的连通块及其大小

##### - getMx(G)

参数 ：

​	图G

返回值：

​    array, 这个array的每个元素代表一个连通图中包含的节点个数


## 网络数据机器学习任务

&emsp;&emsp;基于真实网络数据集进行节点分类（node classification）、节点聚类（node clustering）或链接预测（link prediction）等机器学习任务，具体要求：

* 对每个任务选取5+个现有相应算法进行实验分析，所选算法应多样需覆盖多个技术类别。

### 特征提取

&emsp;&emsp;实验选择表示论文之间的相互引用关系的cit-HepTh和cit-HepPh两个数据集。数据集提供了每篇论文的基本元信息，包括论文标题、作者、发表时间、发表期刊和摘要内容等。根据摘要内容，计算每篇论文的文本特征向量；根据引用网络中的被引用量，对论文进行标签标记。

&emsp;&emsp;feature模块下的feature_extraction.py脚本提供了用于特征提取和数据集获取的接口。该脚本文件为每个函数都提供了比较详细的pydoc文档说明。此处，仅提供顶层接口的输入输出和调用说明，省略对后续分类、聚类、预测等机器学习任务透明的其它函数的说明。

&emsp;&emsp;如下所示，get_datasets()接口接受数据集名称（注意大小写）、机器学习任务类型、特征类型、标签类型、是否打乱数据集顺序以及训练集和验证集的划分比例元组（其余部分作为测试集）作为输入，直接返回按照指定比例划分好的训练集、验证集和测试集。每个子集均为一个列表，由若干论文样本组成，每个元素对应于一个论文样本的特征向量和标签元组。

```python
def get_datasets(dataset="cit-HepTh", task=0, feature_type=0, label_type=2, shuffle=True, proportion=(0.7, 0.2)):
    """指定数据集名称、机器学习任务类型、样本特征类型，获取指定比例的训练集、验证集和测试集
    对于节点分类和聚类任务，每个子集均为一个列表，列表中的每个元素对应于一个论文样本的特征向量和标签元组，即(feature_vector, label)
    对于链接预测任务，每个子集均为一个列表，列表中的每个元素对应于一对论文样本的特征向量和有无有向边的标签元组，即(feature_vector_i, feature_vector_j, 0/1)

    Args:
        dataset (str, optional): 数据集名称，cit-HepTh或cit-HepPh（注意大小写）. Defaults to "cit-HepTh".
        task (int, optional): 机器学习任务类型，0表示节点分类和聚类，1表示链接预测. Defaults to 0.
        feature_type (int, optional): 特征类型，0表示文本特征，1表示网络结构特征，2表示混合特征. Defaults to 0.
        label_type (int, optional): 标签类型，1表示对数标签划分，2表示二分类标签划分. Defaults to 2.
        shuffle (bool, optional): 是否打乱数据集顺序. Defaults to True.
        proportion (tuple, optional): 训练集和验证集的比例. Defaults to (0.7, 0.2).

    Returns:
        tuple: 训练集、验证集、测试集
    """
```

### 分类预测

&emsp;&emsp;实验中使用6种分类算法共计七个实现对提取到的特征数据进行预测，算法包括随机森林“Random Forest”, 贝叶斯分类器，决策树分类器“Dicision Tree”，支持向量机“SVM”（kernel=rbf、sigmoid），Logistic回归“Logistic Regression”，神经网络“Neural Network”.

&emsp;&emsp;classification模块下的classification.py脚本提供了各个分类算法的接口，在repo的主目录可以直接调用该脚本进行预测。使用方法如下：
```
python -d your_dataset_name -m all -sp your_result_save_path
```

&emsp;&emsp;其中 -m 后的参数指定了使用的算法，'all'即使用所有算法，其他的算法调用对应参数可见的classification.py脚本。
注意，各个分类方法的必要输入仅为训练输入与标签，测试输入与标签。该输入与标签可以通过脚本内的sort_data()函数从get_datasets()的输出中直接得到。分类方法的输出结果包括在测试集上的accuracy以及整个算法运行的时间消耗。这些输出也会写入-sp指定路径的csv表格中。

&emsp;&emsp;目前根据已有的数据集运行的各个算法的accuracy以及time cost均记录在classification模块下[表格1](./classification/HepPh-result.csv)以及[表格2](./classification/HepTh-result.csv)中

### 链路预测

运行环境：Win10 + python3.7.9
运行依赖环境可以启动虚拟环境：**虚拟环境目录**：[项目主目录]/env

**启动虚拟环境**:
```shell
PS D:\MyCode_py\NetProject-main> ./env/Scripts/activate
```
**运行示例**：
```shell
(env) PS D:\MyCode_py\NetProject-main> python D:\MyCode_py\NetProject-main\link\link_pred.py -t 1 -d cit-HepPh
```
**退出虚拟环境**：
```shell
(env) PS D:\MyCode_py\NetProject-main> deactivate
```