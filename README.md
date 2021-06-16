# NetProject

## 1 分析

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