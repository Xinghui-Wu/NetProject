import snap
from dataset import dataset_mine
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, integrate
from pylab import *
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
import seaborn as sns  # for making plots

amazon_path='../dataset/com-amazon.ungraph.txt'

dblp_path='../dataset/com-dblp.ungraph.txt'

cit_hepPh_path='../dataset/cit-HepPh.txt'
cit_hepPh_dates_path='../dataset/cit-HepPh-dates.txt'

cit_hepTh_path='../dataset/cit-HepTh.txt'
cit_hepTh_dates_path='../dataset/cit-HepTh-dates.txt'
cit_hepTh_abstracts_path='../dataset/cit-HepTh-abstracts.tar.gz'


G_amazon=dataset_mine.G_return(amazon_path)
G_dblp=dataset_mine.G_return(dblp_path)
G_hepTh=dataset_mine.G_return(cit_hepTh_path)
G_hepPh=dataset_mine.G_return(cit_hepPh_path)
# snap.PrintInfo(G_amazon, "QA Stats", "qa-info.txt", False)





#（度分布）相关函数
#获取图中拥有某度数的节点数
# 打印格式为： m nodes with degree n
# 返回array={'拥有度数1的概率':度数,'拥有度数2的概率':度数2,...,'拥有度数n的概率':度数n},
def getDegreeDistribution(G):
    a=[]
    sum=0
    DegToCntV=G.GetDegCnt()
    NIdCCfH = G.GetNodeClustCfAll()
    node_num=len(NIdCCfH)
    snap.GetDegCnt(G,DegToCntV)
    for item in DegToCntV:
        print("%d nodes with degree %d" % (item.GetVal2(), item.GetVal1()))
        a.append([item.GetVal2()/node_num,item.GetVal1()])
        sum=sum+item.GetVal2()*item.GetVal1()
    avg=sum/node_num
    return np.array(a),avg


#（节点中心性）相关函数
#返回所有节点PRank值的集合
def getPRankH(G):
    a=[]
    sum=0
    PRankH=G.GetPageRank()
    for item in PRankH:
        print(item, PRankH[item])
        a.append(PRankH[item])
        sum=sum+PRankH[item]
    avg=sum/len(a)
    return np.array(a),avg

#（节点中心性）相关函数
#获取所有节点的度中心性集合
def getGraphCenter(G):
    a={}#存每个节点的度数,[{节点序号:度数},...,]
    b=[]#存所有节点的度中心
    sum=0
    InDegV=G.GetNodeInDegV()
    for item in InDegV:
        a[item.GetVal1()]=item.GetVal2()
    node_num=len(a)
    for key in a.keys():
        b.append(a[key]/(node_num-1))
        sum=sum+a[key]/(node_num-1)
    avg=sum/len(b)
    return np.array(b),avg

#(节点中心性）相关函数
#获取每个节点的点介数和每条边的边介数
def getBetweeness(G):
    a=[]#点介数
    b=[]#边介数
    print('进入了！')
    Nodes, Edges = G.GetBetweennessCentr(1.0)
    for node in Nodes:
        print("node: %d centrality: %f" % (node, Nodes[node]))
        a.append(Nodes[node])
    for edge in Edges:
        b.append(Edges[edge])
        print("edge: (%d, %d) centrality: %f" % (edge.GetVal1(), edge.GetVal2(), Edges[edge]))
    #返回点介数，平均点介数，边介数，平均边介数
    return np.array(a),sum(a)/len(a),np.array(b),sum(b)/len(b)



#(节点聚类性）相关函数
#返回每个节点的聚类系数
def getNodeClust(G):
    a=[]
    sum=0
    NIdCCfH = G.GetNodeClustCfAll()
    print(len(NIdCCfH))
    for item in NIdCCfH:
        a.append(NIdCCfH[item])
        sum=sum+NIdCCfH[item]
        # print(item, NIdCCfH[item])
    avg=sum/len(a)
    return np.array(a),avg


#(节点聚类性）相关函数
# 返回 平均聚合系数，dict {'度数':'该度数节点的平均聚类系数'}
def getClusterCof(G):
    a=[]
    GraphClustCoeff = G.GetClustCf(-1)
    print("Clustering coefficient: %f" % GraphClustCoeff)
    Cf, CfVec = G.GetClustCf(True, -1)
    print("Average Clustering Coefficient: %f" % (Cf))
    print("Coefficients by degree:\n")
    for pair in CfVec:
        print("degree: %d, clustering coefficient: %f" % (pair.GetVal1(), pair.GetVal2()))
        a.append([pair.GetVal2(),pair.GetVal1()])
    return GraphClustCoeff,np.array(a)

#网络直径
def getDiameter(G):
    NTestNodes = 10
    IsDir = False
    result = G.GetBfsEffDiamAll(NTestNodes ,IsDir)
    print(result)


#网络连通块
def getCnComV(G):
    ComponentDist = G.GetSccSzCnt()
    for comp in ComponentDist:
        print("Size: %d - Number of Components: %d" % (comp.GetVal1(), comp.GetVal2()))

#获取每个节点所在弱连接中有多少个节点
def getMx(G):
    a= {}#用于节点标记,节点为key
    b=[]#用于记录连通及连通中的节点个数
    InDegV = G.GetNodeInDegV()
    for item in InDegV:
        sum=0
        if a.__contains__(item.GetVal1()):#如果之前记录过该节点就跳过
            pass
        else:#没记录就记录，并且遍历该节点所在连通图中的节点
            a[item.GetVal1()]=1 #加入节点记录
            #获取当前节点所在弱连接
            CnCom = G.GetNodeWcc(item.GetVal1())
            for node in CnCom:
                a[node]=1#将该连通图中所有节点加入dict
                print('当前节点：{},连通图拥有节点：{}'.format(item.GetVal1(),node))
                sum=sum+1
            b.append([sum])#记录下该弱连通图的所有节点数量(数量）
    return np.array(b)

def plotCenterFeatureDist(a,b,x1_label='',x2_label='',y1_label='',y2_label=''):
    from matplotlib.font_manager import FontProperties
    myfont = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf', size=14)
    sns.set(font=myfont.get_name())
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    f, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=False, sharey=False)
    sns.set(style="white", palette="muted", color_codes=True)  # 设置
    sns.distplot(a, label='latitude', color="m", bins=100, ax=axes[0]);
    sns.distplot(b, label='longitude', color="m", bins=100, ax=axes[1]);
    sns.despine(left=True)  # 删除左边边框
    plt.setp(axes, yticks=[])
    axes[0].set_xlabel(x1_label)
    axes[0].set_ylabel(y1_label)
    axes[1].set_xlabel(x2_label)
    axes[1].set_ylabel(y2_label)

    plt.show()

# 画普通条形图的（体现数据分布），可以为节点度数的图服务，一旦数据量一大，这个画法将耗时超长
def pltBar(a,xlabel,ylabel,title,width=0.8,text1='',text2='',text3=''):
    x=a[:,1]
    y=a[:,0]
    plt.bar(x,y,width=width)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.text(max(x)*0.5, max(y)*0.5, text1, fontdict={'size': '15', 'color': 'r'})
    plt.text(max(x)*0.5, max(y)*0.7, text2, fontdict={'size': '15', 'color': 'g'})
    plt.text(max(x)*0.5, max(y)*0.9, text3, fontdict={'size': '15', 'color': 'b'})
    plt.show()

# 画普通一维数据分布的，对数据量很大的一维数据很有效，可为节点中心性服务
def plt1Ddistri(a,xlabel,ylabel,title,text1='',text2='',text3=''):
    sns.distplot(a,kde=False,norm_hist=False)
    # plt.ylim(0,a.shape[0])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    print(text1)
    plt.text(max(a) * 0.3, a.shape[0] * 0.18, text1, fontdict={'size': '15', 'color': 'r'})
    plt.text(max(a) * 0.3, a.shape[0] * 0.22, text2, fontdict={'size': '15', 'color': 'g'})
    plt.text(max(a) * 0.3, a.shape[0] * 0.26, text3, fontdict={'size': '15', 'color': 'b'})
    plt.show()

#一维数据画bar，传入的数组每一个元素都是一个类
def pltBar1(y,xlabel,ylabel,title,width=0.8,text=''):
    x=np.arange(1,y.shape[0]+1)
    plt.axes(yscale='log')
    plt.bar(x,y[:,0],width=width)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.title(title)
    plt.text(max(x)*0.5, max(y[:,0])*0.5, text, fontdict={'size': '16', 'color': 'r'})
    plt.show()

#专门为连通图设立的，画出的横坐标是连通图中的节点个数，纵坐标是有几个这样的连通图
def pltBar2(Mx,xlabel,ylabel,title,width=0.8,text=''):
    Mx_dic = {}
    for i in Mx:

        if Mx_dic.__contains__('{}'.format(i[0])):
            print('xxx')
            print('i[0]:{}'.format(i[0]))
            Mx_dic['{}'.format(i[0])] = 1 + Mx_dic['{}'.format(i[0])]
            print("Mx_dic[{}]".format(i[0]))
        else:
            Mx_dic['{}'.format(i[0])] = 1

    key_sorted_result = sorted(Mx_dic.items(), key=lambda item: int(item[0]), reverse=False)
    x = [x[0] for x in key_sorted_result]
    y = [x[1] for x in key_sorted_result]

    plt.bar(x, y,width=width)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    for a, b in zip(x,y):
        plt.text(a, b + 0.001, '%d' % b, ha='center', va='bottom', fontsize=9)
    plt.show()



if __name__ == '__main__':


#---------------------------------------------------DBLP------------------------------------------------------
    #画关于节点度分布的图
    # degree_node,degree_avg = getDegreeDistribution(G_dblp)
    # degree_max=degree_node.max()
    # degree_min=degree_node.min()
    # pltBar(degree_node,'度数','拥有该度的节点概率',"DBLP数据集节点度数分布图",0.8,
    #        "度的平均={:.2f}".format(degree_avg),"度的最大值={:.2f}".format(degree_max),"度的最小值={:.2f}".format(degree_min))


    # #画关于节点中心性分布的图
    #画出PRank
    # PRankH,PRankH_avg=getPRankH(G_dblp)
    # PRankH_max=PRankH.max()
    # PRankH_min=PRankH.min()
    # Center,avg_Center=getGraphCenter(G_dblp)
    # plt1Ddistri(PRankH,'节点的PRank值','拥有该PRank值的节点数量','DBLP数据集节点中心性分析--节点PRank值分布情况',
    #           "节点平均PRank值={:.8f}".format(PRankH_avg), "节点最大PRank值={:.8f}".format(PRankH_max), "节点最小PRank值={:.8f}".format(PRankH_min))

    # 画出度中心性
    # degreeCenter,degreeCenter_avg=getGraphCenter(G_dblp)
    # degreeCenter_max=degreeCenter.max()
    # degreeCenter_min=degreeCenter.min()
    # plt1Ddistri(degreeCenter,'节点的度中心性值','拥有该度中心性值的节点数量','DBLP数据集节点中心性分析--节点度中心性值分布情况',
    #           "节点平均度中心性值={:.8f}".format(degreeCenter_avg), "节点最大度中心性值={:.8f}".format(degreeCenter_max), "节点最小度中心性值={:.8f}".format(degreeCenter_min))


    # 画出点介数、边介数(要计算很久）
    # nodeBetw,avgNodeBetw,edgeBetw,avgEdgeBetw=getBetweeness(G_hepTh)
    # plotCenterFeatureDist(nodeBetw,edgeBetw)


    # # 画关于节点聚类性的图
    # 画出每个节点的聚类系数
    # nodeClust,nodeClust_avg=getNodeClust(G_dblp)
    # nodeClust_max=nodeClust.max()
    # nodeClust_min=nodeClust.min()
    # print(nodeClust.shape)
    # plt1Ddistri(nodeClust,'节点的聚类系数','拥有该聚类系数的节点数量','DBLP数据集节点聚类分析(1)',
    #         "节点平均聚类系数值={:.8f}".format(nodeClust_avg), "节点最大聚类系数值={:.8f}".format(nodeClust_max), "节点最小聚类系数值={:.8f}".format(nodeClust_min))
    # 画出每个度数的平均聚类系数
    # GraphClustCoeff,clusters=getClusterCof(G_dblp)
    # print(clusters)
    # pltBar(clusters, '度数', '该度数节点的平均聚类系数', "DBLP数据集节点聚类分析(2)",width=0.8,
    #        text1="图的平均聚类系数={:4f}".format(GraphClustCoeff),
    #        text2='',
    #        text3='')

#______________________________________Amazon数据集-----------------------------------------

    degree_node,degree_avg = getDegreeDistribution(G_amazon)
    degree_max=degree_node.max()
    degree_min=degree_node.min()
    pltBar(degree_node, '度数', '拥有该度的节点概率', "Amazon数据集节点度数分布图", 0.8,
       "度的平均={:.2f}".format(degree_avg),"度的最大值={:.2f}".format(degree_max),"度的最小值={:.2f}".format(degree_min))

# #画关于节点中心性分布的图
    #画出PRank
    # PRankH,PRankH_avg=getPRankH(G_amazon)
    # PRankH_max=PRankH.max()
    # PRankH_min=PRankH.min()
    # Center,avg_Center=getGraphCenter(G_amazon)
    # plt1Ddistri(PRankH,'节点的PRank值','拥有该PRank值的节点数量','Amazon数据集节点中心性分析--节点PRank值分布情况',
    #           "节点平均PRank值={:.8f}".format(PRankH_avg), "节点最大PRank值={:.8f}".format(PRankH_max), "节点最小PRank值={:.8f}".format(PRankH_min))

    # 画出度中心性
    # degreeCenter,degreeCenter_avg=getGraphCenter(G_amazon)
    # degreeCenter_max=degreeCenter.max()
    # degreeCenter_min=degreeCenter.min()
    # plt1Ddistri(degreeCenter,'节点的度中心性值','拥有该度中心性值的节点数量','Amazon数据集节点中心性分析--节点度中心性值分布情况',
    #           "节点平均度中心性值={:.8f}".format(degreeCenter_avg), "节点最大度中心性值={:.8f}".format(degreeCenter_max), "节点最小度中心性值={:.8f}".format(degreeCenter_min))


    # 画出点介数、边介数(要计算很久）
    # nodeBetw,avgNodeBetw,edgeBetw,avgEdgeBetw=getBetweeness(G_hepTh)
    # plotCenterFeatureDist(nodeBetw,edgeBetw)


    # # 画关于节点聚类性的图
    # 画出每个节点的聚类系数
    # nodeClust,nodeClust_avg=getNodeClust(G_amazon)
    # nodeClust_max=nodeClust.max()
    # nodeClust_min=nodeClust.min()
    # print(nodeClust)
    # plt1Ddistri(nodeClust,'节点的聚类系数','拥有该聚类系数的节点数量','Amazon数据集节点聚类分析(1)',
    #         "节点平均聚类系数值={:.8f}".format(nodeClust_avg), "节点最大聚类系数值={:.8f}".format(nodeClust_max), "节点最小聚类系数值={:.8f}".format(nodeClust_min))

    # 画出每个度数的平均聚类系数
    # GraphClustCoeff,clusters=getClusterCof(G_amazon)
    # print(clusters)
    # pltBar(clusters, '度数', '该度数节点的平均聚类系数', "Amazon数据集节点聚类分析(2)",width=0.8,
    #        text1="图的平均聚类系数={:8f}".format(GraphClustCoeff),
    #        text2='',
    #        text3='')



#----------------------画小型数据集连通块分析的图----------------------------
    # 画出关于网络连通块分析的图(x=连通块中节点个数，y=拥有该节点数的连通块个数）
    # Mx=getMx(G_hepTh)
    # pltBar2(Mx,'连通块中拥有的节点个数','连通块个数','cit-HepTh数据集中连通块的数量及其拥有节点个数关系')

    # Mx=getMx(G_hepPh)
    # pltBar2(Mx,'连通块中拥有的节点个数','连通块个数','cit-HepPh数据集中连通块的数量及其拥有节点个数关系')

