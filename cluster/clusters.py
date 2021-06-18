#聚类算法：
    # kmeans
    # 密度聚类：DBSCAN
    # 层次聚类：AgglomerativeClustering  
    # 谱聚类：SpectralClustering
    # 分批kmeans：MiniBatchKMeans
# 评价指标：FMI（Fowlkes–Mallows index）
# 排除：特征聚类：FeatureAgglomeration# 亲和传播聚类（AP）聚类：affinity_propagation# 偏移均值向量：MeanShift
import numpy as np
import sklearn.cluster as cluster
import os
import time
import argparse
import csv
from sklearn import metrics
import sys
sys.path.append('./feature/')
import feature_extraction as fe

def sort_data(data_list):
    x_list=[]
    y_list=[]
    for data in data_list:
        x_list.append(data[0])
        y_list.append(data[1])
    x_array=np.array(x_list)
    y_array=np.array(y_list)
    return x_array,y_array

def print_run_time(func):  
    def wrapper(*args, **kw):  
        local_time = time.time()  
        output=func(*args, **kw)
        time_cost=time.time() - local_time
        print('{} run time is {}'.format(func.__name__,time_cost))
        with open("./cluster/tmp.csv","a+") as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow([func.__name__,output,time_cost])
        return output,time_cost
    return wrapper

@print_run_time
def kmeans (train_x,train_y,num_cluster = 5):
    km_cluster = cluster.KMeans(n_clusters=num_cluster)
    km_cluster.fit(train_x)

    #FMI指数：与真实值对比
    fmi = metrics.fowlkes_mallows_score(train_y,km_cluster.labels_)
    # print("kmeans的FMI评价分值为：%f"%(fmi))
    return fmi

@print_run_time
def dbscan(train_x,train_y):
    # 密度聚类
    db = cluster.DBSCAN(eps=0.2,min_samples=3)
    db.fit(train_x)

    #FMI指数：与真实值对比
    fmi = metrics.fowlkes_mallows_score(train_y,db.labels_)
    return fmi

@print_run_time
def AC(train_x,train_y,num_cluster = 5):
    # 层次聚类
    ac = cluster.AgglomerativeClustering(n_clusters=num_cluster)
    ac.fit(train_x)
    predicted_labels = ac.fit_predict(train_x)
    # #计算ARI指数
    # ARI = (metrics.adjusted_rand_score(train_y, predicted_labels))

    #FMI指数：与真实值对比
    fmi = metrics.fowlkes_mallows_score(train_y,ac.labels_)
    
    return fmi
@print_run_time
# def AP(train_x,train_y):
#     #亲和传播聚类（AP）聚类
#     ap = cluster.affinity_propagation(preference=-50).fit(train_x)

#     #FMI指数：与真实值对比
#     fmi = metrics.fowlkes_mallows_score(train_y,ap.labels_)
#     return fmi 

# @print_run_time
# def meanshift(train_x,train_y):
#     #偏移均值向量（meanshift）
#     ms = cluster.MeanShift(bandwidth=2).fit(train_x)

#     #FMI指数：与真实值对比
#     fmi = metrics.fowlkes_mallows_score(train_y,ms.labels_)
#     return fmi

@print_run_time
def S_C(train_x,train_y,num_cluster = 5):
    #谱聚类
    sc = cluster.SpectralClustering(n_clusters=num_cluster).fit(train_x)

    #FMI指数：与真实值对比
    fmi = metrics.fowlkes_mallows_score(train_y,sc.labels_)
    return fmi

# @print_run_time
# def FA(train_x,train_y,num_cluster = 5):
#     #特征聚类
#     fa = cluster.FeatureAgglomeration(n_clusters=num_cluster).fit(train_x)

#     #FMI指数：与真实值对比
#     fmi = metrics.fowlkes_mallows_score(train_y,fa.labels_)
#     return fmi

@print_run_time
def MBK(train_x,train_y,num_cluster = 5):
    #分批kmeans
    mbk = cluster.MiniBatchKMeans(n_clusters=num_cluster).fit(train_x)

    #FMI指数：与真实值对比
    fmi = metrics.fowlkes_mallows_score(train_y,mbk.labels_)
    return fmi

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--dataset", type=str, default="cit-HepPh", help="")
    parser.add_argument("-t", "--task", type=int, default=1, help="")
    parser.add_argument("-f", "--feature_type", type=int, default=0, help="")
    parser.add_argument("-l", "--label_type", type=int, default=2, help="")
    parser.add_argument("-s", "--shuffle", type=bool, default=True, help="")
    parser.add_argument("-p", "--proportion", type=tuple, default=(0.7, 0.3), help="")
    parser.add_argument("-m", "--method", type=str, default='all',choices=['kmeans','dbscan','AC','AP','meanshift','S_C','FA','MBK','all'], help="")
    parser.add_argument("-sp", "--save_path", type=str, default='./cluster/result.csv', help="")  
    args = parser.parse_args()

    training_set, validation_set, test_set = fe.get_datasets(dataset=args.dataset, task=args.task,
                                                             feature_type=args.feature_type, label_type=args.label_type,
                                                             shuffle=args.shuffle, proportion=args.proportion)
    train_x,train_y=sort_data(training_set)
    val_x,val_y=sort_data(validation_set)

    with open("./cluster/tmp.csv","w") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(['method','index','time_cost'])

    if args.method=='kmeans':
        acc = kmeans(train_x,train_y,len(np.unique(train_y)))
    elif args.method=='dbscan':
        acc = dbscan(train_x,train_y)
    elif args.method=='AC':
        acc = AC(train_x,train_y,len(np.unique(train_y)))
    elif args.method=='AP':
        acc = AP(train_x,train_y)
    elif args.method=='meanshift':
        acc = meanshift(train_x,train_y)
    elif args.method=='S_C':
        acc = S_C(train_x,train_y,len(np.unique(train_y)))
    elif args.method=='FA':
        acc = FA(train_x,train_y,len(np.unique(train_y)))
    elif args.method=='MBK':
        acc = MBK(train_x,train_y,len(np.unique(train_y)))
    elif args.method=='all':
        acc_k = kmeans(train_x,train_y,len(np.unique(train_y)))
        acc_ac = AC(train_x,train_y,len(np.unique(train_y)))
        acc_sc = S_C(train_x,train_y,len(np.unique(train_y)))
        # acc_fa = FA(train_x,train_y,len(np.unique(train_y)))           ValueError: Found input variables with inconsistent numbers of samples: [7414, 24684]
        acc_mbk = MBK(train_x,train_y,len(np.unique(train_y)))
        acc_db = dbscan(train_x,train_y)
        # acc_ap = AP(train_x,train_y)          affinity_propagation() missing 1 required positional argument: 'S'
        # acc_ms = meanshift(train_x,train_y)   timesout
        

    tmp_path=os.path.abspath('./cluster/tmp.csv')
    os.rename('./cluster/tmp.csv',args.save_path)
