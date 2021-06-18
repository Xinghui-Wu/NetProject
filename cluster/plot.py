
import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse
import clusters as cl
import sys
sys.path.append('./feature/')
import feature_extraction as fe

#k对kmeans的影响
def test_kmeans(X,labels):
    nums = range(2,10)
    scores = []
    for num in nums:
        score,fmi = cl.kmeans(X,labels,num)
        scores.append(score)
        with open("./cluster/kmeans.csv","a+") as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow([num,score,fmi])
    return scores

#k对网络聚类的影响
def test_AC(X,labels):
    nums = range(2,10)
    ARIS = []
    for num in nums:
        ARI = cl.AC(X,labels,num)
        ARIS.append(ARI)
        with open("./cluster/AC.csv","a+") as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow([num,ARI])
    return ARIS

def pl (s1):
    nums = range(2,10)
    plt.figure(1)
    # plt.subplot(1,2,1)
    plt.plot(nums,s1,color='r',marker = '^')
    plt.set_xlabel("n_clusters")
    plt.set_ylabel("scores")
    # plt.subplot(1,2,2)
    # plt.plot(nums,s2,color='k',marker='+')
    # plt.set_xlabel("n_clusters")
    # plt.set_ylabel("ARIS")
    # plt.figure.suptitle("kmeans&AgglomerativeClustering")
    plt.figure.suptitle("kmeans")
    plt.savefig('./cluster/compare.jpg')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--dataset", type=str, default="cit-HepPh", help="")
    parser.add_argument("-s", "--shuffle", type=bool, default=True, help="")
    parser.add_argument("-p", "--proportion", type=tuple, default=(0.7, 0.3), help="")
    args = parser.parse_args()

    training_set, validation_set, test_set = fe.get_datasets(dataset=args.dataset, shuffle=args.shuffle, proportion=args.proportion)
    train_x,train_y=cl.sort_data(training_set)
    
    with open("./cluster/kmeans.csv","a+") as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow(['number','score','FMI'])

    # with open("./cluster/AC.csv","a+") as csvfile: 
    #         writer = csv.writer(csvfile)
    #         writer.writerow(['number','ARI'])
    
    s1 = test_kmeans(train_x,train_y)
    # s2 = test_AC(train_x,train_y)

    pl(s1)