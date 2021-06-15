'''
Author: your name
Date: 2021-06-15 11:17:02
LastEditTime: 2021-06-15 18:36:41
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NetProject/classification/classification.py
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('./feature/')
sys.path.append('./classification/')
import feature_extraction as fe
import numpy as np
import argparse
import time
from utils import *

from sklearn import datasets,svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import keras
from keras.models import load_model
import csv



def print_run_time(func):  
    def wrapper(*args, **kw):  
        local_time = time.time()  
        output=func(*args, **kw)
        time_cost=time.time() - local_time
        print('{} run time is {}'.format(func.__name__,time_cost))
        with open("./classification/tmp.csv","a+") as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow([func.__name__,output,time_cost])
        return output,time_cost
    return wrapper

@print_run_time
def random_forest(train_x,train_y,val_x,val_y,depth=10):
    clf = RandomForestClassifier(max_depth=depth)
    clf.fit(train_x, train_y)
    pred = clf.predict(val_x)
    acc= np.sum(val_y == pred) / val_x.shape[0]
    return acc

@print_run_time
def support_vector_machine(train_x,train_y,val_x,val_y,method='rbf'):
    if method=='rbf':
        clf = svm.SVC(decision_function_shape="ovo",kernel="rbf",max_iter=10)#decision_function_shape="ovo", 
    elif method=='sigmoid':
        clf = svm.SVC(decision_function_shape="ovo",kernel="sigmoid",max_iter=10)#decision_function_shape="ovo", 
    else:
        print('svm method error')
        os._exit(0)
    train_x=preprocessing.scale(train_x)
    val_x=preprocessing.scale(val_x)
    clf.fit(train_x, train_y)
    pred = clf.predict(val_x)
    acc= np.sum(val_y == pred) / val_x.shape[0]
    print(acc)
    return acc

@print_run_time
def decision_tree(train_x,train_y,val_x,val_y,depth=10):
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(train_x, train_y)
    pred = clf.predict(val_x)
    acc= np.sum(val_y == pred) / val_x.shape[0]
    return acc

@print_run_time
def naive_bayes(train_x,train_y,val_x,val_y,var_smoothing=1e-8):
    clf = GaussianNB(var_smoothing=var_smoothing)
    clf.fit(train_x, train_y)
    pred = clf.predict(val_x)
    acc= np.sum(val_y == pred) / val_x.shape[0]
    return acc

@print_run_time
def logistic_regression(train_x,train_y,val_x,val_y,C=1.0, random_state=0):  
    clf = LogisticRegression(C=C, random_state=random_state)
    clf.fit(train_x, train_y)
    pred = clf.predict(val_x)
    acc= np.sum(val_y == pred) / val_x.shape[0]
    return acc

@print_run_time
def neural_network(train_x,train_y,val_x,val_y,epoch=10):
    train_x=train_x.reshape(train_x.shape[0],-1)
    val_x=val_x.reshape(val_x.shape[0],-1)
    num=max(len(np.unique(train_y)),len(np.unique(val_y)))
    train_y=keras.utils.to_categorical(train_y, num)
    val_y=keras.utils.to_categorical(val_y, num)
    
    input_dim=train_x.shape[1]
    model=build_model(input_dim=input_dim,class_num=num)
    # model=load_model('./classification/model.h5')
    
    model.fit(train_x,train_y,epochs=epoch,batch_size=64,validation_split=0.1)
    acc = model.evaluate(val_x, val_y, batch_size=128)[-1]
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--dataset", type=str, default="cit-HepPh", help="")
    parser.add_argument("-s", "--shuffle", type=bool, default=True, help="")
    parser.add_argument("-p", "--proportion", type=tuple, default=(0.7, 0.3), help="")

    parser.add_argument("-m", "--method", type=str, default='all',choices=['rf','bayes','svm-l','svm-rbf','logi','tree','nn','all'], help="")
    parser.add_argument("-sp", "--save_path", type=str, default='./classification/HepPh-result.csv', help="")  
    args = parser.parse_args()

    training_set, validation_set, test_set = fe.get_datasets(dataset=args.dataset, shuffle=args.shuffle, proportion=args.proportion)

    train_x,train_y=sort_data(training_set)
    val_x,val_y=sort_data(validation_set)

    with open("./classification/tmp.csv","w") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(['method','accuracy','time_cost'])

    if args.method=='rf':
        acc=random_forest(train_x,train_y,val_x,val_y)
    elif args.method=='bayes':
        acc=naive_bayes(train_x,train_y,val_x,val_y)
    elif args.method=='svm-l':
        acc=support_vector_machine(train_x,train_y,val_x,val_y,method='rbf')
    elif args.method=='svm-rbf':
        acc=support_vector_machine(train_x,train_y,val_x,val_y,method='rbf')
    elif args.method=='logi':
        acc=logistic_regression(train_x,train_y,val_x,val_y)
    elif args.method=='tree':
        acc=decision_tree(train_x,train_y,val_x,val_y)
    elif args.method=='nn':
        acc=neural_network(train_x,train_y,val_x,val_y)
    elif args.method=='all':
        acc_bayes=naive_bayes(train_x,train_y,val_x,val_y) #0.25
        acc_rf=random_forest(train_x,train_y,val_x,val_y) # 0.44-10depth
        acc_logi=logistic_regression(train_x,train_y,val_x,val_y)#0.45
        acc_tree=decision_tree(train_x,train_y,val_x,val_y)#0.42
        acc_nn=neural_network(train_x,train_y,val_x,val_y) # 0.4043
        acc_svm_rbf=support_vector_machine(train_x,train_y,val_x,val_y,method='rbf')
        acc_svm_sigmoid=support_vector_machine(train_x,train_y,val_x,val_y,method='sigmoid')

    tmp_path=os.path.abspath('./classification/tmp.csv')
    os.rename('./classification/tmp.csv',args.save_path)

    print(1)
