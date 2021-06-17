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
***
* `Linux上虚拟环境的启动是不一样的，可以查看相关教程。`
* 我的`requirements.txt`如下：
```
certifi==2021.5.30
joblib==1.0.1
numpy==1.20.3
scikit-learn==0.24.2
# scipy==1.6.3 # 版本冲突
scipy==1.4.1
snap-stanford==6.0.0
threadpoolctl==2.1.0
keras==2.3.1
tensorflow==2.1.0
```
***
`HepPh-result.csv`的结果：
```
method,accuracy,time_cost

naive_bayes,0.5028319697923223,3.8866357803344727

random_forest,0.7507866582756451,16.25982928276062

logistic_regression,0.7495280050346129,14.235499620437622

decision_tree,0.7444933920704846,35.98634624481201

neural_network,0.634675920009613,183.2272915840149

support_vector_machine,0.4433606041535557,43.891860008239746

support_vector_machine,0.25298930144745124,46.319403886795044


```