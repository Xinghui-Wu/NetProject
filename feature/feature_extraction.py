import argparse
import os
import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Paper:
    def __init__(self, id, abstract):
        self.id = id
        self.abstract = abstract
        self.text_feature = None
        self.network_feature = None
        self.mix_feature = None
        self.num_citations = 0


    def set_label(self):
        # if self.num_citations == 0:
        #     self.label = 0
        # else:
        #     self.label = round(math.log10(self.num_citations)) + 1
        
        # 二分类
        if self.num_citations <= 5:
            self.label = 0
        else:
            self.label = 1


def get_datasets(dataset="cit-HepTh", task=0, feature_type=0, shuffle=True, proportion=(0.7, 0.2)):
    """指定数据集名称、机器学习任务类型、样本特征类型，获取指定比例的训练集、验证集和测试集
    对于节点分类和聚类任务，每个子集均为一个列表，列表中的每个元素对应于一个论文样本的特征向量和标签元组，即(feature_vector, label)
    对于链接预测任务，每个子集均为一个列表，列表中的每个元素对应于一对论文样本的特征向量和有无有向边的标签元组，即(feature_vector_i, feature_vector_j, 0/1)

    Args:
        dataset (str, optional): 数据集名称，cit-HepTh或cit-HepPh（注意大小写）. Defaults to "cit-HepTh".
        task (int, optional): 机器学习任务类型，0表示节点分类和聚类，1表示链接预测. Defaults to 0.
        feature_type (int, optional): 特征类型，0表示文本特征，1表示网络结构特征，2表示混合特征. Defaults to 0.
        shuffle (bool, optional): 是否打乱数据集顺序. Defaults to True.
        proportion (tuple, optional): 训练集和验证集的比例. Defaults to (0.7, 0.2).

    Returns:
        tuple: 训练集、验证集、测试集
    """
    # 获取全部论文的元信息
    paper_dict = get_all_meta_info()
    # 文本特征提取
    if feature_type != 1:
        paper_dict = extract_text_features(paper_dict)
    # 网络结构特征提取
    if feature_type != 0:
        paper_dict = extract_network_features(paper_dict, dataset)
    # 获取混合数据集
    datasets = get_paper_citation_network(paper_dict, dataset, task, feature_type)

    # 打乱数据集顺序
    if shuffle:
        random.shuffle(datasets)
    
    # 划分训练集、验证集和测试集
    training_set = datasets[: int(proportion[0] * len(datasets))]
    validation_set = datasets[int(proportion[0] * len(datasets)): int((proportion[0] + proportion[1]) * len(datasets))]
    test_set = datasets[int((proportion[0] + proportion[1]) * len(datasets)): ]

    return training_set, validation_set, test_set


def get_all_meta_info():
    """获取全部论文的元信息

    Returns:
        dict: 论文字典
    """
    paper_dict = dict()

    dir = "./dataset/cit-HepTh-abstracts/"

    for year in range(1992, 2004):
        year_dir = os.path.join(dir, str(year))
        abs_filenames = os.listdir(year_dir)
        abs_filenames.sort()

        for abs_filename in abs_filenames:
            id = abs_filename[: -4]
            abs_path = os.path.join(year_dir, abs_filename)
            paper = get_paper_meta_info(id, abs_path)
            paper_dict[paper.id] = paper
    
    return paper_dict


def get_paper_meta_info(id, abs_path):
    """获取一篇论文的元信息

    Args:
        id (str): 论文ID
        abs_path (str): 论文元信息文件路径

    Returns:
        Paper: 论文对象
    """
    # 按行读取论文元信息文件
    with open(abs_path, 'r') as abs_file:
        abs = abs_file.readlines()
        abs = abs[2: -1]
    
    # 论文摘要起始位置
    abstract_start_index = abs.index("\\\\\n") + 1
    
    # 拼接论文摘要文本
    abstract = "".join(abs[abstract_start_index: ])
    abstract = abstract.replace("\n", " ")
    
    return Paper(id, abstract)


def extract_text_features(paper_dict):
    """TF-IDF文本特征提取

    Args:
        paper_dict (dict): 论文字典

    Returns:
        dict: 包含文本特征的论文字典
    """
    # 论文字典ID列表
    id_list = list(paper_dict.keys())

    # 论文摘要文本语料库
    corpus = list()

    for id in id_list:
        paper = paper_dict[id]
        corpus.append(paper.abstract)
    
    # 提取TF-IDF文本特征矩阵
    vectorizer = TfidfVectorizer()
    feature_matrix = vectorizer.fit_transform(corpus).toarray()

    # 设置每个论文样本的文本特征向量
    for i in range(len(id_list)):
        paper = paper_dict[id_list[i]]
        paper.text_feature = feature_matrix[i]
    
    return paper_dict


def extract_network_features(paper_dict, dataset):
    """网络结构特征提取

    Args:
        paper_dict (dict): 论文字典
        dataset (str): 数据集名称，cit-HepTh或cit-HepPh（注意大小写）

    Returns:
        dict: 包含网络结构特征的论文字典
    """
    with open("./results/{}-DeepWalk.txt".format(dataset), 'r') as deepwalk_result_txt:
        deepwalk_result = deepwalk_result_txt.readlines()
        deepwalk_result = deepwalk_result[1: ]
    
    for line in deepwalk_result:
        line = line[: -1]
        partition = line.split(' ')
        paper_id = partition[0]

        if paper_id in paper_dict:
            paper = paper_dict[paper_id]
            paper.network_feature = np.array(partition[1: ], dtype=float)
    
    return paper_dict


def get_paper_citation_network(paper_dict, dataset, task, feature_type):
    """获取论文引用网络信息（真实节点和边）
    对于节点分类和聚类任务，为论文样本标记标签；对于链接预测任务，为一对论文样本的引用关系标记标签
    根据指定的特征类型，返回混合数据集

    Args:
        paper_dict (dict): 论文字典
        dataset (str): 数据集名称，cit-HepTh或cit-HepPh（注意大小写）
        task (int): 机器学习任务类型，0表示节点分类和聚类，1表示链接预测. Defaults to 0.
        feature_type (int): 特征类型，0表示文本特征，1表示网络结构特征，2表示混合特征

    Returns:
        list: 混合数据集，每个元素由单个论文样本或一对论文样本的特征向量和标签元组构成
    """
    datasets = list()
    paper_set = set()
    edge_set = set()

    # 读取论文引用数据集的网络结构文件（包含节点和边信息）
    with open("./dataset/{}.txt".format(dataset), 'r') as paper_citataion_network_txt:
        paper_citation_network = paper_citataion_network_txt.readlines()
        paper_citation_network = paper_citation_network[4: ]
    
    # 遍历数据集网络结构的边，记录节点信息，根据任务类型，累计论文被引用量或记录论文引用有向边信息
    for line in paper_citation_network:
        line = line[: -1]
        from_node_id, to_node_id = line.split('\t')
        from_node_id = from_node_id.zfill(7)
        to_node_id = to_node_id.zfill(7)

        # 判断当前两个论文节点是否存在对应的元信息
        if from_node_id in paper_dict and to_node_id in paper_dict:
            paper_set.add(from_node_id)
            paper_set.add(to_node_id)

            if task == 0:
                to_paper = paper_dict[to_node_id]
                to_paper.num_citations += 1
            else:
                edge_set.add((from_node_id, to_node_id))
    
    # 构建混合数据集
    if task == 0:
        for paper_id in paper_set:
            paper = paper_dict[paper_id]
            paper.set_label()
            if feature_type == 0:
                datasets.append((paper.text_feature, paper.label))
            elif feature_type == 1:
                datasets.append((paper.network_feature, paper.label))
            else:
                paper.mix_feature = np.concatenate(paper.text_feature, paper.network_feature)
                datasets.append((paper.mix_feature, paper.label))
    else:
        paper_list = list(paper_set)
        from_node_id_list = random.sample(paper_list, 100)
        to_node_id_list = random.sample(paper_list, 100)
        edge_list = random.sample(list(edge_set), 10000)

        paper_pair_list = list()

        for from_node_id in from_node_id_list:
            for to_node_id in to_node_id_list:
                if (from_node_id, to_node_id) in edge_set:
                    paper_pair_list.append((from_node_id, to_node_id, 1))
                else:
                    paper_pair_list.append((from_node_id, to_node_id, 0))
        
        for i in range(len(edge_list)):
            paper_pair_list.append((edge_list[i][0], edge_list[i][1], 1))
        
        for i in range(len(paper_pair_list)):
            from_paper = paper_dict[paper_pair_list[i][0]]
            to_paper = paper_dict[paper_pair_list[i][1]]
            label = paper_pair_list[i][2]

            if feature_type == 0:
                datasets.append((from_paper.text_feature, to_paper.text_feature, label))
            elif feature_type == 1:
                datasets.append((from_paper.network_feature, to_paper.network_feature, label))
            else:
                from_paper.mix_feature = np.concatenate(from_paper.text_feature, from_paper.network_feature)
                to_paper.mix_feature = np.concatenate(to_paper.text_feature, to_paper.network_feature)
                datasets.append((from_paper.mix_feature, to_paper.mix_feature, label))
    
    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--dataset", type=str, default="cit-HepTh", help="")
    parser.add_argument("-t", "--task", type=int, default=0, help="")
    parser.add_argument("-f", "--feature_type", type=int, default=0, help="")
    parser.add_argument("-s", "--shuffle", type=bool, default=True, help="")
    parser.add_argument("-p", "--proportion", type=tuple, default=(0.7, 0.2), help="")

    args = parser.parse_args()

    training_set, validation_set, test_set = get_datasets(dataset=args.dataset, task=args.task, feature_type=args.feature_type, 
                                                          shuffle=args.shuffle, proportion=args.proportion)
