import argparse
import os
import math
import random

from sklearn.feature_extraction.text import TfidfVectorizer


class Paper:
    def __init__(self, id, abstract):
        self.id = id
        self.abstract = abstract
        self.feature_vector = None
        self.num_citations = 0
        self.is_used = False


    def set_label(self):
        if self.num_citations == 0:
            self.label = 0
        else:
            self.label = round(math.log10(self.num_citations)) + 1


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
    # 特征提取
    paper_dict = extract_features()
    # 获取混合数据集
    datasets = get_paper_citation_network(paper_dict, dataset)

    # 打乱数据集顺序
    if shuffle:
        random.shuffle(datasets)
    
    # 划分训练集、验证集和测试集
    training_set = datasets[: int(proportion[0] * len(datasets))]
    validation_set = datasets[int(proportion[0] * len(datasets)): int((proportion[0] + proportion[1]) * len(datasets))]
    test_set = datasets[int((proportion[0] + proportion[1]) * len(datasets)): ]

    return training_set, validation_set, test_set


def extract_features():
    """TF-IDF文本特征提取

    Returns:
        dict: 论文字典
    """
    # 获取全部论文的元信息
    paper_dict = get_all_meta_info()
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

    # 设置每个论文样本的特征向量
    for i in range(len(id_list)):
        paper = paper_dict[id_list[i]]
        paper.feature_vector = feature_matrix[i]
    
    return paper_dict


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


def get_paper_citation_network(paper_dict, dataset):
    """获取论文引用网络信息（真实节点和边）

    Args:
        paper_dict (dict): 论文字典
        dataset (str): 数据集名称，cit-HepTh或cit-HepPh（注意大小写）

    Returns:
        list: 混合数据集，每个元素由论文样本的特征向量和标签元组构成
    """
    datasets = list()

    # 读取论文引用数据集的网络结构文件（包含节点和边信息）
    with open("./dataset/{}.txt".format(dataset), 'r') as paper_citataion_network_txt:
        paper_citation_network = paper_citataion_network_txt.readlines()
        paper_citation_network = paper_citation_network[4: ]

    # 累计论文被引用量值
    for line in paper_citation_network:
        line = line[: -1]
        from_node_id, to_node_id = line.split('\t')
        from_node_id = from_node_id.zfill(7)
        to_node_id = to_node_id.zfill(7)

        # 判断当前两个论文节点是否存在对应的元信息
        if from_node_id in paper_dict and to_node_id in paper_dict:
            from_paper = paper_dict[from_node_id]
            from_paper.is_used = True

            to_paper = paper_dict[to_node_id]
            to_paper.num_citations += 1
            to_paper.is_used = True
    
    # 构建混合数据集
    for key in paper_dict.keys():
        paper = paper_dict[key]
        if paper.is_used:
            paper.set_label()
            datasets.append((paper.feature_vector, paper.label))

    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--dataset", type=str, default="cit-HepTh", help="")
    parser.add_argument("-s", "--shuffle", type=bool, default=True, help="")
    parser.add_argument("-p", "--proportion", type=tuple, default=(0.7, 0.2), help="")

    args = parser.parse_args()

    training_set, validation_set, test_set = get_datasets(dataset=args.dataset, shuffle=args.shuffle, proportion=args.proportion)
