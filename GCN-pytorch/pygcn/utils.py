import numpy as np
import scipy.sparse as sp
import torch
import pickle as pkl
import networkx as nx


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))#每一引文二值特征向量求和，构成2708*1矩阵
    r_inv = np.power(rowsum, -1).flatten()#求-1次幂，并转为2708维向量
    r_inv[np.isinf(r_inv)] = 0.#趋于无穷置0
    r_mat_inv = sp.diags(r_inv)#创建对角阵，r_inv元素都在对角线上，维度2708*2708
    mx = r_mat_inv.dot(mx)#标准化所有引文二值特征向量（0,1）区间
    return mx
    #return r_mat_inv.dot(r_mat_inv.dot(mx)).tocoo()

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) #(2708, 1)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() #(1, 2708)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    # 这里对邻接矩阵两个维度都采取归一化
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() #因为adj已是对称，所以不需再加一次transpose()

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def to_tuple(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32) #输出是coo_matrix格式稀疏矩阵，(2708,2708)
    #sparse_mx.row是稀疏矩阵非零项的横(x)坐标，type为np.array，sparse_mx.col同理，(13264,)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))#两组坐标数组纵向拼接，(2,13264)，并转为tensor
    values = torch.from_numpy(sparse_mx.data)#sparse_mx中的数值，转为tensor，(,13264)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)#转为稀疏tensor，(2708,2708)

def load_data(dataset_str="citeseer"):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1])) # lil_matrix非零元素构建稀疏矩阵
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
    citeseer_test_feature_mask = test_idx_range.tolist() #选取citeseer模型结果中的有效test node feature

    # 特征矩阵
    features = sp.vstack((allx, tx)).tolil() # 垂直合并 train, test数据集特征向量
    features[test_idx_reorder, :] = features[test_idx_range, :] # 恢复测试集顺序
    features = normalize_features(features) # 归一化
    features = torch.FloatTensor(np.array(features.todense()))
    # 邻接矩阵
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)) # 已对称
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = to_tuple(adj)
    # 标签矩阵
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = torch.LongTensor(np.where(labels)[1])  # 维度(2708,)，输出的是引文标签向量1的索引，即值域[0,6]
    # 划分test/train/val
    idx_test = torch.LongTensor(test_idx_range.tolist()) #[1708, 2707]
    idx_train = torch.LongTensor(range(len(y))) #[0,140)
    idx_val = torch.LongTensor(range(len(y), len(y)+500)) #[140,640)

    return adj, features, labels, idx_train, idx_val, idx_test, citeseer_test_feature_mask


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)#行最大值索引
    correct = preds.eq(labels).double()#预测和真实标签向量比较对应位，相同置1，不同置0
    correct = correct.sum()
    return correct / len(labels)

if __name__=='__main__':
    data = load_data('citeseer')



