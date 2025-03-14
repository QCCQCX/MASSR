import os
import numpy as np
from time import time
import pickle
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from utility.parser import parse_args
from utility.norm import build_sim, build_knn_normalized_graph
args = parse_args()

class MASSR(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats, text_feats):

        super().__init__() # 调用父类的构造函数初始化模型
        self.n_users = n_users # 用户数量
        self.n_items = n_items # 物品（项目）数量
        self.embedding_dim = embedding_dim # 嵌入维度
        self.weight_size = weight_size # 权重大小（一个列表）
        self.n_ui_layers = len(self.weight_size) # 权重大小列表的长度
        self.weight_size = [self.embedding_dim] + self.weight_size # 将嵌入维度添加到权重大小列表的开头

        # 定义图像和文本特征的线性转换层
        self.image_trans = nn.Linear(image_feats.shape[1], args.embed_size) # 图像特征线性转换层
        self.text_trans = nn.Linear(text_feats.shape[1], args.embed_size) # 文本特征线性转换层
        nn.init.xavier_uniform_(self.image_trans.weight) # 使用Xavier初始化方法初始化图像线性转换层的权重
        nn.init.xavier_uniform_(self.text_trans.weight) # 使用Xavier初始化方法初始化文本线性转换层的权重
        self.encoder = nn.ModuleDict() # 创建一个模块字典用于存储编码器模块
        self.encoder['image_encoder'] = self.image_trans # 将图像编码器存储在模块字典中
        self.encoder['text_encoder'] = self.text_trans # 将文本编码器存储在模块字典中

        # 定义共享的线性转换层
        self.common_trans = nn.Linear(args.embed_size, args.embed_size) # 共享的线性转换层
        nn.init.xavier_uniform_(self.common_trans.weight) # 使用Xavier初始化方法初始化共享线性转换层的权重
        self.align = nn.ModuleDict() # 创建一个模块字典用于存储对齐模块（对齐模块：确保不同类型特征具有一致的表示（统一多模态））
        self.align['common_trans'] = self.common_trans # 将共享的线性转换层存储在模块字典中

        self.user_id_embedding = nn.Embedding(n_users, self.embedding_dim) # 用户ID嵌入层
        self.item_id_embedding = nn.Embedding(n_items, self.embedding_dim) # 物品ID嵌入层

        nn.init.xavier_uniform_(self.user_id_embedding.weight) # 使用Xavier初始化方法初始化用户ID嵌入层的权重
        nn.init.xavier_uniform_(self.item_id_embedding.weight) # 使用Xavier初始化方法初始化物品ID嵌入层的权重
        self.image_feats = torch.tensor(image_feats).float().cuda() # 将图像特征数据转换为PyTorch张量，并移动到GPU上
        self.text_feats = torch.tensor(text_feats).float().cuda() # 将文本特征数据转换为PyTorch张量，并移动到GPU上
        self.image_embedding = nn.Embedding.from_pretrained(torch.Tensor(image_feats), freeze=False) # 创建图像特征的嵌入层
        self.text_embedding = nn.Embedding.from_pretrained(torch.Tensor(text_feats), freeze=False) # 创建文本特征的嵌入层

        self.softmax = nn.Softmax(dim=-1) # Softmax激活函数
        self.act = nn.Sigmoid()   # Sigmoid激活函数
        self.sigmoid = nn.Sigmoid() # Sigmoid激活函数
        self.dropout = nn.Dropout(p=args.drop_rate) # 丢弃率为给定的 dropout 率（可能为超参数）
        self.batch_norm = nn.BatchNorm1d(args.embed_size) # 批量归一化层
        self.tau = 0.5 # 温度参数（可能为超参数）

        initializer = nn.init.xavier_uniform_ # 初始化器函数
        self.weight_dict = nn.ParameterDict({ # 定义参数字典用于存储各种权重参数
            'w_q': nn.Parameter(initializer(torch.empty([args.embed_size, args.embed_size]))), # 查询权重参数
            'w_k': nn.Parameter(initializer(torch.empty([args.embed_size, args.embed_size]))), # 键权重参数
            'w_v': nn.Parameter(initializer(torch.empty([args.embed_size, args.embed_size]))), # 值权重参数
            'w_self_attention_item': nn.Parameter(initializer(torch.empty([args.embed_size, args.embed_size]))), # 物品自注意力权重参数
            'w_self_attention_user': nn.Parameter(initializer(torch.empty([args.embed_size, args.embed_size]))), # 用户自注意力权重参数
            'w_self_attention_cat': nn.Parameter(initializer(torch.empty([args.head_num*args.embed_size, args.embed_size]))), # 合并注意力权重参数
        })
        self.embedding_dict = {'user':{}, 'item':{}} # 创建嵌入字典用于存储用户和物品的嵌入信息

    # 定义矩阵乘法函数mm，输入x和y是两个矩阵
    def mm(self, x, y):
        # 如果args.sparse为True，表示使用稀疏矩阵乘法torch.sparse.mm
        if args.sparse:
            return torch.sparse.mm(x, y)
        # 如果args.sparse为False，表示使用普通矩阵乘法torch.mm
        else:
            return torch.mm(x, y)

    # 定义计算相似度函数sim，输入z1和z2是两个向量(update4)
    def sim(self, z1, z2):
        # 对输入向量进行L2归一化
        z1 = F.normalize(z1)

        z2 = F.normalize(z2)

        # 计算余弦相似度
        cosine_sim = torch.mm(z1, z2.t())

        # 应用非线性变换（例如，ReLU）
        similarity = F.relu(cosine_sim)

        # 如果需要，可以进一步添加其他复杂性，如加权、归一化等

        return similarity

    # 定义批次三元组损失函数batched_triplet_loss，输入z1和z2是两个特征矩阵（学习样本之间的相似性）
    def batched_triplet_loss(self, z1, z2, batch_size=4096):
        # 获取z1所在的设备（GPU或CPU）
        device = z1.device
        # 获取节点数量（张量z1的第一维度的大小）
        num_nodes = z1.size(0)
        # 计算批次数量（// 截断小数（整数）做除法）
        num_batches = (num_nodes - 1) // batch_size + 1
        # 定义一个指数函数，用于计算相似度的权重
        f = lambda x: torch.exp(x / self.tau)
        # 创建一个索引张量，用于迭代处理不同批次的节点
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            # 创建一个掩码，用于选择当前批次的节点
            mask = indices[i * batch_size:(i + 1) * batch_size]
            # 计算自反相似度
            refl_sim = f(self.sim(z1[mask], z1))
            # 计算节点之间的相似度
            between_sim = f(self.sim(z1[mask], z2))

            # 计算损失，并添加到损失列表中
            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        # 将损失列表中的损失张量连接成一个向量
        loss_vec = torch.cat(losses)
        # 返回损失向量的均值作为最终的损失值
        return loss_vec.mean()

    # 定义CSR矩阵的归一化函数，输入csr_mat是CSR格式的矩阵
    def csr_norm(self, csr_mat, mean_flag=False):
        # 计算每行的和
        rowsum = np.array(csr_mat.sum(1))
        # 对每行的和进行平方根倒数运算，加上一个小的常数避免除零错误
        rowsum = np.power(rowsum+1e-8, -0.5).flatten()
        # 处理无穷大的情况，将其设为0
        rowsum[np.isinf(rowsum)] = 0.
        # 对角矩阵（行和） 用于按行归一化
        rowsum_diag = sp.diags(rowsum)

        # 计算每列的和
        colsum = np.array(csr_mat.sum(0))
        # 对每列的和进行平方根倒数运算，加上一个小的常数避免除零错误
        colsum = np.power(colsum+1e-8, -0.5).flatten()
        # 处理无穷大的情况，将其设为0
        colsum[np.isinf(colsum)] = 0.
        # 对角矩阵（列和） 用于按列归一化
        colsum_diag = sp.diags(colsum)

        # 根据输入参数判断进行 按行归一化还是按列归一化
        if mean_flag == False:
            return rowsum_diag*csr_mat*colsum_diag
        else:
            return rowsum_diag*csr_mat

    # 定义将稀疏矩阵转换为稀疏张量的函数
    def matrix_to_tensor(self, cur_matrix):
        # 如果输入的矩阵不是COO格式，则转换为COO格式
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()  #
        # 从COO格式矩阵中提取行和列的索引，构建张量
        indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  #
        # 从COO格式矩阵中提取数据
        values = torch.from_numpy(cur_matrix.data)  #
        # 获取矩阵的形状
        shape = torch.Size(cur_matrix.shape)

        # 返回稀疏张量，转换为GPU上的浮点型张量
        return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()  #

        # 类似字典类型的对象 转换成 张量
    def para_dict_to_tenser(self, para_dict):  
        """
        :param para_dict: nn.ParameterDict()
        :return: tensor
        """
        tensors = []
        # 创建一个空列表tensors，用于存储从para_dict中提取的参数张量。
        for beh in para_dict.keys():
            tensors.append(para_dict[beh])
        # 包含多个张量的列表 叠加为 一个张量 （dim 为叠加维度）
        tensors = torch.stack(tensors, dim=0)

        return tensors

    # BERT(update5)
    # BERT
    def BERT_attention(self, trans_w, embedding_t_1, embedding_t):

        # 获取查询向量q和键值对k
        q = self.para_dict_to_tenser(embedding_t)

        v = k = self.para_dict_to_tenser(embedding_t_1)
        # beh 表示批次大小，N 表示嵌入向量的维度，d_h 表示每个注意力头的维度
        beh, N, d_h = q.shape[0], q.shape[1], args.embed_size / args.head_num

        # 使用线性变换计算Q、K、V
        Q = torch.matmul(q, trans_w['w_q'])


        K = torch.matmul(k, trans_w['w_k'])
        V = v

        # 重塑Q和K，以便进行BERT自注意力操作
        Q = Q.reshape(beh, N, args.head_num, int(d_h)).permute(2, 0, 1, 3)
        K = K.reshape(beh, N, args.head_num, int(d_h)).permute(2, 0, 1, 3)

        # 为Q、K、V添加维度，以便计算注意力分数
        Q = torch.unsqueeze(Q, 2)
        K = torch.unsqueeze(K, 1)
        V = torch.unsqueeze(V, 1)

        # 计算注意力分数
        att = torch.mul(Q, K) / torch.sqrt(torch.tensor(d_h))
        att = torch.sum(att, dim=-1)
        att = torch.unsqueeze(att, dim=-1)
        att = F.softmax(att, dim=2)

        # 使用注意力分数对V进行加权平均
        Z = torch.mul(att, V)
        Z = torch.sum(Z, dim=2)

        # 将BERT自注意力的结果拼接起来
        Z_list = [value for value in Z]
        Z = torch.cat(Z_list, -1)
        Z = torch.matmul(Z, self.weight_dict['w_self_attention_cat'])

        # 标准化张量 Z，使其具有零均值和单位方差
        mean = Z.mean(dim=2, keepdim=True)
        std = Z.std(dim=2, keepdim=True)
        Z = (Z - mean) / (std + 1e-8)  # 添加一个小的常数以防止除零错误

        # 添加残差连接
        Z += self.para_dict_to_tenser(embedding_t)

        # 进行归一化
        Z = args.model_cat_rate * F.normalize(Z, p=2, dim=2)

        return Z, att.detach()

    def forward(self, ui_graph, iu_graph, image_ui_graph, image_iu_graph, text_ui_graph, text_iu_graph):

        # 通过线性变换层将图像和文本特征转换为嵌入向量
        image_feats = image_item_feats = self.dropout(self.image_trans(self.image_feats))
        text_feats = text_item_feats = self.dropout(self.text_trans(self.text_feats))

        for i in range(args.layers):
            # 图像特征的传播
            image_user_feats = self.mm(ui_graph, image_feats)
            image_item_feats = self.mm(iu_graph, image_user_feats)

            image_user_id = self.mm(image_ui_graph, self.item_id_embedding.weight)
            image_item_id = self.mm(image_iu_graph, self.user_id_embedding.weight)

            # 文本特征的传播
            text_user_feats = self.mm(ui_graph, text_feats)
            text_item_feats = self.mm(iu_graph, text_user_feats)

            text_user_id = self.mm(text_ui_graph, self.item_id_embedding.weight)
            text_item_id = self.mm(text_iu_graph, self.user_id_embedding.weight)

        # 更新嵌入字典
        self.embedding_dict['user']['image'] = image_user_id
        self.embedding_dict['user']['text'] = text_user_id

        self.embedding_dict['item']['image'] = image_item_id
        self.embedding_dict['item']['text'] = text_item_id

        # 使用BERT得到用户和物品的嵌入
        user_z, _ = self.BERT_attention(self.weight_dict, self.embedding_dict['user'], self.embedding_dict['user']) # e-mu
        item_z, _ = self.BERT_attention(self.weight_dict, self.embedding_dict['item'], self.embedding_dict['item']) # e-mi

        # 对BERT自注意力的结果进行平均，得到最终用户和物品的嵌入
        user_emb = user_z.mean(0) # e-u
        item_emb = item_z.mean(0) # e-i

        # 更新用户和物品的嵌入（多模态高阶连通）
        u_g_embeddings = self.user_id_embedding.weight + args.id_cat_rate * F.normalize(user_emb, p=2, dim=1) # E^u0
        i_g_embeddings = self.item_id_embedding.weight + args.id_cat_rate * F.normalize(item_emb, p=2, dim=1) # E^i0

        # 在多层传播后对用户和物品的嵌入进行进一步的操作
        user_emb_list = [u_g_embeddings]
        item_emb_list = [i_g_embeddings]
        for i in range(self.n_ui_layers):    
            if i == (self.n_ui_layers-1):
                u_g_embeddings = self.softmax(torch.mm(ui_graph, i_g_embeddings))
                i_g_embeddings = self.softmax(torch.mm(iu_graph, u_g_embeddings))
            else:
                u_g_embeddings = torch.mm(ui_graph, i_g_embeddings)
                i_g_embeddings = torch.mm(iu_graph, u_g_embeddings)

            user_emb_list.append(u_g_embeddings)
            item_emb_list.append(i_g_embeddings)

        # 对多层传播的结果进行平均，得到最终用户和物品的嵌入 （E^u）
        u_g_embeddings = torch.mean(torch.stack(user_emb_list), dim=0)
        i_g_embeddings = torch.mean(torch.stack(item_emb_list), dim=0)

        # 将图像和文本特征嵌入添加到用户和物品嵌入中
        u_g_embeddings = u_g_embeddings + args.model_cat_rate * F.normalize(image_user_feats, p=2, dim=1) + args.model_cat_rate * F.normalize(text_user_feats, p=2, dim=1)
        i_g_embeddings = i_g_embeddings + args.model_cat_rate * F.normalize(image_item_feats, p=2, dim=1) + args.model_cat_rate * F.normalize(text_item_feats, p=2, dim=1)

        # 返回最终的用户和物品嵌入，以及其他中间结果
        return u_g_embeddings, i_g_embeddings, image_item_feats, text_item_feats, image_user_feats, text_user_feats, u_g_embeddings, i_g_embeddings, image_user_id, text_user_id, image_item_id, text_item_id


# (update1)
class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        # 定义一个神经网络模型，使用nn.Sequential包装了一系列的层
        self.net = nn.Sequential(
            # 第一层：全连接层，输入维度为dim，输出维度为dim/4
            nn.Linear(dim, int(dim / 4)),
            # 激活函数：ReLU
            nn.ReLU(),
            # L2正则化
            nn.Dropout(args.G_drop1),

            # 新的隐藏层
            nn.Linear(int(dim / 4), int(dim / 8)),
            # 激活函数：LeakyReLU
            nn.LeakyReLU(True),
            # Dropout层
            nn.Dropout(args.G_drop2),

            # 新的卷积层1
            nn.Conv1d(2048, 1024, kernel_size=3, stride=1, padding=1),  # 输入通道为2048，输出通道为1024，3x3卷积核
            nn.ReLU(),

            # 新的卷积层2
            nn.Conv1d(1024, 512, kernel_size=3, stride=1, padding=1),  # 输入通道为1024，输出通道为512，3x3卷积核
            nn.ReLU(),

            # 第三层：全连接层，输入维度为512 * dim / 8，输出维度为1
            nn.Linear(int(dim / 8), 1),
            # Sigmoid激活函数
            nn.Sigmoid()
        )

    # 定义前向传播函数，该函数接收输入x并计算模型的输出
    def forward(self, x):
        # 将输入x传递给模型中的神经网络模型self.net，并将结果乘以100
        output = 100*self.net(x.float())
        # 将输出结果的形状调整为一维的张量
        return output.view(-1)

