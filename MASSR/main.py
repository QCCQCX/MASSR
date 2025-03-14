from datetime import datetime
import math
import os
import random
import sys
from time import time
from tqdm import tqdm
import dgl
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import visdom

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.sparse as sparse
from torch import autograd

import copy

from utility.parser import parse_args
from Models import MASSR, Discriminator
from utility.batch_test import *
from utility.logging import Logger
from utility.norm import build_sim, build_knn_normalized_graph
from torch.utils.tensorboard import SummaryWriter


args = parse_args()
# 手动释放内存
class Trainer(object):
    def __init__(self, data_config):
        # 初始化训练器
        # 创建任务名称，记录日志
        self.task_name = "%s_%s_%s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), args.dataset, args.cf_model,)
        self.logger = Logger(filename=self.task_name, is_debug=args.debug)
        self.logger.logging("PID: %d" % os.getpid())
        self.logger.logging(str(args))

        # 从参数中解析配置信息
        self.mess_dropout = eval(args.mess_dropout)
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.weight_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
 
        # 从数据配置中加载图像和文本特征
        self.image_feats = np.load(args.data_path + '{}/image_feat.npy'.format(args.dataset))
        self.text_feats = np.load(args.data_path + '{}/text_feat.npy'.format(args.dataset))
        self.image_feat_dim = self.image_feats.shape[-1]
        self.text_feat_dim = self.text_feats.shape[-1]

        # 从数据配置中加载用户-物品交互图
        self.ui_graph = self.ui_graph_raw = pickle.load(open(args.data_path + args.dataset + '/train_mat','rb'))
        self.image_ui_graph_tmp = self.text_ui_graph_tmp = torch.tensor(self.ui_graph_raw.todense()).cuda()
        self.image_iu_graph_tmp = self.text_iu_graph_tmp = torch.tensor(self.ui_graph_raw.T.todense()).cuda()

        # 初始化图索引
        self.image_ui_index = {'x':[], 'y':[]}
        self.text_ui_index = {'x':[], 'y':[]}

        # 获取用户和物品的数量
        self.n_users = self.ui_graph.shape[0]
        self.n_items = self.ui_graph.shape[1]

        # 计算物品-用户图
        self.iu_graph = self.ui_graph.T
        self.ui_graph = self.matrix_to_tensor(self.csr_norm(self.ui_graph, mean_flag=True))
        self.iu_graph = self.matrix_to_tensor(self.csr_norm(self.iu_graph, mean_flag=True))
        self.image_ui_graph = self.text_ui_graph = self.ui_graph
        self.image_iu_graph = self.text_iu_graph = self.iu_graph

        # 初始化模型
        self.model = MASSR(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.mess_dropout, self.image_feats, self.text_feats)
        self.model = self.model.cuda()

        # 初始化判别器
        self.D = Discriminator(self.n_items).cuda()
        self.D.apply(self.weights_init)

        # 初始化判别器的优化器
        self.optim_D = optim.Adam(self.D.parameters(), lr=args.D_lr, betas=(0.5, 0.9))

        # 初始化总体优化器（用于训练模型）
        self.optimizer_D = optim.AdamW(
        [
            {'params':self.model.parameters()},      
        ]
            , lr=self.lr)  
        # 设置学习率调度器
        self.scheduler_D = self.set_lr_scheduler()

    # 根据 lambda 函数 fac 计算的倍增因子逐渐减小学习率，学习率会在每个 epoch 更新一次
    def set_lr_scheduler(self):
        # 定义一个 lambda 函数 'fac'，用于根据当前 epoch 计算学习率的倍增因子。
        fac = lambda epoch: 0.96 ** (epoch / 50)
        # 创建一个学习率调度器 'scheduler_D'，它使用 lambda 函数 'fac' 来更新学习率。
        scheduler_D = optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=fac)
        # 返回创建的学习率调度器。
        return scheduler_D

    # CSR 矩阵的标准化
    def csr_norm(self, csr_mat, mean_flag=False):
        # 计算 CSR 矩阵每行的总和并将其转换为 numpy 数组。
        rowsum = np.array(csr_mat.sum(1))
        # 对 rowsum 进行元素级操作，计算带有小 epsilon 的倒数平方根。
        rowsum = np.power(rowsum+1e-8, -0.5).flatten()
        # 将任何无穷大的值替换为 0。
        rowsum[np.isinf(rowsum)] = 0.
        # 从 rowsum 创建对角矩阵 'rowsum_diag'。
        rowsum_diag = sp.diags(rowsum)

        # 计算 CSR 矩阵每列的总和，并执行类似的操作。
        colsum = np.array(csr_mat.sum(0))
        colsum = np.power(colsum+1e-8, -0.5).flatten()
        colsum[np.isinf(colsum)] = 0.
        colsum_diag = sp.diags(colsum)

        # 如果 'mean_flag' 为 False，则返回经过完全标准化的 CSR 矩阵，包括行和列缩放。
        if mean_flag == False:
            return rowsum_diag*csr_mat*colsum_diag
        # 如果 'mean_flag' 为 True，则返回仅进行行缩放的标准化 CSR 矩阵。
        else:
            return rowsum_diag*csr_mat

    # 将 COO 格式的矩阵转换为 PyTorch 稀疏张量
    def matrix_to_tensor(self, cur_matrix):
        if type(cur_matrix) != sp.coo_matrix:
            # 如果 'cur_matrix' 不是 COO 格式，将其转换为 COO 格式。
            cur_matrix = cur_matrix.tocoo()
        # 从 COO 矩阵提取行和列索引以及数据。
        indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))
        values = torch.from_numpy(cur_matrix.data)  #
        # 使用提取的数据和形状创建一个 PyTorch 稀疏浮点张量。
        shape = torch.Size(cur_matrix.shape)

        return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()

    # 内积
    def innerProduct(self, u_pos, i_pos, u_neg, j_neg):
        # 计算正样本（'u_pos' 和 'i_pos'）和负样本（'u_neg' 和 'j_neg'）之间的内积。
        pred_i = torch.sum(torch.mul(u_pos,i_pos), dim=-1)
        pred_j = torch.sum(torch.mul(u_neg,j_neg), dim=-1)  
        return pred_i, pred_j

    def sampleTrainBatch_dgl(self, batIds, pos_id=None, g=None, g_neg=None, sample_num=None, sample_num_neg=None):

        # 从图 'g' 中为给定批次 'batIds' 采样邻居节点，使用指定的采样参数。
        sub_g = dgl.sampling.sample_neighbors(g.cpu(), {'user':batIds}, sample_num, edge_dir='out', replace=True)
        row, col = sub_g.edges()
        row = row.reshape(len(batIds), sample_num)
        col = col.reshape(len(batIds), sample_num)

        # 如果没有提供负图 'g_neg'，则返回采样的行和列。
        if g_neg==None:
            return row, col
        # 如果提供了 'g_neg'，则在相同的批次中为其采样邻居并返回行和列。
        else:
            sub_g_neg = dgl.sampling.sample_neighbors(g_neg, {'user':batIds}, sample_num_neg, edge_dir='out', replace=True)
            row_neg, col_neg = sub_g_neg.edges()
            row_neg = row_neg.reshape(len(batIds), sample_num_neg)
            col_neg = col_neg.reshape(len(batIds), sample_num_neg)
            return row, col, col_neg 

    def weights_init(self, m):
        # 对神经网络层 'm' 进行权重初始化，使用 Kaiming 正态分布初始化线性层的权重，并将偏差设置为零。
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)

    # 梯度惩罚(update2)
    def gradient_penalty(self, D, xr, xf): # D 是判别器模型，xr 是真实样本，xf 是生成器生成样本

        # 设置梯度惩罚的超参数（可能为超参数）
        LAMBDA = 0.3

        # 分离生成器的输出
        xf = xf.detach()
        xr = xr.detach()

        # 生成一个随机数alpha，并将其扩展成与xr相同的形状
        alpha = torch.rand(args.batch_size*2, 1).cuda()
        alpha = alpha.expand_as(xr)

        # 将 alpha 归一化为单位范数
        alpha_normalized = alpha / alpha.norm(dim=1, keepdim=True)

        # 计算球面线性插值
        interpolated_point = (alpha_normalized * xr) + ((1 - alpha_normalized) * xf)
        interpolated_point.requires_grad_()

        # 通过鉴别器计算插值点的分数
        disc_interpolates = D(interpolated_point)

        # 计算插值点的梯度
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolated_point,
                                grad_outputs=torch.ones_like(disc_interpolates),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

        # 计算梯度惩罚项
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

        return gp

    # 计算权重和，用于生成嵌入向量
    def weighted_sum(self, anchor, nei, co):

        # 计算anchor和neighbor的权重和
        ac = torch.multiply(anchor, co).sum(-1).sum(-1)
        nc = torch.multiply(nei, co).sum(-1).sum(-1)

        # 提取anchor和neighbor的嵌入向量的第一维
        an = (anchor.permute(1, 0, 2)[0])
        ne = (nei.permute(1, 0, 2)[0])

        # 权重加权
        an_w = an*(ac.unsqueeze(-1).repeat(1, args.embed_size))
        ne_w = ne*(nc.unsqueeze(-1).repeat(1, args.embed_size))

        # 计算结果
        res = (args.anchor_rate*an_w + (1-args.anchor_rate)*ne_w).reshape(-1, args.sample_num_ii, args.embed_size).sum(1)

        return res


    # 从相似度矩阵中采样top-k节点
    def sample_topk(self, u_sim, users, emb_type=None):
        # 选择top-k相似度和对应的节点ID
        topk_p, topk_id = torch.topk(u_sim, args.ad_topk*10, dim=-1)
        topk_data = topk_p.reshape(-1).cpu()
        topk_col = topk_id.reshape(-1).cpu().int()
        topk_row = torch.tensor(np.array(users)).unsqueeze(1).repeat(1, args.ad_topk*args.ad_topk_multi_num).reshape(-1).int()  #

        # 创建一个CSR格式的稀疏矩阵
        topk_csr = csr_matrix((topk_data.detach().numpy(), (topk_row.detach().numpy(), topk_col.detach().numpy())), shape=(self.n_users, self.n_items))
        topk_g = dgl.heterograph({('user','ui','item'):topk_csr.nonzero()})

        # 获取采样的节点ID
        _, topk_id = self.sampleTrainBatch_dgl(users, g=topk_g, sample_num=args.ad_topk, pos_id=None, g_neg=None, sample_num_neg=None)
        self.gene_fake[emb_type] = topk_id

        # 获取对应节点的相似度
        topk_id_u = torch.arange(len(users)).unsqueeze(1).repeat(1, args.ad_topk)
        topk_p = u_sim[topk_id_u, topk_id]
        return topk_p, topk_id

    # 计算自监督损失
    def ssl_loss_calculation(self, ssl_image_logit, ssl_text_logit, ssl_common_logit):
        # 创建用于自监督损失计算的标签
        ssl_label_1_s2 = torch.ones(1, self.n_items).cuda() # 创建标签，1 表示正样本
        ssl_label_0_s2 = torch.zeros(1, self.n_items).cuda() # 创建标签，0 表示负样本
        ssl_label_s2 = torch.cat((ssl_label_1_s2, ssl_label_0_s2), 1) # 将正负样本标签连接在一起

        # 计算图像和文本的自监督损失
        ssl_image_s2 = self.bce(ssl_image_logit, ssl_label_s2)
        ssl_text_s2 = self.bce(ssl_text_logit, ssl_label_s2)
        ssl_loss_s2 = ssl_image_s2 + ssl_text_s2 # 将图像和文本的自监督损失相加得到总的自监督损失

        ssl_label_1_c2 = torch.ones(1, self.n_items*2).cuda() # 创建标签，1 表示正样本
        ssl_label_0_c2 = torch.zeros(1, self.n_items*2).cuda() # 创建标签，0 表示负样本
        ssl_label_c2 = torch.cat((ssl_label_1_c2, ssl_label_0_c2), 1) # 将正负样本标签连接在一起

        ssl_result_c2 = self.bce(ssl_common_logit, ssl_label_c2) # 使用二进制交叉熵计算共享特征的自监督损失
        ssl_loss_c2 = ssl_result_c2 # 共享特征的自监督损失

        ssl_loss2 = args.ssl_s_rate*ssl_loss_s2 + args.ssl_c_rate*ssl_loss_c2  # 综合考虑图像、文本和共享特征的自监督损失

        # 返回总的自监督损失
        return ssl_loss2

    # 计算 z1 和 z2 之间的余弦相似度矩阵
    def sim(self, z1, z2):
        # 对输入张量 z1 进行 L2 归一化
        z1 = F.normalize(z1)
        # 对输入张量 z2 进行 L2 归一化
        z2 = F.normalize(z2)
        # z1 = z1/((z1**2).sum(-1) + 1e-8)
        # z2 = z2/((z2**2).sum(-1) + 1e-8)

        # 计算 z1 和 z2 之间的余弦相似度矩阵
        return torch.mm(z1, z2.t())

    # 三元组损失计算函数（update7）
    def batched_triplet_loss(self, z1, z2, batch_size=1024):

        # 获取输入张量 z1 的设备
        device = z1.device
        # 获取节点的数量
        num_nodes = z1.size(0)
        # 计算批次数量
        num_batches = (num_nodes - 1) // batch_size + 1
        # 定义一个间隔
        margin = 0.2

        # 创建索引张量
        indices = torch.arange(0, num_nodes).to(device)
        # 存储损失的列表
        losses = []

        for i in range(num_batches):
            # 获取当前批次的节点索引范围
            tmp_i = indices[i * batch_size:(i + 1) * batch_size]

        for j in range(num_batches):
            # 获取另一个批次的节点索引范围
            tmp_j = indices[j * batch_size:(j + 1) * batch_size]

            # 计算锚点到正样本的距离
            distance_positive = torch.norm(z1[tmp_i] - z2[tmp_i], p=2, dim=1)
            # 计算锚点到负样本的距离
            distance_negative_1 = torch.norm(z1[tmp_i] - z1[tmp_j], p=2, dim=1)
        distance_negative_2 = torch.norm(z1[tmp_i] - z2[tmp_j], p=2, dim=1)

        # 计算三元组损失，并将其添加到损失列表中
        loss_batch = torch.clamp(distance_positive - distance_negative_1 - distance_negative_2 + margin, min=0.0)


        losses.append(loss_batch)

        # 将损失列表中的损失张量连接成一个向量
        loss_vec = torch.cat(losses)
        # 返回损失向量的均值
        return loss_vec.mean()

    # 计算特征正则化损失(update3)
    def feat_reg_loss_calculation(self, g_item_image, g_item_text, g_user_image, g_user_text):
        # 计算特征正则化损失
        feat_reg = 1. / 2 * (g_item_image ** 2).sum() + 1. / 2 * (g_item_text ** 2).sum() \
                   + 1. / 2 * (g_user_image ** 2).sum() + 1. / 2 * (g_user_text ** 2).sum()
        # 将损失除以物品数量，以平均化
        feat_reg = feat_reg / self.n_items
        # 乘以特定的超参数（可能为超参数）
        feat_emb_loss = args.feat_reg_decay * feat_reg

        return feat_emb_loss

    # 计算生成器损失，包括真实样本损失和生成样本损失
    def fake_gene_loss_calculation(self, u_emb, i_emb, emb_type=None):
        if self.gene_u!=None:
            # 计算生成器损失，包括真实样本损失和生成样本损失
            gene_real_loss = (-F.logsigmoid((u_emb[self.gene_u]*i_emb[self.gene_real]).sum(-1)+1e-8)).mean()
            gene_fake_loss = (1-(-F.logsigmoid((u_emb[self.gene_u]*i_emb[self.gene_fake[emb_type]]).sum(-1)+1e-8))).mean()

            # 计算总损失
            gene_loss = gene_real_loss + gene_fake_loss
        else:
            gene_loss = 0

        return gene_loss

    # 计算奖励损失
    def reward_loss_calculation(self, users, re_u, re_i, topk_id, topk_p):
        # 创建一个张量 self.gene_u，用于存储用户索引的重复值
        self.gene_u = torch.tensor(np.array(users)).unsqueeze(1).repeat(1, args.ad_topk)
        # 从 re_u 中获取用户对应的奖励值
        reward_u = re_u[self.gene_u]
        # 从 re_i 中获取与 topk 推荐物品对应的奖励值
        reward_i = re_i[topk_id]
        # 计算用户和物品之间的奖励值
        reward_value = (reward_u*reward_i).sum(-1)

        # 计算奖励损失
        reward_loss = -(((topk_p*reward_value).sum(-1)).mean()+1e-8).log()
        
        return reward_loss

    # 计算用户之间的相似度
    def u_sim_calculation(self, users, user_final, item_final):
        # 获取用户的嵌入向量
        topk_u = user_final[users]
        # 将用户-物品交互矩阵（稀疏矩阵）转换为稠密的PyTorch张量
        u_ui = torch.tensor(self.ui_graph_raw[users].todense()).cuda()

        # 计算物品之间的相似度，这是一个批量操作，将物品划分为多个批次计算
        num_batches = (self.n_items - 1) // args.batch_size + 1
        indices = torch.arange(0, self.n_items).cuda()
        u_sim_list = []

        for i_b in range(num_batches):
            # 获取当前批次的物品索引
            index = indices[i_b * args.batch_size:(i_b + 1) * args.batch_size]
            # 计算用户嵌入与当前批次物品嵌入之间的相似度矩阵
            sim = torch.mm(topk_u, item_final[index].T)
            # 通过与用户-物品交互矩阵进行逐元素相乘，过滤掉已有交互的物品
            sim_gt = torch.multiply(sim, (1-u_ui[:, index]))

            # 将每个批次计算得到的相似度矩阵添加到列表中
            u_sim_list.append(sim_gt)

        # 将所有批次的相似度矩阵连接起来，并对结果进行归一化处理
        u_sim = F.normalize(torch.cat(u_sim_list, dim=-1), p=2, dim=1)
        return u_sim


    # 对模型进行测试评估 输入参数包括待测试的用户列表和一个标志
    def test(self, users_to_test, is_val):
        # 设置模型为评估模式（不进行梯度计算）
        self.model.eval()
        # 禁用梯度计算上下文管理器，以减少内存消耗
        with torch.no_grad():
            # 调用模型进行推断，获取用户和物品的嵌入向量
            ua_embeddings, ia_embeddings, *rest = self.model(self.ui_graph, self.iu_graph, self.image_ui_graph, self.image_iu_graph, self.text_ui_graph, self.text_iu_graph)
        # 调用test_torch函数进行评估，传递用户嵌入向量、物品嵌入向量、待测试的用户列表和是否为验证集标志
        result = test_torch(ua_embeddings, ia_embeddings, users_to_test, is_val)
        return result

    def train(self):

        # 获取当前时间并格式化成字符串，用于生成日志文件名
        now_time = datetime.now()
        run_time = datetime.strftime(now_time,'%Y_%m_%d__%H_%M_%S')

        # 初始化用于记录训练过程中的各项指标的列表
        training_time_list = []
        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
        line_var_loss, line_g_loss, line_d_loss, line_sn_loss, line_var_recall, line_var_precision, line_var_ndcg = [], [], [], [], [], [], []

        # 初始化早停计数器和标志
        stopping_step = 0
        should_stop = False
        cur_best_pre_0 = 0.

        # 可视化工具初始化
        tb_writer = SummaryWriter(log_dir="/home/ww/Code/work5/MICRO2Ours/tensorboard/")
        tensorboard_cnt = 0

        # 计算训练集的批次数量
        n_batch = data_generator.n_train // args.batch_size + 1

        # 初始化最佳召回率
        best_recall = 0

        # 开始训练循环，迭代指定的训练轮数
        for epoch in range(args.epoch):
            t1 = time()

            # 初始化损失变量
            loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
            triplet_loss = 0.
            n_batch = data_generator.n_train // args.batch_size + 1
            sample_time = 0.
            self.gene_u, self.gene_real, self.gene_fake = None, None, {}

            # 初始化字典变量
            self.topk_p_dict, self.topk_id_dict = {}, {}

            # 迭代处理每个批次
            for idx in tqdm(range(n_batch)):
                self.model.train()
                sample_t1 = time()
                # 从数据生成器中采样用户、正样本物品和负样本物品
                users, pos_items, neg_items = data_generator.sample()
                sample_time += time() - sample_t1       

                # 使用torch.no_grad()上下文管理器，禁用梯度计算，并获取模型的嵌入向量
                with torch.no_grad():
                    ua_embeddings, ia_embeddings, image_item_embeds, text_item_embeds, image_user_embeds, text_user_embeds \
                                    , _, _, _, _, _, _ \
                            = self.model(self.ui_graph, self.iu_graph, self.image_ui_graph, self.image_iu_graph, self.text_ui_graph, self.text_iu_graph)

                # 计算用户与物品之间的"相似性"，并将结果分别存储，生成器计算
                ui_u_sim_detach = self.u_sim_calculation(users, ua_embeddings, ia_embeddings).detach()
                image_u_sim_detach = self.u_sim_calculation(users, image_user_embeds, image_item_embeds).detach()
                text_u_sim_detach = self.u_sim_calculation(users, text_user_embeds, text_item_embeds).detach()

                # 将图像和文本方面的相似性连接起来，并通过神经网络模型"self.D"进行预测，然后计算预测结果的平均损失lossf （A^）
                inputf = torch.cat((image_u_sim_detach, text_u_sim_detach), dim=0)
                predf = (self.D(inputf))
                lossf = (predf.mean())

                # 计算用户 - 物品评分矩阵，应用softmax函数和归一化处理，并计算真实数据损失lossr（A~）
                u_ui = torch.tensor(self.ui_graph_raw[users].todense()).cuda()
                u_ui = F.softmax(u_ui - args.log_log_scale * torch.log(-torch.log(
                    torch.empty((u_ui.shape[0], u_ui.shape[1]), dtype=torch.float32)
                        .uniform_(0, 1).cuda() + 1e-8) + 1e-8) / args.real_data_tau, dim=1) #0.002
                u_ui += ui_u_sim_detach * args.ui_pre_scale
                u_ui = F.normalize(u_ui, dim=1)

                inputr = torch.cat((u_ui, u_ui), dim=0)
                predr = (self.D(inputr))
                lossr = - (predr.mean())

                # 计算梯度惩罚项gp（真与假的中间插值）
                gp = self.gradient_penalty(self.D, inputr, inputf.detach())

                # 计算总的判别器损失loss_D，并对判别器进行优化
                loss_D = lossr + lossf + args.gp_rate * gp

                # 对判别器的梯度进行清零，然后反向传播并更新判别器的参数！！！
                self.optim_D.zero_grad()

                loss_D.backward()

                self.optim_D.step()

                # 记录判别器损失到line_d_loss中
                line_d_loss.append(loss_D.detach().data)


                # 再次获取模型的嵌入向量，这次是为了生成负样本
                G_ua_embeddings, G_ia_embeddings, G_image_item_embeds, G_text_item_embeds, G_image_user_embeds, G_text_user_embeds \
                                , G_user_emb, _, G_image_user_id, G_text_user_id, _, _ \
                        = self.model(self.ui_graph, self.iu_graph, self.image_ui_graph, self.image_iu_graph, self.text_ui_graph, self.text_iu_graph)

                # 获取生成的用户、正样本物品和负样本物品的嵌入向量
                G_u_g_embeddings = G_ua_embeddings[users]
                G_pos_i_g_embeddings = G_ia_embeddings[pos_items]
                G_neg_i_g_embeddings = G_ia_embeddings[neg_items]

                # 计算生成样本的损失，包括MF损失、嵌入损失和正则化损失
                G_batch_mf_loss, G_batch_emb_loss, G_batch_reg_loss = self.bpr_loss(G_u_g_embeddings, G_pos_i_g_embeddings, G_neg_i_g_embeddings)

                G_image_u_sim = self.u_sim_calculation(users, G_image_user_embeds, G_image_item_embeds)
                G_text_u_sim = self.u_sim_calculation(users, G_text_user_embeds, G_text_item_embeds)
                # 计算生成的图像和文本方面的相似性
                # 分别存储在G_image_u_sim_detach和G_text_u_sim_detach中
                G_image_u_sim_detach = G_image_u_sim.detach()
                G_text_u_sim_detach = G_text_u_sim.detach()


                if idx%args.T==0 and idx!=0:
                    # 创建一个稀疏的图矩阵 self.image_ui_graph_tmp
                    # 它使用了之前记录的用户-物品交互索引 self.image_ui_index，并将它们的值都设为1。
                    self.image_ui_graph_tmp = csr_matrix((torch.ones(len(self.image_ui_index['x'])),(self.image_ui_index['x'], self.image_ui_index['y'])), shape=(self.n_users, self.n_items))
                    self.text_ui_graph_tmp = csr_matrix((torch.ones(len(self.text_ui_index['x'])),(self.text_ui_index['x'], self.text_ui_index['y'])), shape=(self.n_users, self.n_items))
                    self.image_iu_graph_tmp = self.image_ui_graph_tmp.T
                    self.text_iu_graph_tmp = self.text_ui_graph_tmp.T

                    # 将 self.image_ui_graph_tmp 转换为 PyTorch 稀疏张量
                    # 并将其归一化后存储在 self.image_ui_graph 中。
                    self.image_ui_graph = self.sparse_mx_to_torch_sparse_tensor(
                        self.csr_norm(self.image_ui_graph_tmp, mean_flag=True)
                        ).cuda() 
                    self.text_ui_graph = self.sparse_mx_to_torch_sparse_tensor(
                        self.csr_norm(self.text_ui_graph_tmp, mean_flag=True)
                        ).cuda()

                    # 对于用户-物品反向关系，也进行了相同的转换和归一化处理
                    # 结果存储在 self.image_iu_graph 中。
                    self.image_iu_graph = self.sparse_mx_to_torch_sparse_tensor(
                        self.csr_norm(self.image_iu_graph_tmp, mean_flag=True)
                        ).cuda()
                    self.text_iu_graph = self.sparse_mx_to_torch_sparse_tensor(
                        self.csr_norm(self.text_iu_graph_tmp, mean_flag=True)
                        ).cuda()
                    # 将 self.image_ui_index 和 self.text_ui_index 重新初始化为空字典
                    # 以准备记录下一个批次的用户-物品交互索引
                    self.image_ui_index = {'x':[], 'y':[]}
                    self.text_ui_index = {'x':[], 'y':[]}

                # 如果不是更新图结构的时刻，将生成的图信息添加到用户 - 物品交互图的索引中，以便后续更新图结构
                else:
                    _, image_ui_id = torch.topk(G_image_u_sim_detach, int(self.n_items*args.m_topk_rate), dim=-1)
                    self.image_ui_index['x'] += np.array(torch.tensor(users).repeat(1, int(self.n_items*args.m_topk_rate)).view(-1)).tolist()
                    self.image_ui_index['y'] += np.array(image_ui_id.cpu().view(-1)).tolist()
                    _, text_ui_id = torch.topk(G_text_u_sim_detach, int(self.n_items*args.m_topk_rate), dim=-1)
                    self.text_ui_index['x'] += np.array(torch.tensor(users).repeat(1, int(self.n_items*args.m_topk_rate)).view(-1)).tolist()
                    self.text_ui_index['y'] += np.array(text_ui_id.cpu().view(-1)).tolist()

                # 计算特征正则化损失
                feat_emb_loss = self.feat_reg_loss_calculation(G_image_item_embeds, G_text_item_embeds, G_image_user_embeds, G_text_user_embeds)

                # update6
                # 计算三元组损失，包括图像和文本方面的损失
                # 初始化三元组损失
                batch_triplet_loss = 0
                # 计算图像方面的三元组损失
                batch_triplet_loss1 = self.batched_triplet_loss(G_image_user_id[users],G_user_emb[users])
                # 计算文本方面的三元组损失
                batch_triplet_loss2 = self.batched_triplet_loss(G_text_user_id[users],G_user_emb[users])
                # 总三元组损失等于图像和文本三元组损失之和
                batch_triplet_loss = batch_triplet_loss1 + batch_triplet_loss2

                # 计算生成样本的总损失batch_loss，包括MF损失、嵌入三元组损失、损失、特征正则化损失和正则化损失
                G_inputf = torch.cat((G_image_u_sim, G_text_u_sim), dim=0)
                G_predf = (self.D(G_inputf))
                G_lossf = -(G_predf.mean())

                # update6
                batch_loss = G_batch_mf_loss + G_batch_emb_loss + G_batch_reg_loss + feat_emb_loss + args.sn_rate * batch_triplet_loss + args.G_rate * G_lossf  #feat_emb_loss
                # batch_loss = G_batch_mf_loss + G_batch_emb_loss + G_batch_reg_loss + feat_emb_loss + args.G_rate * G_lossf  #feat_emb_loss

                # 将总损失、生成器损失和三元组损失记录下来
                line_var_loss.append(batch_loss.detach().data)
                line_g_loss.append(G_lossf.detach().data)
                # update6
                line_sn_loss.append(batch_triplet_loss.detach().data)

                # 清空判别器优化器的梯度，为下一轮迭代做准备
                self.optimizer_D.zero_grad()

                # 反向传播总损失，更新判别器的参数
                batch_loss.backward(retain_graph=False)
                self.optimizer_D.step()

                # 更新总损失、MF损失、嵌入损失、正则化损失
                loss += float(batch_loss)
                mf_loss += float(G_batch_mf_loss)
                emb_loss += float(G_batch_emb_loss)
                reg_loss += float(G_batch_reg_loss)
                triplet_loss += float(batch_triplet_loss)

            # 本轮训练的所有批次结束！

            # 删除不再需要的嵌入向量，以释放内存
            del ua_embeddings, ia_embeddings, G_ua_embeddings, G_ia_embeddings, G_u_g_embeddings, G_neg_i_g_embeddings, G_pos_i_g_embeddings

            # 如果损失为NaN，记录错误信息并退出
            if math.isnan(loss) == True:
                self.logger.logging('ERROR: loss is nan. Loss is : {}'.format(loss))
                self.logger.logging('mf_loss is : {}'.format(mf_loss))
                self.logger.logging('emb_loss is : {}'.format(emb_loss))
                self.logger.logging('reg_loss is : {}'.format(reg_loss))
                self.logger.logging('feat_emb_loss is : {}'.format(float(feat_emb_loss)))
                self.logger.logging('G_lossf is : {}'.format(float(G_lossf)))
                sys.exit()

            # 在每个周期结束时，如果不需要详细输出，则记录性能信息
            if (epoch + 1) % args.verbose != 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f  + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss, reg_loss, triplet_loss)
                training_time_list.append(time() - t1)
                self.logger.logging(perf_str)

            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            users_to_val = list(data_generator.val_set.keys())

            # 在验证集上进行性能测试
            ret = self.test(users_to_val, is_val=True)
            training_time_list.append(t2 - t1)

            t3 = time()

            # 记录性能指标和损失
            loss_loger.append(loss)
            rec_loger.append(ret['recall'].data)
            pre_loger.append(ret['precision'].data)
            ndcg_loger.append(ret['ndcg'].data)

            # 记录特定指标（recall、precision、ndcg）
            line_var_recall.append(ret['recall'][1])
            line_var_precision.append(ret['precision'][1])
            line_var_ndcg.append(ret['ndcg'][1])

            # 使用可视化工具记录性能指标
            tags = ["recall", "precision", "ndcg"]
            tb_writer.add_scalar(tags[0], ret['recall'][1], epoch)
            tb_writer.add_scalar(tags[1], ret['precision'][1], epoch)
            tb_writer.add_scalar(tags[2], ret['ndcg'][1], epoch)

            # 如果需要详细输出，则记录性能信息
            if args.verbose > 0:
                perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f, %.5f, %.5f], ' \
                           'precision=[%.5f, %.5f, %.5f, %.5f], ndcg=[%.5f, %.5f, %.5f, %.5f]' % \
                           (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, ret['recall'][0], ret['recall'][1], ret['recall'][2],
                            ret['recall'][-1],
                            ret['precision'][0], ret['precision'][1], ret['precision'][2], ret['precision'][-1],
                            ret['ndcg'][0], ret['ndcg'][1], ret['ndcg'][2], ret['ndcg'][-1])
                self.logger.logging(perf_str)

            # 如果在验证集上的召回率超过之前最佳的召回率，则进行测试，并记录测试结果
            if ret['recall'][1] > best_recall:
                best_recall = ret['recall'][1]
                test_ret = self.test(users_to_test, is_val=False)
                self.logger.logging("Test_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[1], test_ret['recall'][1], test_ret['precision'][1], test_ret['ndcg'][1]))
                stopping_step = 0

            # 如果连续多次召回率没有提高，则提前停止训练
            elif stopping_step < args.early_stopping_patience:
                stopping_step += 1
                self.logger.logging('#####Early stopping steps: %d #####' % stopping_step)
            else:
                self.logger.logging('#####Early stop! #####')
                break

        # 所有轮次训练结束！

        # 输出指标结果
        self.logger.logging(str(test_ret))

        # 返回最佳召回率和运行时间
        return best_recall, run_time


    def bpr_loss(self, users, pos_items, neg_items):
        # 计算正样本得分
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        # 计算负样本得分
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        # 计算正则化项
        regularizer = 1./2*(users**2).sum() + 1./2*(pos_items**2).sum() + 1./2*(neg_items**2).sum()
        regularizer = regularizer / self.batch_size

        # 计算最大化对数sigmoid损失
        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        # 嵌入损失和正则化损失都初始化为0
        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        """将稀疏的 scipy 矩阵转换为稀疏的 torch 稀疏张量。"""

        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

def set_seed(seed):
    # 设置随机数生成器的种子，用于确保实验的可重复性
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  

if __name__ == '__main__':
    # 设置可见的GPU设备
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # 设置随机种子，以确保实验的可重复性
    set_seed(args.seed)
    # 创建一个空字典config并设置两个键值对
    config = dict()
    # 存储用户数
    config['n_users'] = data_generator.n_users
    # 存储物品数
    config['n_items'] = data_generator.n_items

    # 创建一个名为trainer的Trainer类的实例，传入数据配置config
    trainer = Trainer(data_config=config)
    # 调用trainer的train方法来开始训练模型
    trainer.train()
