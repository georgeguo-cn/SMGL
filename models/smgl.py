# coding: utf-8

import os
import pdb
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss

class SMGL(GeneralRecommender):
    def __init__(self, config, dataset):
        super(SMGL, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.k = config['ori_mm_k']
        self.cf_model = config['cf_model']
        self.m_layer = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.mask_rate = config['mask_rate']
        self.knn_k = config['knn_k']
        self.tau = 0.2
        self.mm_weight = config['mm_weight']
        self.weight_size = config['weight_size']
        self.ssl_weight = config['ssl_weight']
        self.reg_weight = config['reg_weight']

        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.adj = self.scipy_matrix_to_sparse_tenser(self.interaction_matrix, torch.Size((self.n_users, self.n_items)))
        self.num_inters, self.norm_adj = self.get_norm_adj_mat()
        self.num_inters = torch.FloatTensor(1.0 / (self.num_inters + 1e-7)).to(self.device)

        self.item_adj, self.user_adj = None, None
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.item_graph_dict = np.load(os.path.join(dataset_path, config['item_graph_dict_file']), allow_pickle=True).item()
        self.item_adj = self.gen_homo_graph(self.item_graph_dict, self.n_items, 'softmax')
        
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.enc_mask_token = nn.Parameter(nn.init.xavier_uniform_(torch.zeros((1, self.feat_embed_dim))))
        # self.v_feat = None
        # self.t_feat = None
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)
            self.item_image_trs = nn.Parameter(nn.init.xavier_uniform_(torch.zeros((self.v_feat.shape[1], self.feat_embed_dim))))
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)
            self.item_text_trs = nn.Parameter(nn.init.xavier_uniform_(torch.zeros((self.t_feat.shape[1], self.feat_embed_dim))))
        
        self.reg_loss = EmbLoss()
    
    def scipy_matrix_to_sparse_tenser(self, matrix, shape):
        row = matrix.row
        col = matrix.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(matrix.data)
        return torch.sparse.FloatTensor(i, data, shape).to(self.device)
    
    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        return sumArr, self.scipy_matrix_to_sparse_tenser(L, torch.Size((self.n_nodes, self.n_nodes)))

    def pre_epoch_processing(self):
        return

    def gen_homo_graph(self, home_dic, num_homo, aggr_mode='mean'):
        HA = sp.dok_matrix((num_homo, num_homo), dtype=np.float32)
        data_dict = dict()
        n_k = 0
        for i in range(len(home_dic)):
            if len(home_dic[i][0]) == 0:
                continue
            if len(home_dic[i][0]) < self.k:
                n_k = len(home_dic[i][0])
            if len(home_dic[i][0]) >= self.k:
                n_k = self.k
            heads = [i] * n_k
            tails = home_dic[i][0][:n_k]
            weight = home_dic[i][1][:n_k]
            if aggr_mode == 'softmax':
                weight = F.softmax(torch.tensor(weight), dim=0)  # softmax
            if aggr_mode == 'mean':
                weight = torch.ones(n_k) / n_k  # mean
            data_dict.update(zip(zip(heads, tails), weight))
        
        HA._update(data_dict)
        HL = sp.coo_matrix(HA)
        
        return self.scipy_matrix_to_sparse_tenser(HL, torch.Size((num_homo, num_homo)))

    def forward_mm(self, adj, embs):
        for l in range(self.m_layer):
            embs = torch.sparse.mm(adj, embs)
        return embs
    
    def forward(self, adj, model_type='lightgcn'):
        if model_type == 'mf':
            return self.user_embedding.weight, self.item_id_embedding.weight
        
        elif model_type == 'lightgcn':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                ego_embeddings = side_embeddings
                all_embeddings += [ego_embeddings]
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            return u_g_embeddings, i_g_embeddings
        
        elif model_type == 'resgcn':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                ego_embeddings = side_embeddings
                all_embeddings += [ego_embeddings]
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            return u_g_embeddings, i_g_embeddings + self.item_id_embedding.weight

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss
    
    def sce_loss(self, x, y, alpha=3):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)

        # loss =  - (x * y).sum(dim=-1)
        # loss = (x_h - y_h).norm(dim=1).pow(alpha)

        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
        loss = loss.mean()
        return loss
    
    def setup_mm_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(self.sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion
    
    def ssl_triple_loss(self, z1: torch.Tensor, z2: torch.Tensor, all_emb: torch.Tensor):
        norm_emb1 = F.normalize(z1)
        norm_emb2 = F.normalize(z2)
        norm_all_emb = F.normalize(all_emb)
        pos_score = torch.mul(norm_emb1, norm_emb2).sum(dim=1)
        ttl_score = torch.matmul(norm_emb1, norm_all_emb.transpose(0, 1))
        pos_score = torch.exp(pos_score / self.tau)
        ttl_score = torch.exp(ttl_score / self.tau).sum(dim=1)

        ssl_loss = -torch.log(pos_score / ttl_score).sum()
        return ssl_loss
    
    def agg_mm_neighbors(self, str='v'):
        if str == 'v':
            item_feats = torch.mm(self.image_embedding.weight, self.item_image_trs)
        elif str == 't':
            item_feats = torch.mm(self.text_embedding.weight, self.item_text_trs)
        user_feats = torch.sparse.mm(self.adj, item_feats) * self.num_inters[:self.n_users]
        return user_feats, item_feats
    
    def get_sim_mat(self, x):
        with torch.no_grad():
            x = x.div(torch.norm(x, p=2, dim=-1, keepdim=True))
            sim = torch.mm(x, x.transpose(1, 0))
        return sim
    
    def get_sim_adj(self, x):
        sim = self.get_sim_mat(x)
        sim = torch.div(torch.add(sim, 1), 2)
        inter = torch.where(sim < 0.8, torch.zeros_like(sim), sim).to_sparse()
        indices = inter.indices()
        data = torch.ones_like(inter.values())
        pruned_sim_adj = torch.sparse.FloatTensor(indices, data, torch.Size((self.n_items, self.n_items))).coalesce()
        
        # normalize
        pruned_sim_indices = pruned_sim_adj.indices()
        diags = torch.sparse.sum(pruned_sim_adj, dim=1).to_dense() + 1e-7
        diags = torch.pow(diags, -1)
        diag_lookup = diags[pruned_sim_indices[0, :]]

        pruned_sim_adj_value = pruned_sim_adj.values()
        normal_sim_value = torch.mul(pruned_sim_adj_value, diag_lookup)
        normal_sim_adj = torch.sparse.FloatTensor(pruned_sim_indices, normal_sim_value, torch.Size((self.n_items, self.n_items))).to(self.device).coalesce()
        return normal_sim_adj
    
    def get_knn_adj_mat(self, mm_embeddings):
        sim = self.get_sim_mat(mm_embeddings)
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return self.compute_normalized_laplacian(indices, adj_size)
    
    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -1)
        values = r_inv_sqrt[indices[0]]
        return torch.sparse.FloatTensor(indices, values, adj_size).to(self.device)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        # 1. masked modal graph autoencoder
        self.criterion = self.setup_mm_loss_fn('mse', 2.0)
        mm_loss = 0.0
        user_image_feats, item_image_feats = None, None
        user_text_feats, item_text_feats = None, None
        if self.mask_rate > 0:
            # random masking
            perm = torch.randperm(self.n_items)
            num_mask_nodes = int(self.mask_rate * self.n_items)
            mask_nodes = perm[: num_mask_nodes]
            if self.v_feat is not None:
                user_image_feats, item_image_feats = self.agg_mm_neighbors('v')
                x = item_image_feats.clone()
                x[mask_nodes] = 0.0
                x[mask_nodes] += self.enc_mask_token
                image_embs = self.forward_mm(self.item_adj, x)
                re_item_image_embs = torch.mm(image_embs, torch.transpose(self.item_image_trs, 0, 1))
                mm_loss += self.criterion(re_item_image_embs[mask_nodes], self.image_embedding.weight[mask_nodes])
            
            if self.t_feat is not None:
                user_text_feats, item_text_feats = self.agg_mm_neighbors('t')
                x = item_text_feats.clone()
                x[mask_nodes] = 0.0
                x[mask_nodes] += self.enc_mask_token
                text_embs = self.forward_mm(self.item_adj, x)
                re_item_text_embs = torch.mm(text_embs, torch.transpose(self.item_text_trs, 0, 1))
                mm_loss += self.criterion(re_item_text_embs[mask_nodes], self.text_embedding.weight[mask_nodes])
        
        # 2. cross-modal knowledge alignment
        cs_loss = 0.0
        if self.v_feat is not None and self.t_feat is not None:
            image_adj = self.get_knn_adj_mat(image_embs+item_image_feats)
            ia_v_embeddings = self.forward_mm(image_adj, self.item_id_embedding.weight)

            text_adj = self.get_knn_adj_mat(text_embs+item_text_feats)
            ia_t_embeddings = self.forward_mm(text_adj, self.item_id_embedding.weight)

            cs_loss += self.ssl_triple_loss(ia_v_embeddings[pos_items], ia_t_embeddings[pos_items], ia_t_embeddings)
            cs_loss += self.ssl_triple_loss(ia_t_embeddings[pos_items], ia_v_embeddings[pos_items], ia_v_embeddings)
        # pdb.set_trace()

        # 3. fusion and prediction
        ua_embeddings, ia_embeddings = self.forward(self.norm_adj, self.cf_model)
        if self.v_feat is not None:
            ua_embeddings = ua_embeddings + F.normalize(user_image_feats, p=2, dim=1)
            ia_embeddings = ia_embeddings + F.normalize(item_image_feats, p=2, dim=1)
        if self.t_feat is not None:
            ua_embeddings = ua_embeddings + F.normalize(user_text_feats, p=2, dim=1)
            ia_embeddings = ia_embeddings + F.normalize(item_text_feats, p=2, dim=1)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_bpr_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        # pdb.set_trace()

        reg_loss = self.reg_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        return batch_bpr_loss  + self.mm_weight * mm_loss + self.ssl_weight * cs_loss + self.reg_weight * reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj, self.cf_model)
        if self.v_feat is not None:
            user_image_feats, item_image_feats = self.agg_mm_neighbors('v')
            restore_user_e = restore_user_e + F.normalize(user_image_feats, p=2, dim=1)
            restore_item_e = restore_item_e + F.normalize(item_image_feats, p=2, dim=1)
            
        if self.t_feat is not None:
            user_text_feats, item_text_feats = self.agg_mm_neighbors('t')
            restore_user_e = restore_user_e + F.normalize(user_text_feats, p=2, dim=1)
            restore_item_e = restore_item_e + F.normalize(item_text_feats, p=2, dim=1)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores