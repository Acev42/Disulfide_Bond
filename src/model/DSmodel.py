from torch import nn
from torch.nn import functional as F
import torch
from .gram_schmidt import rigidFrom3Points_torch
from .quaternion import rigid_transform
from einops import rearrange
from torch_geometric.utils import scatter, softmax
from .attention import GlobalLinearAttention,CombineCoorAtten
import pytorch_lightning as pl
from .calc_feature import _dihedrals
from .mlp import MLP
import numpy as np
import torchmetrics

def check_nan(x,name):
    has_nan = torch.isnan(x).any()
    print(f"{name} Has NaN: {has_nan}")

class DSgraphLayer(nn.Module):
    def __init__(self, edge_dim, hidden_dim, node_dim,n_head):
        super().__init__()
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.n_head = n_head

        dim_head = int(hidden_dim / n_head)
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(node_dim, hidden_dim, bias=False)
        self.to_kv = nn.Linear(node_dim, hidden_dim * 2, bias=False)
        self.to_out = nn.Linear(hidden_dim, node_dim)
        self.to_q_edge = nn.Linear(edge_dim, hidden_dim, bias=False)
        ndummy_atom = 0

        self.rbfMin = 2.0
        self.rbfMax = 30.0
        self.rbfN = 16
        self.register_buffer("D_mu", torch.linspace(self.rbfMin, self.rbfMax, self.rbfN))

        self.norm_node = nn.LayerNorm(node_dim)
        self.norm_edge = nn.LayerNorm(edge_dim)
        self.w0 = MLP(node_dim, node_dim * 2, 0.15)
        self.sidechaincoor_combine = CombineCoorAtten(-1, 64)


        self.gij_mlp = nn.Sequential(
            nn.Linear((ndummy_atom + 5) * (self.rbfN + 3), node_dim * 2, bias=True),
            nn.LeakyReLU(),
            nn.Linear(node_dim * 2, node_dim)
        )
        self.atten_weight_mlp = nn.Sequential(
            # nn.Linear(dim_head*3 + ndummy_atom*3 + ndummy_atom*self.rbfN, dim_head, bias=True),
            nn.Linear(dim_head * 4, dim_head, bias=True),
            nn.LeakyReLU(),
            nn.Linear(dim_head, 1)
        )
        self.v_mlp = nn.Sequential(
            # nn.Linear(dim_head*2 + ndummy_atom*3 + ndummy_atom*self.rbfN, dim_head, bias=True),
            nn.Linear(dim_head * 3, dim_head, bias=True),
            nn.LeakyReLU(),
            nn.Linear(dim_head, dim_head, bias=True)
        )
        self.edge_mlp = nn.Sequential(
                #nn.Linear(dim_head*3 + ndummy_atom*3 + ndummy_atom*self.rbfN, dim_head, bias=True),
            nn.Linear(edge_dim*4, edge_dim, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(0.15),
            nn.Linear(edge_dim, edge_dim, bias=True)
            )

    def rbf(self, D):
        # Distance radial basis function
        # device = D.device
        D_min, D_max, D_count = self.rbfMin, self.rbfMax, self.rbfN
        # D_mu = torch.linspace(D_min, D_max, D_count).to(device)
        D_mu = self.D_mu
        S = [1] * len(D.shape)
        D_mu = D_mu.view([*S, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF

    def update_hij(self, hij):
        hij_l = torch.linalg.vector_norm(hij, dim=-1, keepdim=True) + 1e-5  # vector length
        hij = hij / hij_l
        hij_l_rbf = torch.squeeze(self.rbf(hij_l), -2)  # nedge ndummy nrbf

        gij = torch.cat([hij, hij_l_rbf], -1)
        gij = rearrange(gij, 'n h d -> n (h d)')  # nedge, (ndummy+5)*3 + (ndummy+5)*nrbf
        return gij

    def calc_gij(self, edge_index, BBcoors, transform):

        src, target = edge_index
        BBcoors_src_rot = transform.reverse_apply_residue(BBcoors)[src]
        coor_src = BBcoors_src_rot

        transform_src = transform[src]
        BBcoors_target_rot = transform_src.reverse_apply_residue(BBcoors[target])
        coor_target = BBcoors_target_rot

        hij = self.sidechaincoor_combine(coor_src, coor_target)
        return self.update_hij(hij)

    def forward(self, node_feats, edge_feats, edge_index, backbone, transform):
        sidechain_coors = None
        gij = self.calc_gij(edge_index, backbone, transform)  # nedge, ndummy*3 + ndummy+16
        gij = self.gij_mlp(gij)
        gij = rearrange(gij, 'n (h d) -> n h d', h=self.n_head)

        q = self.to_q(node_feats)
        kv = self.to_kv(node_feats).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'n (h d) -> n h d', h = self.n_head), (q, *kv))
        edgeq = self.to_q_edge(edge_feats)
        edgeq = rearrange(edgeq, 'n (h d) -> n h d', h = self.n_head)
        q_src = torch.index_select(q, 0, edge_index[0])
        k_target = torch.index_select(k, 0, edge_index[1])

        atten = self.atten_weight_mlp(torch.cat([q_src, k_target, edgeq, gij], -1))  #
        atten = softmax(atten, index=edge_index[0])  # m h 1

        v_target = self.v_mlp(torch.cat((k_target, edgeq, gij), -1))
        out = atten * v_target
        node_update = scatter(out, index=edge_index[0], reduce='sum')  # n_node h d
        node_update = rearrange(node_update, 'n h d -> n (h d)', h=self.n_head)  # * node_gate
        node_update = self.norm_node(self.w0(node_update) + node_feats)
        gij_merge = rearrange(gij, 'n h d -> n (h d)', h=self.n_head)
        edges = torch.cat((torch.index_select(node_update, 0, edge_index[0]),
                           torch.index_select(node_update, 0, edge_index[1]),
                           edge_feats, gij_merge), dim=-1)  # m h d*4
        edges_update = self.edge_mlp(edges)  # m h
        # edges_update = rearrange(edges_update, 'n h d -> n (h d)', h = h)
        edges_update = self.norm_edge(edges_update + edge_feats)

        return node_update, edges_update


class DSModel(nn.Module):
    def __init__(self,node_dim, edge_dim, edge_cutoff, n_graph_layer, n_head,esm_dim = 960):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.edge_cutoff = edge_cutoff
        self.n_graph_layer = n_graph_layer
        self.n_head = n_head
        self.esm_dim = esm_dim

        self.embed_resname = nn.Embedding(20, self.node_dim)
        self.rbfMin = 2.0
        self.rbfMax = 30.0
        self.rbfN = 16
        self.register_buffer("D_mu", torch.linspace(self.rbfMin, self.rbfMax, self.rbfN))

        self.rbfMin_coor = -15
        self.rbfMax_coor = 15
        self.rbfN_coor = 24
        self.register_buffer("D_mu_coor", torch.linspace(self.rbfMin_coor, self.rbfMax_coor, self.rbfN_coor))
        self.resind_diff_dim = 16
        self.embed_resind_diff = nn.Embedding(65, self.resind_diff_dim)

        self.node_mlp = nn.Sequential(
            nn.Linear(4*self.rbfN+ 4 * 3 + 14 + esm_dim,node_dim , bias=True),
            nn.LeakyReLU(),
            #nn.Linear(dim*4, dim*4, bias=True),
            #nn.LeakyReLU(),
            #nn.Linear(dim*4, dim, bias=True)
            )
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.rbfN + 3 + 10 * 3 * self.rbfN_coor + self.resind_diff_dim, edge_dim, bias=True),
            nn.LeakyReLU(),
            # nn.Linear(edge_dim*4, edge_dim*4, bias=True),
            # nn.LeakyReLU(),
            # nn.Linear(edge_dim*4, edge_dim, bias=True)
        )
        self.nglobal_token = 24
        self.global_tokens = nn.Parameter(torch.randn(self.nglobal_token, node_dim))
        self.global_attn = GlobalLinearAttention(dim=node_dim,heads=self.n_head, dim_head=int(node_dim / self.n_head))

        self.graph = nn.ModuleList()
        for i in range(self.n_graph_layer):
            self.graph.append(DSgraphLayer(node_dim=node_dim,edge_dim= edge_dim, hidden_dim=node_dim, n_head=self.n_head))

        self.out_mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_dim * 2),
            nn.LayerNorm(edge_dim * 2),
            nn.ReLU(),
            # nn.Linear(dim*4, dim*4),
            # nn.ReLU(),
            nn.Linear(edge_dim * 2, 1),
            # nn.Sigmoid()
        )

    def rbf_coor(self, D):
        # Distance radial basis function
        # device = D.device
        D_min, D_max, D_count = self.rbfMin_coor, self.rbfMax_coor, self.rbfN_coor
        # D_mu = torch.linspace(D_min, D_max, D_count).to(device)
        D_mu = self.D_mu_coor
        S = [1] * len(D.shape)
        D_mu = D_mu.view([*S, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF

    def rbf(self, D):
        # Distance radial basis function
        #device = D.device
        D_min, D_max, D_count = self.rbfMin, self.rbfMax, self.rbfN
        #D_mu = torch.linspace(D_min, D_max, D_count).to(device)
        D_mu = self.D_mu
        S = [1] * len(D.shape)
        D_mu = D_mu.view([*S,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF


    def calc_node_feats_geo(self, BBcoors, transform, connect):
        BBcoors_rev = transform.reverse_apply_residue(BBcoors) #n, 5, 3  n ca c o cb
        BBcoors_rev = BBcoors_rev[:, [0,2,3,4], :] #N, 4, 3 atoms exclude Ca
        coor_dist = torch.linalg.vector_norm(BBcoors_rev, dim=-1, keepdim=True) + 1e-5 #N, 3, 1
        unitvector = BBcoors_rev / coor_dist
        unitvector = rearrange(unitvector, 'n i d -> n (i d)')
        coor_dist_rbf = self.rbf(torch.squeeze(coor_dist, -1)) #N 4 rbf
        coor_dist_rbf = rearrange(coor_dist_rbf, 'n i d -> n (i d)')
        #unit vector and distance of n c o cb atom to ca atom in the standard position
        geo_feats = torch.cat((unitvector, coor_dist_rbf), -1) #N, 4*nrbf + 4*3

        angle_torsion = _dihedrals(BBcoors, connect, eps=1e-7) #N 14
        geo_feats = torch.cat((geo_feats, angle_torsion), -1)

        return geo_feats

    def get_edge_indices(self,batch):
        backbone = batch['coord']
        # [n,5,3] N,Ca,C,O,Cb
        cb_coord = backbone[:,4,:]
        dist_matrix = torch.cdist(cb_coord, cb_coord)
        # n_res n_res
        dist_matrix_bin = dist_matrix <= self.edge_cutoff
        index = dist_matrix_bin.nonzero()
        src = index[:, 0]
        target = index[:, 1]
        return src, target

    def calc_edge_feats(self, CAcoors, BBcoors, src, target, transform):
        CAcoor_diff = CAcoors[target] - CAcoors[src]
        edge_dist = torch.linalg.vector_norm(CAcoor_diff, dim=-1, keepdim=True) + 1e-5  # distance between residue pairs
        edge_dist_rbf = torch.squeeze(self.rbf(edge_dist), 1)
        transform_src = transform[src]

        BBcoors_src_rotated = transform.reverse_apply_residue(BBcoors)[src]
        BBcoors_target_rotated = transform_src.reverse_apply_residue(BBcoors[target])
        edge_vector = (BBcoors_target_rotated[:, 1, :] - BBcoors_src_rotated[
            :, 1, :]) / edge_dist  # unit vector between residue pairs, after the src residue is rotated

        edge_dist_pair = torch.cat([BBcoors_src_rotated, BBcoors_target_rotated], dim=1)  # n 10 3
        # edge_dist_pair = rearrange(edge_dist_pair, 'n i j d -> n (i j d)') #m 5 5 3 -> m 75
        edge_dist_pair = rearrange(edge_dist_pair, 'n i d -> n (i d)')  # m 10 3 -> m 30
        edge_dist_pair = self.rbf_coor(edge_dist_pair)
        edge_dist_pair = rearrange(edge_dist_pair, 'n i d -> n (i d)')  # m 30 self.rbfN_coor -> m 30*self.rbfN_coor
        edge_feats = torch.cat((edge_dist_rbf, edge_vector, edge_dist_pair), dim=-1)
        return edge_feats


    def forward(self,batch):
        backbone = batch['coord']
        # [n,5,3] N,Ca,C,O,Cb
        ca_coord = backbone[:,1,:]
        n_coord = backbone[:,0,:]
        c_coord = backbone[:,2,:]
        # [n,3]
        res_id = batch['res_id']
        embedding = batch['embedding']
        # n_res, 960
        connect = batch['connect']
        # n_res,2
    #  node feature
        rotate_mats = rigidFrom3Points_torch(n_coord,ca_coord,c_coord)
        # [n,3,3], each [3,3] is a 3d basis matrix
        relative_coord_transform = rigid_transform(ca_coord, rotate_mats)
        node_feat_geo = self.calc_node_feats_geo(backbone, relative_coord_transform, connect)
        try:
            node_feats = self.node_mlp(torch.cat([node_feat_geo,embedding], -1))
        except Exception as e:
            print(batch['pdb'])
            print(backbone.shape)
            print(embedding.shape)
    #   edge feature
        src, target = self.get_edge_indices(batch)
        edge_index = torch.stack((src, target))
        edge_feats = self.calc_edge_feats(ca_coord, backbone, src, target, relative_coord_transform)
        resind_diff = torch.clip(res_id[src] - res_id[target], max=32, min=-32) + 32

        resind_diff_embed = self.embed_resind_diff(resind_diff)
        edge_feats = torch.cat([resind_diff_embed, edge_feats], dim=-1)
        edge_feats = self.edge_mlp(edge_feats)
        global_tokens = torch.unsqueeze(self.global_tokens, 0)
        for i in range(self.n_graph_layer):
            node_feats, edge_feats = self.graph[i](
                node_feats,
                edge_feats,
                edge_index,
                backbone,
                relative_coord_transform)
            node_feats = torch.unsqueeze(node_feats, 0)
            node_feats, global_tokens = self.global_attn(node_feats, global_tokens)
            node_feats = torch.squeeze(node_feats, 0)
        probs = self.out_mlp(edge_feats)
        return probs, edge_index
    # return edge indices

class DS(pl.LightningModule):
    def __init__(self, node_dim, edge_dim, distcut,nlayer, n_head, esmdim=960,
                 loss_weight_chi=1.0, lr=1e-4, decay=0,loss_weight_CB=1.0,
                 threshold=0.5):
        super().__init__()
        self.save_hyperparameters("node_dim", "edge_dim", "distcut", "nlayer",
                                  "n_head", "loss_weight_chi", "lr", "decay",
                                    "loss_weight_CB", "esmdim",'threshold')
        self.loss_weight_chi = loss_weight_chi
        self.loss_weight_CB = loss_weight_CB
        self.lr = lr
        self.weight_decay = decay
        self.threshold = threshold
        self.dsmodel = DSModel(node_dim, edge_dim, distcut, nlayer, esm_dim=esmdim, n_head=n_head)
        self.pos_weight = torch.tensor([72])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        self.val_acc = torchmetrics.classification.BinaryAccuracy()
        self.val_auc = torchmetrics.classification.BinaryAUROC()
        self.val_confmat = torchmetrics.classification.BinaryConfusionMatrix()
        self.test_acc = torchmetrics.classification.BinaryAccuracy()
        self.test_auc = torchmetrics.classification.BinaryAUROC()
        self.test_confmat = torchmetrics.classification.BinaryConfusionMatrix()

        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.test_step_outputs = []


    def forward(self, batch):
        pred,edge_indices = self.dsmodel(batch)
        return pred,edge_indices

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


    def get_pred_and_label(self, train_batch):
        prob,edge_indices = self(train_batch)
        pdb = train_batch['pdb']
        chain = train_batch['chain']
        res_id = train_batch['res_id']
        samples = train_batch['sample']
        label_list = train_batch['label']

        res_to_index = {int(res): idx for idx, res in enumerate(res_id)}
        prob_list = []
        for res1, res2 in samples:
            res1_idx = res_to_index.get(int(res1))
            res2_idx = res_to_index.get(int(res2))
            mask1 = (edge_indices[0, :] == res1_idx) & (edge_indices[1, :] == res2_idx)
            mask2 = (edge_indices[0, :] == res2_idx) & (edge_indices[1, :] == res1_idx)
            edge_index1 = np.where(mask1)[0]
            edge_index2 = np.where(mask2)[0]
            if len(edge_index1) == 0 or len(edge_index2) == 0:
                #print(f" res1_idx={res1_idx}, res2_idx={res2_idx} edge not found")
                continue
            probs = torch.stack([prob[edge_index1], prob[edge_index2]])
            prob_list.append(torch.mean(probs))
            '''edge_index1 = np.where(mask1)[0]
            if len(edge_index1) == 0:
                print(f" {pdb} {chain} res1_idx={res1_idx},res2_idx={res2_idx} edge not found")
                continue
            prob_list.append(prob[edge_index1])'''
        prob_list = torch.stack(prob_list, dim=0).squeeze()
        if prob_list.shape != label_list.shape:
            return None,None

        return prob_list, label_list

    def training_step(self, train_batch, batch_idx):
        pred_list,label_list = self.get_pred_and_label(train_batch)
        if pred_list is None:
            return None
        loss= self.criterion(target=label_list,input=pred_list)
        self.log('loss', loss, on_step=True, prog_bar=True, logger=True,batch_size=1)
        self.training_step_outputs.append({"loss":loss})
        return loss

    def on_train_epoch_end(self):
        loss = torch.stack([i["loss"] for i in self.training_step_outputs])
        loss = torch.mean(loss)
        self.training_step_outputs.clear()
        self.log('train_loss_epoch', loss, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self,val_batch,batch_idx):

        pred_list, label_list = self.get_pred_and_label(val_batch)
        if pred_list is None:
            return None
        loss= self.criterion(target=label_list,input=pred_list)
        self.validation_step_outputs.append({"loss": loss, "label": label_list, "pred":pred_list })
        preds = (pred_list >= 0.5).int()
        self.val_acc.update(preds, label_list)
        self.val_auc.update(pred_list, label_list)
        self.val_confmat.update(preds, label_list)
        self.log('loss', loss, on_step=True, prog_bar=True, logger=True,batch_size=1)
        self.log("val_auc", self.val_auc, on_step=True, prog_bar=True, logger=True,batch_size=1)
        self.log("val_acc", self.val_acc, on_step=True, prog_bar=True, logger=True,batch_size=1)
        return loss

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return None
        outs = self.validation_step_outputs
        labels = np.concatenate([i["label"].cpu().to(torch.float).numpy() for i in outs])
        probs = np.concatenate([i["pred"].cpu().to(torch.float).numpy() for i in outs])

        cm = self.val_confmat.compute()
        TN, FP, FN, TP = cm.flatten()
        fpr = FP / (FP + TN + 1e-8)
        fnr = FN / (FN + TP + 1e-8)

        self.log("val_fpr", fpr, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_fnr", fnr, on_epoch=True, prog_bar=True, logger=True)

        if self.logger is not None and hasattr(self.logger, "experiment"):
            self.logger.experiment.add_pr_curve(
                "val_PR_curve",
                labels,
                probs,
                global_step=self.current_epoch
            )
        self.validation_step_outputs.clear()

    def test_step(self,test_batch, batch_idx):
        pred_list,label_list = self.get_pred_and_label(test_batch)
        if pred_list is None:
            return None
        loss= self.criterion(target=label_list,input=pred_list)
        preds = (pred_list >= 0.5).int()
        self.test_acc(preds, label_list)
        self.test_auc(pred_list, label_list)
        self.test_confmat(preds, label_list)
        self.test_step_outputs.append({"loss": loss, "label": label_list, "pred":pred_list })
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=1)
        self.log("test_auc", self.test_auc, on_step=True, prog_bar=True, logger=True,batch_size=1)
        self.log("test_acc", self.test_acc, on_step=True, prog_bar=True, logger=True,batch_size=1)
        return loss

    def on_test_epoch_end(self):
        outs = self.test_step_outputs
        labels = np.concatenate([i["label"].cpu().to(torch.float).numpy() for i in outs])
        probs = np.concatenate([i["pred"].cpu().to(torch.float).numpy() for i in outs])

        cm = self.test_confmat.compute()
        TN, FP, FN, TP = cm.flatten()

        fpr = FP / (FP + TN + 1e-8)
        fnr = FN / (FN + TP + 1e-8)
        self.log("test_fpr", fpr, prog_bar=True, on_epoch=True,logger=True)
        self.log("test_fnr", fnr, prog_bar=True, on_epoch=True,logger=True)


        if self.logger is not None and hasattr(self.logger, "experiment"):
            self.logger.experiment.add_pr_curve(
                "test_PR_curve",
                labels,
                probs,
                global_step=self.current_epoch)

        self.test_step_outputs.clear()

    def predict_step(self,batch):
        res_id = batch["res_id"].squeeze()
        threshold = self.threshold
        index_to_res = {int(idx): int(res) for idx, res in enumerate(res_id)}
        prediction,edge_indices = self(batch)
        prediction = torch.sigmoid(prediction)
        # prediction n,1  n= n_edge
        # edge_indices 2,n
        possible_mask = prediction >= threshold
        selected_prediction = prediction[possible_mask].squeeze()
        if selected_prediction.dim() == 0:
            selected_prediction = selected_prediction.unsqueeze(0)
        possible_mask = possible_mask.squeeze()

        possible_edge = edge_indices[:,possible_mask]
        possible_edge = possible_edge.T
        # n,2
        output_edge = {}
        if possible_mask.sum() == 0:
            print('No potential disulfide bond found')
        else:
            for i,edge in enumerate(possible_edge):
                src,tgt = edge
                src_res_id = index_to_res[int(src)]
                tgt_res_id = index_to_res[int(tgt)]

                prob = float(selected_prediction[i])
                if (tgt_res_id,src_res_id) not in output_edge:
                    output_edge[(src_res_id,tgt_res_id)] = prob
                else:
                    output_edge[(tgt_res_id,src_res_id)] = 0.5*(prob+output_edge[(tgt_res_id,src_res_id)])
        return batch['pdb'],batch['chain'],output_edge
