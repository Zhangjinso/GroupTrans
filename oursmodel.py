import torch.nn as nn
import torch
from bert_transformer import *
import numpy as np
class MLP(nn.Module):
    def __init__(self,indim,outdim) -> None:
        super().__init__()
        self.fc1=nn.Sequential(nn.Linear(indim,outdim),nn.ReLU())
    def forward(self,x,attention_mask = None):
        return self.fc1(x)
class TemporalDenseConvLayer(nn.Module):

    def __init__(self,indim,outdim,num_convs = 2):
        super().__init__()
        self.convs = nn.ModuleList([nn.Sequential(nn.Conv1d(indim + i*outdim, outdim,3,1,1,bias=False,padding_mode="replicate"),nn.BatchNorm1d(outdim),nn.ReLU()) for i in range(num_convs)])
        self.outconv = nn.Sequential(nn.Conv1d(indim + num_convs*outdim, outdim,3,1,1,bias=False,padding_mode="replicate"),nn.BatchNorm1d(outdim),nn.ReLU())

    def forward(self,x):
        # x : [ num_person, feature_dim, seq_len]
        for cv in self.convs:
            x1 = cv(x)
            x = torch.cat([x,x1],dim = 1)
        return self.outconv(x)
class TemporalDenseConvLayer1(nn.Module):

    def __init__(self,indim,outdim,middle_dim,num_convs = 2):
        super().__init__()
        self.convs = nn.ModuleList([nn.Sequential(nn.Conv1d(indim + i*middle_dim, middle_dim,3,1,1,bias=False,padding_mode="replicate"),nn.BatchNorm1d(middle_dim),nn.ReLU()) for i in range(num_convs)])
        self.outconv = nn.Sequential(nn.Conv1d(indim + num_convs*middle_dim, outdim,3,1,1,bias=False,padding_mode="replicate"),nn.BatchNorm1d(outdim),nn.ReLU())

    def forward(self,x):
        # x : [ num_person, feature_dim, seq_len]
        for cv in self.convs:
            x1 = cv(x)
            x = torch.cat([x,x1],dim = 1)
        return self.outconv(x)
class TemporalDenseConvLayer2(nn.Module):

    def __init__(self,indim,outdim,num_convs = 2):
        super().__init__()
        self.convs = nn.ModuleList([nn.Sequential(nn.Conv1d(indim + i*outdim, outdim,3,1,1,bias=False,padding_mode="replicate"),nn.BatchNorm1d(outdim),nn.ReLU()) for i in range(num_convs)])
        self.outconv = nn.Sequential(nn.Conv1d(indim + num_convs*outdim, outdim,3,1,1,bias=False,padding_mode="replicate"),nn.BatchNorm1d(outdim),nn.ReLU())

    def forward(self,x):
        # x : [ num_person, feature_dim, seq_len]
        for cv in self.convs:
            x1 = cv(x)
            x = torch.cat([x,x1],dim = 1)
        return self.outconv(x)

config = BertConfig(hidden_size=128,
                 num_hidden_layers=2,   # 2
                 num_attention_heads=4, # 4
                 intermediate_size=128,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.0, # 0.1
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=64,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,)

class SpatialTransformerEncoder(nn.Module):

    def __init__(self,indim,outdim, use_pos_emb = False):
        super(SpatialTransformerEncoder, self).__init__()

        config.hidden_size = outdim

        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.img_dim = indim
        self.use_pos_emb = use_pos_emb
        try:
            self.use_img_layernorm = config.use_img_layernorm
        except:
            self.use_img_layernorm = None

        self.img_embedding = nn.Linear(self.img_dim, outdim, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, img_feats, input_ids=None, token_type_ids=None, attention_mask=None,
            position_ids=None, head_mask=None):
        # img_feats : [num frames, num persons, features]
        if img_feats.shape[1] <= 300:
            return self.forward_afew(img_feats,input_ids,token_type_ids,attention_mask,position_ids,head_mask)
        else:
            res = []
            stride = 10
            for t in range(0,img_feats.shape[0],stride):
                subfts = img_feats[t:t+stride]
                submsk = attention_mask[t:t+stride]
                res.append(self.forward_afew(subfts,input_ids,token_type_ids,submsk,position_ids,head_mask))
            return torch.cat(res)

    def forward_afew(self, img_feats, input_ids=None, token_type_ids=None, attention_mask=None,
            position_ids=None, head_mask=None):
        # img_feats : [num frames, num persons, features]

        batch_size = len(img_feats)
        seq_length = len(img_feats[0])
        input_ids = torch.zeros([batch_size, seq_length],dtype=torch.long).cuda()

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if self.use_pos_emb:
            position_embeddings = self.position_embeddings(position_ids)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # Project input token features to have spcified hidden size
        img_embedding_output = self.img_embedding(img_feats)

        # We empirically observe that adding an additional learnable position embedding leads to more stable training
        if self.use_pos_emb:
            # print(position_embeddings.shape)
            # print(img_embedding_output.shape)
            embeddings = position_embeddings + img_embedding_output
        else:
            embeddings = img_embedding_output

        if self.use_img_layernorm:
            embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        encoder_outputs = self.encoder(embeddings,
                extended_attention_mask, head_mask=head_mask)
        sequence_output = encoder_outputs[0]

        return sequence_output



class PredictionHead(nn.Module):

    def __init__(self,indim,embdim,reduce_cross_frame = 'mean') -> None:
        super().__init__()
        mask_value = 20
        self.dropout = nn.Dropout()
        # self.dropout = nn.Identity()
        if embdim > 0:
            self.proj = nn.Sequential(nn.Linear(indim,embdim),nn.ReLU())
        else:
            self.proj = nn.Identity()
            embdim = indim
        self.cls = nn.Linear(embdim,1)
        self.reduce_frame = {
            'mean' : lambda x,m : (x * self.dropout(m)).sum(dim = 0) / (m.sum(dim = 0) + 0.0001),
            'max'  : lambda x,m : (x - mask_value + mask_value * m).max(dim = 0),
            'top10': lambda x,m : torch.sort(x - mask_value + mask_value * m,dim = 0)[0][-10:].mean(dim = 0),
            'top5': lambda x,m : torch.sort(x - mask_value + mask_value * m,dim = 0)[0][-5:].mean(dim = 0),
            'top2': lambda x,m : torch.sort(x - mask_value + mask_value * m,dim = 0)[0][-3:].mean(dim = 0),
        }[reduce_cross_frame]
    
    def forward(self,x,eoi,psmask,require_framewise = False):
        # x :   [num frames, num persons, feature dim]
        # eoi : [2, edges_num]
        # n,p,d = x.shape
        # x = self.proj(x.reshape([n*p,d])).reshape([n,p,d])
        x = self.proj(x)
        
        if eoi.shape[1] < 1000:
            # edl = x[:,eoi[0]] + x[:,eoi[1]]
            edl = torch.abs(x[:,eoi[0]] - x[:,eoi[1]])
            edm = psmask[:,eoi[0]] * psmask[:,eoi[1]]
            edl = self.cls(edl)[:,:,0]
            res = self.reduce_frame(edl,edm)
            if require_framewise:
                return edl,edm,res
        else:
            res = []
            for i in range(0,eoi.shape[1],1000):
                seoi = eoi[:,i:i+1000]
                # edl = x[:,seoi[0]] + x[:,seoi[1]]
                edl = torch.abs(x[:,seoi[0]] - x[:,seoi[1]])
                edm = psmask[:,seoi[0]] * psmask[:,seoi[1]]
                edl = self.cls(edl)[:,:,0]
                res.append(self.reduce_frame(edl,edm))
            res = torch.cat(res)
        return res

# config structure
config_template = {
    'appearance_feature_dim' : 0,  # 0 represent disabled appearance feature input
    'temporal_hidden_dim1' : 32,
    'temporal_hidden_layernum1' : 2,
    'temporal_hidden_dim2' : 64,
    'temporal_hidden_layernum2' : 2,
    'spatial_transformer_dim1' : 128,
    'spatial_transformer_dim2' : 128,
    'prediction_embedding_dim' : 64,
    'prediction_reduce_time' : 'top10',
    'use_occusion_attention' : True
}

def filter_impossible_edges(feature,edges,mask,dis_thr,cond_only = False, min_time_overlap = 0,verbose = False):
    feature = feature[:,:,:4]
    # feature : person num, frame num, bbox
    if edges.shape[1] > 100000:
        pairdis = []
        timemsk = []
        for spl in range(0,edges.shape[1],100000):
            sedges = edges[:,spl:spl + 100000]
            parimsk = (1 - mask[sedges[0]] * mask[sedges[1]]) * dis_thr ** 2
            umsk = mask[sedges[0]] + mask[sedges[1]]
            tm = (mask[sedges[0]] * mask[sedges[1]]).sum(axis = 1) / (umsk / (umsk + 0.00001)).sum(axis = 1)
            pd = (((feature[sedges[0]] - feature[sedges[1]]) ** 2).mean(axis = 2) + parimsk).min(axis = 1)
            pairdis.append(pd)
            timemsk.append(tm)
        pairdis = np.concatenate(pairdis)
        timemsk = np.concatenate(timemsk)
    else:
        parimsk = (1 - mask[edges[0]] * mask[edges[1]]) * dis_thr ** 2
        umsk = mask[edges[0]] + mask[edges[1]]
        timemsk = (mask[edges[0]] * mask[edges[1]]).sum(axis = 1) / (umsk / (umsk + 0.00001)).sum(axis = 1)
        pairdis = (((feature[edges[0]] - feature[edges[1]]) ** 2).mean(axis = 2) + parimsk).min(axis = 1)
    if verbose:
        print('----')
        print(timemsk)
        print(timemsk > min_time_overlap)
        # print((edges.T)[timemsk > min_time_overlap])
    cond = (pairdis < (dis_thr ** 2)) & (timemsk > min_time_overlap)
    if not cond_only:
        return edges[:,cond],cond
    else:
        return cond

class OcclusionAttention(nn.Module):

    def __init__(self,indim,config):
        super().__init__()
        self.use_att = config['use_occusion_attention']
        app_ftdim = config['appearance_feature_dim']
        if self.use_att:
            self.proj = nn.Sequential(nn.Linear(indim,1024),nn.ReLU(),)
        self.value = nn.Sequential(nn.Linear(indim,app_ftdim),nn.ReLU(),)
    
    def forward(self, x, mask):
        # x : [num person, num frame, num feature dim]
        # mask : [num frames, num persons] 
        if self.use_att:
            xp = self.proj(x)
            xp = xp / (xp.norm(dim = 2,keepdim = True) + 1e-8)
            mask = torch.transpose(mask,0,1).reshape([x.shape[0],1,x.shape[1]])
            att = (torch.bmm(xp,torch.transpose(xp,1,2)) * mask).sum(dim = 2,keepdim=True) / mask.sum(dim = 2,keepdim=True)
            xv = self.value(x)
            return att * xv
        else:
            return self.value(x)


class OverallModel1(nn.Module):

    def __init__(self,indim, config) -> None:
        super().__init__()
        self.temporal_conv1 = TemporalDenseConvLayer1(indim,config['temporal_hidden_dim1'],config['temporal_hidden_middle_dim1'],config['temporal_hidden_layernum1'])
        hidden_dim1 = config['temporal_hidden_dim1']
        self.temporal_conv2 = TemporalDenseConvLayer1(hidden_dim1,config['temporal_hidden_dim2'],config['temporal_hidden_middle_dim2'],config['temporal_hidden_layernum2'])
        hidden_dim2 = config['temporal_hidden_dim2']
        self.spatial_encoder1 = SpatialTransformerEncoder(hidden_dim1 + config['appearance_feature_dim'], config['spatial_transformer_dim1'])
        self.spatial_encoder2 = SpatialTransformerEncoder(hidden_dim2 + config['spatial_transformer_dim1'], config['spatial_transformer_dim2'])

        predindim = config['spatial_transformer_dim2']
        self.shortcut2pred = config['shortcut2predhead']
        if self.shortcut2pred:
            predindim += hidden_dim1 + hidden_dim2 + config['spatial_transformer_dim1'] + indim
        self.pred = PredictionHead(predindim,config['prediction_embedding_dim'],config['prediction_reduce_time'])
        self.use_app = config['appearance_feature_dim'] != 0

        if self.use_app:
            self.occlusion_att = OcclusionAttention(4*2*2048,config)
    
    def convert_temporal_spatial(self,x):
        return torch.transpose(torch.transpose(x,1,2),0,1)

    def forward(self,x_rbbox,eoi,psmask, x_app = None, is_app_attended = False,require_framewise = False):
        # x_bbox : [person num, seq len, feature dim]
        x_bbox = self.temporal_conv1(x_rbbox)
        if self.use_app:
            if not is_app_attended:
                x_app = self.occlusion_att(x_app,psmask)
            x_app = torch.transpose(x_app,1,2)
            x = self.convert_temporal_spatial(torch.cat([x_bbox, x_app],dim = 1))
        else:
            x = self.convert_temporal_spatial(x_bbox)
       
        x = self.spatial_encoder1(x,attention_mask = psmask)
        x_bbox1 = self.temporal_conv2(x_bbox) 
        x_bbox1 = self.convert_temporal_spatial(x_bbox1)
        
        x1 = torch.cat([x_bbox1, x],dim = 2)
        x = self.spatial_encoder2(x1,attention_mask = psmask)
        if self.shortcut2pred:
            x = torch.cat([self.convert_temporal_spatial(torch.cat([x_bbox,x_rbbox],dim = 1)),x1,x],dim = 2)
        res = self.pred(x,eoi,psmask,require_framewise = require_framewise)

        return res

