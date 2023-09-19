import torch
import os, sys, json, argparse
import random
from random import choice, sample
# from tensorboardX import SummaryWriter
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import numpy as np
import pickle
from train_ours import get_person_traj_group_list,model_config,load_person_appearance
from tqdm import tqdm
from graph import graph_propagation
from group_eval import group_eval
from oursmodel import OverallModel1, filter_impossible_edges 

import config_param
import time
def init_weights(m):
    for name, param in m.named_parameters():
        torch.nn.init.uniform_(param.data, -0.08, 0.08)


def apply_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.train()


class TrajectoryEmbeddingNet(torch.nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout, embedding_num=512, use_linear=True, use_in=True):
        super().__init__()
        self.embedding_num = embedding_num
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.rnn = torch.nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout, bidirectional=True)

        self.dropout = torch.nn.Dropout(dropout)
        self.apply(init_weights)
        self.normalization = torch.nn.InstanceNorm1d(1)
        self.linear = torch.nn.Linear(self.n_layers * 2 * 1 * self.hid_dim * 2, self.embedding_num)
        self.use_linear = use_linear
        self.use_in = use_in

    def forward(self, src):
        # src = [src sent len, input dim]
        src = src.view(len(src), 1, -1)
        dropouted = self.dropout(src)

        # dropouted = [src sent len, batch size(1), input dim]

        outputs, (hidden, cell) = self.rnn(dropouted)
        embedding = torch.cat([hidden.view(-1), cell.view(-1)], 0).view(1, 1, -1)
        if self.use_linear:
            embedding = self.linear(embedding)
        if self.use_in:
            embedding = self.normalization(embedding)
        return embedding


def plot_traj_pair(persona, personb):
    tranj_a = np.array(persona["traj"])
    tranj_b = np.array(personb["traj"])
    time_max = max(tranj_a[:, 0].max(), tranj_b[:, 0].max())
    fig = plt.figure(dpi=300)
    for id, segment in enumerate([1.0, 2.0, 3.0], 1):
        tranj_a_seg_mask = np.less(tranj_a[:, 0], time_max * segment / 3.0)
        tranj_a_seg = tranj_a[tranj_a_seg_mask, 1:3]

        tranj_b_seg_mask = np.less(tranj_b[:, 0], time_max * segment / 3.0)
        tranj_b_seg = tranj_b[tranj_b_seg_mask, 1:3]
        plt.subplot("13{}".format(id))
        plt.axis([0, 1, 0, 1])
        plt.scatter(tranj_a_seg[:, 0], tranj_a_seg[:, 1], c="green", marker="o", s=1, alpha=0.1)
        plt.scatter(tranj_b_seg[:, 0], tranj_b_seg[:, 1], c="blue", marker="v", s=1, alpha=0.1)
        plt.tight_layout()
    return fig

def convert_feature(x):
    # print(x.shape)
    m,x = x[0],x[1]
    #print(m,len(m),x.shape)
    pad = np.zeros(x[0].shape)
    res = []
    vi = 0
    for i in range(len(m)):
        if m[i]:
            res.append(x[vi])
            vi += 1
        else:
            res.append(pad)
    return np.stack(res)

def test_group_embedding():
    MIN_DIS_THR = config_param.TEST_MIN_DIS_THR
    traj_net.eval()
    with torch.no_grad():
        for id, scene in enumerate(val_set):
            all = scene[0]
            all_ps = list(all.keys())
            # print(all_ps)
            sampled_ps = all_ps
            #print('id',sampled_ps)
            person_label = []
            for p in all_ps:
                person_label.append(all[p]["group"] if len(all[p]["group"]) != 0 else [])
            input_feat = []
            frame_mask = []
            input_skeletonft = []
            for p in sampled_ps:
                o_input_feat = all[p]['traj']
                o_frame_mask = all[p]['traj_mask']
                skeletonft = o_input_feat
                input_feat.append(skeletonft)
                frame_mask.append(o_frame_mask)
            input_feat = np.array(input_feat)
            frame_mask = np.array(frame_mask)

            eoi = []
            for i in range(1,len(sampled_ps)):
                for j in range(i):
                    eoi.append([i,j])
            eoi = np.array(eoi,np.int).T
            print("before filter",eoi.shape[1])
            eoi,_ = filter_impossible_edges(np.transpose(input_feat,[0,2,1]),eoi,frame_mask,MIN_DIS_THR,min_time_overlap=config_param.TEST_MIN_TIMEOVERLAP_THR)
            print("after filter",eoi.shape[1])
            frame_mask = frame_mask.T

            input_feat = torch.from_numpy(input_feat).float().cuda()
            frame_mask = torch.from_numpy(frame_mask).float().cuda()
            eoicd = torch.from_numpy(eoi).long().cuda()
            res = np.zeros([eoi.shape[1]])
            
            if cfg.appearance:
                
                if len(sampled_ps) < 300:
                    x_app = load_person_appearance(val_set_name[id],sampled_ps).cuda()
                    
                    pred = torch.sigmoid(traj_net(input_feat,eoicd,frame_mask,x_app = x_app)).cpu()
                else:
                    res = []
                    for i in range(0,len(sampled_ps),300):
                        x_app = load_person_appearance(val_set_name[id],sampled_ps[i:i+300]).cuda()
                        # sxapp = x_app[i:i+300].cuda()
                        res.append(traj_net.occlusion_att(x_app,frame_mask[:,i:i+300]))
                        del x_app
                    res = torch.cat(res)
                    pred = torch.sigmoid(traj_net(input_feat,eoicd,frame_mask,x_app = res,is_app_attended = True)).cpu()
            else:
                x_app = None
                pred = torch.sigmoid(traj_net(input_feat,eoicd,frame_mask,x_app = x_app)).cpu()

            similarity_dis_matrix = np.zeros([len(sampled_ps),len(sampled_ps)])
            for i in range(eoi.shape[1]):
                a,b = eoi[0,i],eoi[1,i]
                similarity_dis_matrix[a,b] = similarity_dis_matrix[b,a] = pred[i]
            #print('1',person_label)
            #print('2',similarity_dis_matrix)
            np.save(os.path.join(".", "test_result", cfg.taskname, "test_scene_{}".format(id), "person_label.npy"),
                    person_label)
            np.save(
                os.path.join(".", "test_result", cfg.taskname, "test_scene_{}".format(id), "l2_distance_matrix.npy"),
                similarity_dis_matrix)



def read_all_result_and_label_propagation():
    # test_option=["vanilla_traj","traj_visual_random","traj_visual_uncertainty"]
    test_option=["vanilla_traj"]
    results={k:[] for k in test_option}
    SCORE_TH=config_param.SCORE_TH
    LABEL_PROPAGATION_MAX_SIZE=config_param.LABEL_PROPAGATION_MAX_SIZE
    LABEL_PROPAGATION_STEP=config_param.LABEL_PROPAGATION_STEP
    LABEL_PROPAGATION_POOL=config_param.LABEL_PROPAGATION_POOL
    GROUP_EVAL_METRIX='half'
    
    for scene_id, scene in enumerate(val_set):
        # visual cmp1: vanilla traj
        if "vanilla_traj" in test_option:
            rawsc = np.load(os.path.join(".", "test_result", cfg.taskname, "test_scene_{}".format(scene_id), "l2_distance_matrix.npy"))
            label = np.load(os.path.join(".", "test_result", cfg.taskname, "test_scene_{}".format(scene_id), "person_label.npy"),allow_pickle=True)
            margin_mask = rawsc > SCORE_TH
            # print("Max scores",rawsc.max(),'min scores',rawsc.min())
            edges = []
            scores = []
            t1=time.time()
            for i in range(rawsc.shape[0]):
                for j in range(i):
                    if margin_mask[i, j] == True:
                        edges.append((i, j))
                        scores.append(rawsc[i, j])
            scores = np.array(scores)
                        
            assert scores.max() <= 1
            assert scores.min() >= 0
            clusters = graph_propagation(edges, scores, max_sz=LABEL_PROPAGATION_MAX_SIZE, step=LABEL_PROPAGATION_STEP, pool=LABEL_PROPAGATION_POOL)
            gppsn = 0
            group = []
            for ci, c in enumerate(clusters):
                group.append([])
                for xid in c:
                    group[ci].append(xid.name)
                gppsn += len(group[ci])

            gt_group = []
            t_label_set=set()
            for i in label:
                t_label_set=t_label_set.union(set(i))
            for i in t_label_set:
                tmp_new_group = []
                for id, j in enumerate(label, 0):
                    if len(j)<1 :
                        continue
                    for k in j:
                        if k == i and i != -1:
                            tmp_new_group.append(id)
                if len(tmp_new_group) > 0:
                    gt_group.append(tmp_new_group)
            t2 = time.time()
            
            print(group)
            print('gt',gt_group)
            
            tmp = group_eval(group, gt_group, GROUP_EVAL_METRIX)
            results["vanilla_traj"].append(tmp)
            
        
    
    def show_method_avg_performance(mth):
        rs = results[mth]
        avg = [0,0,0]
        for item in rs:
            avg[0] += item[3]
            avg[1] += item[4]
            avg[2] += item[5]
        print('--- Method',mth,'detailed performance (Precision, Recall, F1) :')
        #print(rs)
        TP,FP,FN = avg
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = (2 * precision * recall) / (precision + recall + 1e-8)
        print('average pr,rc,f1 :',precision,recall,f1)
        return precision,recall,f1

    for mth in test_option:
        prc,rcl,f1 = show_method_avg_performance(mth)

    return prc,rcl,f1,t2-t1


def cluster_acc(y_true, y_pred):
    ypd,ygt = {},{}
    for i,t in enumerate(y_true):
        for x in t:
            ygt[x] = i
    for i,t in enumerate(y_pred):
        for x in t:
            ypd[x] = i
    D = max(len(y_pred), len(y_true))
    w = np.zeros((D, D))
    for i in range(len(y_pred)):
        w[ypd[i], ygt[i]] += 1
    from scipy.optimize import linear_sum_assignment
    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return sum([w[i, j] for i, j in ind]) , y_pred.size
    
def count_param(model):
    param_count = 0
    for param in model.parameters():
      param_count+=param.view(-1).size()[0]
    return param_count

if __name__ == '__main__':
    ds_split = 'test'
    READ_DATA_PATH_PERSON = f"./dataset/panda_annotation/video_annos/"
    READ_DATA_PATH_GROUP = f'./dataset/panda_annotation/grouping_annotation_train/'
    
    # Full_raw_release = "<FILL_WITH_RIGHT_PATH>"
    # cfg
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--taskname', type=str, default='oursmth')
    # dataset
    parser.add_argument('--dataset_root', type=int, default=6, help="")
    # traj_net
    parser.add_argument("--refresh_ana", type=int, default=1, help="")
    # test
    parser.add_argument("--loading_tracjory_net", type=str, default="", help="")
    parser.add_argument("--n_negtive_sample_group", type=int, default=1, help="")
    parser.add_argument("--n_positive_sample_group", type=int, default=50, help="")
    parser.add_argument("--uncertainty_inference", type=int, default=10, help="")
    parser.add_argument("--add_edge", type=int, default=10, help="")
    parser.add_argument("--pool", type=str, default="avg", help="")
    parser.add_argument('--gpu', type=str,default='3', help='gpuidx')
    parser.add_argument('--appearance', action="store_true",help='Use appearance information')
    parser.add_argument('--skeleton', action="store_true",help='Use skeleton information')
    parser.add_argument('--onlyskeleton',action='store_true',help='only use skeleton')
    # parser.add_argument("--save", type=str, default='traj_net', help="")
    cfg = parser.parse_args()

    print("Using GPU",cfg.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"]=cfg.gpu
    if cfg.skeleton:
        input_dim = 37
    elif cfg.onlyskeleton:
        input_dim = 32     
    else:
        input_dim = 5

    # tensorboard
    # writer = SummaryWriter(comment="test" + cfg.taskname)

    if cfg.refresh_ana == 1:
        # dataset
        val_set = []
        val_set_name = []
        with open(f"./dataset/panda_annotation/group_{ds_split}.txt") as f:
            print("start gen pkl")
            tmp_all = f.readlines()
            for i in tqdm(range(0, len(tmp_all))):
                val_set.append(
                    get_person_traj_group_list(
                        os.path.join(READ_DATA_PATH_PERSON,tmp_all[i].strip(),"tracks_new.json"),
                        os.path.join(READ_DATA_PATH_GROUP,tmp_all[i].strip()+'.json')
                    )
                )
                val_set_name.append(tmp_all[i].strip("\n"))
            print("finish gen pkl")
        with open("{}_{}ana.pkl".format(ds_split,cfg.taskname), "wb") as f:
            pickle.dump([val_set,val_set_name], f)
    else:
        with open("{}_{}ana.pkl".format(ds_split,cfg.taskname), "rb") as f:
            tmp = pickle.load(f)
            val_set = tmp[0]
            val_set_name = tmp[1]
            tmp = None
        print("loaded cached annotation file...")
    traj_net = OverallModel1(input_dim,model_config).cuda()
    
    if cfg.loading_tracjory_net != "":
        traj_net.load_state_dict(torch.load(cfg.loading_tracjory_net))
    for id, scene in enumerate(val_set):
        if not os.path.exists(os.path.join(".", "test_result", cfg.taskname, "test_scene_{}".format(id))):
            os.makedirs(os.path.join(".", "test_result", cfg.taskname, "test_scene_{}".format(id)))
    res=count_param(traj_net)
    test_group_embedding()
    #
    read_all_result_and_label_propagation()
    print("Enumerate Scores")
    score = 0.9
    
    s = score 
    print("Score",s)
    config_param.SCORE_TH = s
    pr,rc,f1,ttime= read_all_result_and_label_propagation()
    print('precision',pr,'recall',rc,'f1',f1)

    