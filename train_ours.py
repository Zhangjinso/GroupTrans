import torch
import os, sys, json, argparse
import random
from random import choice, sample
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import matplotlib.pyplot as plt
import numpy as np
import pickle
from predata import person_analysis,group_analysis
# from tqdm import tqdm
from torch.optim import lr_scheduler
import copy
import oursmodel
import torch.nn as nn
import torch.nn.functional as F
import config_param 
import tqdm
import torch.utils.data as tdata

APP_PATH = './dataset/panda_annotation/video_annos/'
def get_person_traj_group_list(person_json_path, group_json_path):
    # paring raw input
    rt = {}
    persons_file = person_analysis(person_json_path)
    persons_file_save_for_rt = copy.deepcopy(persons_file)
    tmp_persons_file = {}
    for person in persons_file:
        tmp_persons_file[int(person["idx"])]=person
    persons_file=tmp_persons_file

    group_file = group_analysis(group_json_path)
    group_file_save_for_rt = copy.deepcopy(group_file)
    tmp_group_file = {}
    for group in group_file:
        tmp_group_file[int(group["group_id"])] = group
    group_file = tmp_group_file


    # find max frame count
    max_frame_count = 0
    min_frame_count = 9999
    for person_id,person in persons_file.items():
        for i,frame in enumerate(person["frames list"]):
            max_frame_count = max(max_frame_count, frame['frame_id'])
            min_frame_count = min(min_frame_count, frame['frame_id'])
            # if frame['frame_id'] == 0:
            #     print(person_id,i,'frame id 0')
    # print(min_frame_count,max_frame_count)
    max_frame_count += 1
    empty_feature = [0,0,0,0,0]
    # create attribute dictionary for each person
    for person_id, person in persons_file.items():
        tmp_person = {"traj": [empty_feature for i in range(max_frame_count)],
                      "traj_mask" : [0 for i in range(max_frame_count)],
                      "group": [], "same_group_persons": [], "diff_group": [],
                      "different_group_persons": []}
        for frame_id in range(len(person["frames list"])):
            #print(person["face orientation"][frame_id])
            fid = person["frames list"][frame_id]['frame_id']
            assert fid >= 0
            tmp_person["traj"][fid] = \
                [
                    float(person["frames list"][frame_id]['rect']['tl']['y']),
                    float(person["frames list"][frame_id]['rect']['tl']['x']),
                    float(person["frames list"][frame_id]['rect']['br']['y']),
                    float(person["frames list"][frame_id]['rect']['br']['x']),
                    int(person["face orientation"][frame_id])
                ]
            tmp_person['traj_mask'][fid] = 1
        # pad one frame
        for frame_id in range(1,max_frame_count-1):
            if tmp_person['traj_mask'][frame_id] != 1:
                continue
            if tmp_person['traj_mask'][frame_id - 1] == 0:
                tmp_person['traj'][frame_id - 1] = tmp_person['traj'][frame_id]
            if tmp_person['traj_mask'][frame_id + 1] == 0:
                tmp_person['traj'][frame_id + 1] = tmp_person['traj'][frame_id]
        
        tmp_person['traj'] = np.array(tmp_person['traj']).T
        tmp_person['traj_mask'] = np.array(tmp_person['traj_mask'])
        rt[person_id] = tmp_person

    # add group attribute excepting single person
    for group_id,group in group_file.items():
        if group['group_type'] != 'Single person':
            for group_member in group["person"]:
                person_id = group_member["idx"]
                if person_id in rt.keys():
                    rt[person_id]["group"].append(int(group["group_id"]))

    for person_id, person in rt.items():

        # add person who share the same group as anchor person
        for pos_group in person["group"]:
            for positive_person in group_file[int(pos_group)]["person"]:
                pid = int(positive_person['idx'])
                if pid in rt.keys():
                    rt[person_id]["same_group_persons"].append(pid)

        # add person who is in the different group of anchor person
        for neg_group_id, neg_group in group_file.items():
            if int(neg_group["group_id"]) in person["group"]:
                continue
            tmp_neg_group = {
                "group_id": int(neg_group["group_id"]),
                "person_id": []
            }
            for negtive_person in group_file[int(neg_group["group_id"])]["person"]:
                npid = int(negtive_person['idx'])
                if npid in rt.keys() and npid not in rt[person_id]["same_group_persons"]:
                    tmp_neg_group["person_id"].append(negtive_person['idx'])
            rt[person_id]["diff_group"].append(tmp_neg_group)

    return rt, persons_file_save_for_rt, group_file_save_for_rt


def init_weights(m):
    for name, param in m.named_parameters():
        torch.nn.init.uniform_(param.data, -0.08, 0.08)



class SceneDataset:

    def __init__(self, scene_list, scene_name_list) -> None:
        self.names = scene_name_list
        self.scenes = scene_list
        self.gpmap = [{} for i in range(len(scene_list))]

    def __getitem__(self,oidx):
        dsid,index,gp = oidx

        if cfg.appearance:
            path = f'{APP_PATH}{self.names[dsid]}/ann_pkl/{self.names[dsid]}_pid_{index+1}.pkl'
            ft = np.load(path,allow_pickle=True)
            ft = convert_feature(ft)
            ft = ft.reshape([ft.shape[0],-1])
        else:
            ft = 0
        trajft = self.scenes[dsid][0][index]['traj']
        trajmask = self.scenes[dsid][0][index]['traj_mask']
        return index,gp,ft,trajft,trajmask
    
    def __len__(self):
        return 9999999

class MyTrainBatchSampler(tdata.Sampler):

    def __init__(self, ds , gpcount):
        self.gpcount = gpcount
        self.ds = ds
        self.perm = None
        self.pidx = len(ds.scenes)

    def __iter__(self):
        while True:
            if self.pidx >= len(self.ds.scenes):
                self.perm = np.random.permutation(len(self.ds.scenes))
                self.pidx = 0
            dsid = self.perm[self.pidx]
            all = self.ds.scenes[dsid][0]
            all_ps = list(all.keys())
            sampled_ps = []
            sampled_gp = []
            itr = 0
            while itr < self.gpcount:
                ps = random.sample(all_ps, 1)[0]
                # print(all[ps])
                gpps = all[ps]["same_group_persons"]
                if ps not in gpps:
                    gpps = gpps + [ps]
                gpps = [x for x in gpps if x not in sampled_ps]
                if len(gpps) == 0:
                    continue
                sampled_ps += gpps
                sampled_gp += [itr for i in gpps]
                all_ps = [x for x in all_ps if x not in gpps]
                # print(len(all_ps))
                itr += 1

            # self.ds.gpmap[dsid].update({sampled_ps[i] : sampled_gp[i] for i in range(len(sampled_ps))})
            
            self.pidx += 1
            yield [(dsid,sampled_ps[i],sampled_gp[i]) for i in range(len(sampled_ps))]


    def __len__(self):
        return 99999999
 
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


def load_person_appearance(scene, pids):
    # scene : str
    # pids  : list, each element be a person id
    # return : Tensor containing persons appearance features
    # path example : ./dataset/panda_annotation/video_annos/01_University_Canteen/ann_pkl/01_University_Canteen_pid_99.pkl
    res = []
    for p in tqdm.tqdm(pids):
        path = f'{APP_PATH}{scene}/ann_pkl/{scene}_pid_{p+1}.pkl'
        ft = np.load(path,allow_pickle=True)
        ft = convert_feature(ft)
        ft = ft.reshape([ft.shape[0],-1])
        res.append(ft)
    res = torch.from_numpy(np.stack(res)).float()
    return res
        

def evaluate_group_embedding():
    SampleNumGP = 40

    traj_net.eval()
    corr,tot = 0,0
    tp,tpa = 0,0
    rt_loss = 0
    with torch.no_grad():
        for id, scene in enumerate(val_set):
            all = scene[0]
            all_ps = list(all.keys())
            sampled_ps = []
            sampled_gp = []
            itr = 0
            while len(all_ps) > 0:
                ps = random.sample(all_ps, 1)[0]
                gpps = all[ps]["same_group_persons"]
                if ps not in gpps:
                    gpps = gpps + [ps]
                gpps = [x for x in gpps if x not in sampled_ps]
                if len(gpps) == 0:
                    continue
                sampled_ps += gpps
                sampled_gp += [itr for i in gpps]
                all_ps = [x for x in all_ps if x not in gpps]
                itr += 1
            input_feat = []
            frame_mask = []
            for p in sampled_ps:
                input_feat.append(all[p]['traj'])
                frame_mask.append(all[p]['traj_mask'])
            input_feat = np.array(input_feat)
            frame_mask = np.array(frame_mask).T

            if cfg.appearance:
                x_app = load_person_appearance(val_scene_names[id],sampled_ps).cuda()
            else:
                x_app = None

            eoi = []
            label = []
            for i in range(1,len(sampled_ps)):
                for j in range(i):
                    eoi.append([i,j])
                    label.append((sampled_gp[i] == sampled_gp[j]) + 0)
            eoi = np.array(eoi,np.int).T
            label = np.array(label,np.int)

            input_feat = torch.from_numpy(input_feat).float().cuda()
            frame_mask = torch.from_numpy(frame_mask).float().cuda()
            eoi = torch.from_numpy(eoi).long().cuda()
            label = torch.from_numpy(label).float()

            pred = traj_net(input_feat,eoi,frame_mask,x_app = x_app).cpu()
            prednp = pred.numpy()
            labelnp = label.numpy()
            corr += np.sum(((prednp > 0) == (labelnp > 0.5)) + 0)
            tot += pred.shape[0]

            pncond = labelnp > 0.5
            tp += np.sum((prednp > 0)[pncond])
            tpa += np.sum(pncond)

            loss = criterion(pred,label)
            rt_loss += loss.item()

    return rt_loss,corr / tot,tp / tpa


def train_group_embedding():
    MIN_DIS_THR = config_param.TRAIN_MIN_DIS_THR
    mini_batchnum = config_param.TRAIN_SceneItrNum
    
    pos_edge, all_edge = 0,0

    traj_net.train()
   
    with tqdm.tqdm(total = mini_batchnum * len(train_set)) as pbar:
        for ep in range(len(train_set)):
            avgloss = 0

            tranj_optimizer.zero_grad()
            for stp in range(mini_batchnum):
                sampled_ps,sampled_gp,x_app,input_feat,frame_mask = next(train_itr)

                if not cfg.appearance:
                    #x_app = load_person_appearance(train_scene_names[id],sampled_ps).cuda()
                    x_app = None
                else:
                    x_app = x_app.cuda().float()

                eoi = []
                label = []
                for i in range(1,len(sampled_ps)):
                    for j in range(i):
                        eoi.append([i,j])
                        label.append((sampled_gp[i] == sampled_gp[j]) + 0)
                eoi = np.array(eoi,np.int).T
                label = np.array(label,np.int)
                if model_config['filter_edges']:

                    emsk = oursmodel.filter_impossible_edges(np.transpose(input_feat.numpy(),[0,2,1]),eoi,frame_mask.numpy(),MIN_DIS_THR,cond_only = True,min_time_overlap=config_param.TRAIN_MIN_TIMEOVERLAP_THR)
                    frame_mask = torch.transpose(frame_mask,0,1)
                    emsk = (label > 0) | emsk
                    eoi = eoi[:,emsk]
                    label = label[emsk]
                else:
                    frame_mask = torch.transpose(frame_mask,0,1)

                pos_edge += np.sum(label)
                all_edge += label.shape[0]

                input_feat = input_feat.float().cuda()
                frame_mask = frame_mask.float().cuda()
                eoi = torch.from_numpy(eoi).long().cuda()
                label = torch.from_numpy(label).float().cuda()

                pred = traj_net(input_feat,eoi,frame_mask,x_app = x_app)

                positive_vals = label.sum()
                if positive_vals:
                    pos_weight = (label.shape[0] - positive_vals) / positive_vals
                else: # If there are no positives labels, avoid dividing by zero
                    pos_weight = torch.zeros([1]).cuda()
                # print(pos_weight)
                loss = F.binary_cross_entropy_with_logits(pred, label, pos_weight= pos_weight) / mini_batchnum
                avgloss += loss.item()
                loss.backward()
                pbar.update(1)

            tranj_optimizer.step()


    avgloss /= mini_batchnum * len(train_set)
    print("---AvgLoss",avgloss,'PosLabelRatio',pos_edge / all_edge)


model_config = config_param.model_config

if __name__ == '__main__':
    READ_DATA_PATH_PERSON = "./dataset/panda_annotation/video_annos/"
    READ_DATA_PATH_GROUP = './dataset/panda_annotation/grouping_annotation_train/'
    # cfg
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--taskname', type=str, default='localtest')
    # dataset
    parser.add_argument('--dataset_root', type=int, default=6, help="")
    # traj_net0

    parser.add_argument('--traj_net_with_face_orientation', type=int, default=5,
                        help="5 for non face origentation, 6 for with oritation")

    # training
    parser.add_argument('--num_epoch', type=int, default=200, help="")
    parser.add_argument('--resume', type=int, default=0, help="")
    parser.add_argument("--loading_tracjory_net", type=str, default="", help="pretrained model")
    parser.add_argument("--saving_tracjory_net", type=str, default="./ckpt1", help="")
    parser.add_argument("--train_embedding", type=int, default=1, help="")
    parser.add_argument("--save_every_n_epoch", type=int, default=2, help="")
    parser.add_argument("--refresh_ana", type=int, default=0, help="")
    parser.add_argument('--gpu', type=str,default='2', help='gpuidx')
    parser.add_argument('--appearance', action="store_true",help='Use appearance information')
    cfg = parser.parse_args()

    print("Using GPU",cfg.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"]=cfg.gpu

    input_dim = 5

    # mkdir
    if not os.path.exists(os.path.join(cfg.saving_tracjory_net, cfg.taskname)):
        os.makedirs(os.path.join(cfg.saving_tracjory_net, cfg.taskname))

    if cfg.refresh_ana == 1:
        # dataset
        train_set = []
        train_scene_names = []
        val_set = []
        val_scene_names = []
        with open("./dataset/panda_annotation/group_train.txt") as f:
            print("start gen trainpkl")
            tmp_all = f.readlines()
            for i in tqdm.tqdm(range(0, len(tmp_all))):
                if i<=8:
                    train_set.append(
                        get_person_traj_group_list(
                            os.path.join(READ_DATA_PATH_PERSON,tmp_all[i].replace("\n", ""),"tracks_new.json"),
                            os.path.join(READ_DATA_PATH_GROUP,tmp_all[i].strip("\n")+'.json'),
                            )
                        )
                else:
                    train_set.append(
                        get_person_traj_group_list(
                            os.path.join(READ_DATA_PATH_PERSON,tmp_all[i],"tracks_new.json"),
                            os.path.join(READ_DATA_PATH_GROUP,tmp_all[i])+'.json'),
                            )
                train_scene_names.append(tmp_all[i].strip("\n"))
                        
        with open("./dataset/panda_annotation/group_test.txt") as f:
            print("start gen testpkl")
            tmp_all = f.readlines()
            for i in tqdm.tqdm(range(0, len(tmp_all))):
                val_set.append(
                    get_person_traj_group_list(
                        os.path.join(READ_DATA_PATH_PERSON,tmp_all[i].replace("\n", ""),"tracks_new.json"),
                        os.path.join(READ_DATA_PATH_GROUP,tmp_all[i].strip("\n")+'.json'),
                    )
                )
                val_scene_names.append(tmp_all[i].strip("\n"))
            
            print("finish gen testpkl")
        with open("ana.pkl", "wb") as f:
            pickle.dump([train_set, val_set, train_scene_names, val_scene_names], f)
    else:
        with open("ana.pkl", "rb") as f:
            tmp = pickle.load(f)
            train_set = tmp[0]
            val_set = tmp[1]
            train_scene_names = tmp[2]
            val_scene_names = tmp[3]
            tmp = None

    train_ds = SceneDataset(train_set,train_scene_names)
    val_ds_list = SceneDataset(val_set,val_scene_names)
    train_batch_sampler = MyTrainBatchSampler(train_ds,config_param.TRAIN_SampleNumGP)
    train_dataloader = tdata.DataLoader(train_ds,batch_sampler=train_batch_sampler,num_workers=4,pin_memory=True)

    train_itr = iter(train_dataloader)
    
    traj_net = oursmodel.OverallModel1(input_dim,model_config).cuda()
    if cfg.loading_tracjory_net != "":
        traj_net.load_state_dict(torch.load(cfg.loading_tracjory_net))
    criterion = nn.BCEWithLogitsLoss().cuda()
    # optimizer
    tranj_optimizer = torch.optim.SGD(traj_net.parameters(),lr = config_param.lr,weight_decay=config_param.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(tranj_optimizer, config_param.milestones ,gamma= config_param.lr_decay)
    cfg.__setattr__("current_step", 0)
    for i in range(cfg.num_epoch):
        cfg.__setattr__("current_epoch", i)
        print("Epoch",i+1)
        if i % cfg.save_every_n_epoch == 0:
            torch.save(traj_net.state_dict(),
                       os.path.join(cfg.saving_tracjory_net, cfg.taskname, "traj_net_{}.pt".format(str(i + 1))))
        train_group_embedding()
