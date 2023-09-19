import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import models
import cv2
import os
import pickle
from tqdm import tqdm
import json
from tqdm import tqdm
import numpy
os.environ["CUDA_VISIBLE_DEVICES"]='2'

img_path = './dataset/JRDB/bbox_jrdb/'
ann_path = './dataset/jrdb_annotation/video_anns/'
origin_img_path = './dataset/JRDB/images/image_stitched_key/'
save_path = '../../dataset/jrdb_annotation/jrdb_annos/'
class base_resnet(nn.Module):
    def __init__(self):
        super(base_resnet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        #self.model.load_state_dict(torch.load('./model/resnet50-19c8e357.pth'))
        #self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
 
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        #x = self.model.avgpool(x)
        return x
     
 
totensor = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5],[0.25,0.25,0.25])])   

def open_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img,(64,128))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = totensor(img)
    return img

def save_dict(data, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

ann_list= '''tressider-2019-03-16_0
svl-meeting-gates-2-2019-04-08_1
svl-meeting-gates-2-2019-04-08_0
stlc-111-2019-04-19_0
packard-poster-session-2019-03-20_2
packard-poster-session-2019-03-20_1
packard-poster-session-2019-03-20_0
memorial-court-2019-03-16_0
jordan-hall-2019-04-22_0
huang-lane-2019-02-12_0
huang-basement-2019-01-25_0
hewlett-packard-intersection-2019-01-24_0
gates-to-clark-2019-02-28_1
gates-basement-elevators-2019-01-17_1
gates-159-group-meeting-2019-04-03_0
forbes-cafe-2019-01-22_0
cubberly-auditorium-2019-04-22_0
clark-center-intersection-2019-02-28_0
clark-center-2019-02-28_0
bytes-cafe-2019-02-07_0
clark-center-2019-02-28_1
gates-ai-lab-2019-02-08_0
huang-2-2019-01-25_0
meyer-green-2019-03-16_0
nvidia-aud-2019-04-18_0
tressider-2019-03-16_1
tressider-2019-04-26_2'''
ann_list = ann_list.split('\n')

exp = base_resnet().cuda()
exp.eval()


with torch.no_grad():
    for i in range(0,len(ann_list)):
        crepath = os.path.join(save_path,ann_list[i],'/ann_pkl/')
        if not os.path.isdir(crepath):
            os.makedirs(crepath)
        img_list_path = img_path + ann_list[i]
        ann_list_path = ann_path + ann_list[i] + '/tracks_new.json'
        tmp_path = origin_img_path+ ann_list[i]
        nums_of_frames =  os.listdir(tmp_path)
        #print(img_list_path)
        with open(ann_list_path, 'r', encoding='UTF-8') as load_f:
            tracks_annotation = json.load(load_f)
           
            for track_ann in tqdm(tracks_annotation):
                person_id = track_ann['track id']
                final_pkl = []
                tmp = []
                data_list = []
                mask_list = []
                #print(len(nums_of_frames))
                padtensor = numpy.zeros([1,2048,4,2])
                for one_frame in track_ann['frames']:
                    #print(len(track_ann['frames']))
                    index_tmp = int(one_frame['frame id'])
                    tmp.append(index_tmp)
                #print(tmp)
                for j in range(1,len(nums_of_frames)+1):
                    if j in tmp:
                        oneimg_path = img_list_path +'/'+ ann_list[i]+'_frame_'+str(j)+'_'+str(person_id)+'.jpg'
                        if os.path.exists(oneimg_path):
                            oneimg = open_img(oneimg_path)
                            oneimg = oneimg.cuda()
                            c,h,w = oneimg.shape
                            oneimg = oneimg.reshape([1,c,h,w])
                            res = exp(oneimg)
                            #print(res)
                            res=res.cpu().numpy().astype(numpy.float32)
                            res = padtensor
                            data_list.append(res)
                            mask_list.append(1)
                            #print(res)
                            #exit(-1)
                        else:
                            #data_list.append(padtensor)
                            mask_list.append(0)
                            #print('kk')
                    else:#padding
                        #data_list.append(padtensor)
                        mask_list.append(0)
                        #print('jj')
                #print(len(mask_list))
                #print(len(data_list))
                final_pkl.append(mask_list)
                final_pkl.append(data_list)
                #data_list = numpy.concatenate(data_list)
                #print(data_list.shape)
                save_dict(final_pkl,f'{save_path}{ann_list[i]}/ann_pkl/{ann_list[i]}_pid_{person_id}') 