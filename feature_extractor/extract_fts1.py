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
os.environ["CUDA_VISIBLE_DEVICES"]='1'


img_path = '/mnt/8T/glf/baseline/pytorch-i3d/bbox_panda/'
ann_path = '/mnt/8T/glf/baseline/panda_annotation/video_annos/'
save_path = '../dataset/panda_annotation/zjs_video_annos/'

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

ann_list = ['01_University_Canteen','02_OCT_Habour','03_Xili_Crossroad','04_Primary_School','05_Basketball_Court'
,'06_Xinzhongguan','07_University_Campus','08_Xili_Street_1','09_Xili_Street_2','10_Huaqiangbei']

exp = base_resnet().cuda()
exp.eval()


if not os.path.exists(save_path):
    os.makedirs(save_path)
with torch.no_grad():
    for i in range(0,10):
        if not os.path.exists(os.path.join(save_path,ann_list[i])):
            os.makedirs(os.path.join(save_path,ann_list[i], 'ann_pkl'))
        img_list_path = img_path + ann_list[i]
        ann_list_path = ann_path + ann_list[i] + '/tracks_new.json'
        tmp_path = ann_path + ann_list[i] + '/' + ann_list[i]+'_frame_json'
        nums_of_frames =  os.listdir(tmp_path)
        with open(ann_list_path, 'r', encoding='UTF-8') as load_f:
            tracks_annotation = json.load(load_f)
            for track_ann in tqdm(tracks_annotation):
                person_id = track_ann['track id']
                final_pkl = []
                tmp = []
                data_list = []
                mask_list = []
                padtensor = numpy.zeros([1,2048,4,2])
                for one_frame in track_ann['frames']:
                    index_tmp = int(one_frame['frame id'])
                    tmp.append(index_tmp)
                for j in range(1,len(nums_of_frames)+1):
                    if j in tmp:
                        oneimg_path = img_list_path +'/'+ ann_list[i]+'_frame_'+str(j)+'_'+str(person_id)+'.jpg'
                        if os.path.exists(oneimg_path):
                            oneimg = open_img(oneimg_path)
                            oneimg = oneimg.cuda()
                            c,h,w = oneimg.shape
                            oneimg = oneimg.reshape([1,c,h,w])
                            res = exp(oneimg)
                            res=res.cpu().numpy().astype(numpy.float32)
                            data_list.append(res)
                            mask_list.append(1)
                        else:
                            mask_list.append(0)
                    else:
                        mask_list.append(0)
                final_pkl.append(mask_list)
                final_pkl.append(data_list)
                save_dict(final_pkl,f'{save_path}{ann_list[i]}/ann_pkl/{ann_list[i]}_pid_{person_id}') 
