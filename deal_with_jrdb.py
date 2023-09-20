import numpy as np
import json
import os

label_path = './dataset/JRDB/train_labels/labels_2d_stitched/'
img_path = './dataset/JRDB/images/image_stitched/'

train_list= '''tressider-2019-03-16_0
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
# train_scene:
# tressider-2019-03-16_0
# svl-meeting-gates-2-2019-04-08_1
# svl-meeting-gates-2-2019-04-08_0
# stlc-111-2019-04-19_0
# packard-poster-session-2019-03-20_2
# packard-poster-session-2019-03-20_1
# packard-poster-session-2019-03-20_0
# memorial-court-2019-03-16_0
# jordan-hall-2019-04-22_0
# huang-lane-2019-02-12_0
# huang-basement-2019-01-25_0
# hewlett-packard-intersection-2019-01-24_0
# gates-to-clark-2019-02-28_1
# gates-basement-elevators-2019-01-17_1
# gates-159-group-meeting-2019-04-03_0
# forbes-cafe-2019-01-22_0
# cubberly-auditorium-2019-04-22_0
# clark-center-intersection-2019-02-28_0
# clark-center-2019-02-28_0
# bytes-cafe-2019-02-07_0
train_list = train_list.split('\n')

dir_name_train = './dataset/JRDB/images/image_stitched_key/'
if not os.path.isdir(dir_name_train):
    os.makedirs(dir_name_train)
dir_name_key = './dataset/JRDB/train_labels/labels_2d_stitched_key/'
if not os.path.isdir(dir_name_key):
    os.makedirs(dir_name_key)
for ith,f_train in enumerate(train_list):
    imgs_folder = img_path+f_train
    label_file = label_path + f_train + '.json'
    #read labels
    dir_name = dir_name_train+f_train
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    file_name = dir_name_key + f_train + '.json'
    new_ana = {'labels':{}}
    with open(label_file, 'r') as f:
        data = json.load(f)
        ls = data['labels']
        frames = len(ls)
        for k in ls: ls[k] = (k,ls[k])
        frame_sq = [ls[k] for k in ls]
        frame_sq.sort(key = lambda k: k[0])
        frame_sq = [frame_sq[i] for i in range(15,len(frame_sq),15)]
        for i,t_img in enumerate(frame_sq):
            readimg = imgs_folder+'/'+t_img[0]
            strscene = '%02d' % int(ith+1)
            strframe = '%03d' % (i+1)
            os.system(f'cp {readimg} {dir_name}/SEQ_{strscene}_{strframe}.jpg')
            if t_img[0] not in new_ana['labels'].keys():
                tstr = strframe+'.jpg'
                new_ana['labels'][tstr] = t_img[1]
            json_data = json.dumps(new_ana)
            with open(f'{file_name}', 'w') as f_six:
                f_six.write(json_data)





label_path='./dataset/JRDB/train_labels/labels_2d_stitched_key/'
img_path = './dataset/JRDB/images/image_stitched_key/'

storepath = './dataset/jrdb_annotation/video_annos/'
w=3760.0
h=480.0
for ith,imgfolder in enumerate(train_list):
    res = {}
    dir_name = storepath + imgfolder
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    t_anapath = label_path+imgfolder+'.json'
    with open(t_anapath,'r') as load_f:
        data = json.load(load_f)
        lbs = data['labels']
        for frame in lbs:
            fmid = int(frame.split('.')[0])
            for ps in lbs[frame]:
                psid = int(ps["label_id"].split(':')[1])
                loclist = ps['box']
                x1=(loclist[0])/w
                y1=(loclist[1])/h
                x2=(loclist[0]+loclist[2])/w
                y2=(loclist[1]+loclist[3])/h
                
                occ = ps["attributes"]["occlusion"]
                frame_info = {
                "frame id": fmid,
                "rect": {
                    "tl": {
                        "y": y1,
                        "x": x1
                    },
                    "br": {
                        "y": y2,
                        "x": x2
                    }
                },
                "face orientation": " ",
                "occlusion": occ
            }
                if psid not in res.keys():
                    res[psid]={"track id":psid,
                               "frames":[frame_info]}
                else:
                    res[psid]['frames'].append(frame_info)
    ress = []
    for _ in res:
        ress.append(res[_])
    json_data = json.dumps(ress)
    final_name = dir_name+'/tracks_new.json'
    with open(final_name, 'w') as f_six:
        f_six.write(json_data)

storepath = './dataset/jrdb_annotation/grouping_annotation_train/'
if not os.path.exists(storepath):
    os.makedirs(storepath)
for ith,imgfolder in enumerate(train_list):
    res = {}
    t_anapath = label_path+imgfolder+'.json'
    with open(t_anapath,'r') as load_f:
        data = json.load(load_f)
        lbs = data['labels']
        pslist = []
        for frame in lbs:
            fmid = int(frame.split('.')[0])
            for ps in lbs[frame]:
                psid = int(ps["label_id"].split(':')[1])
                groupid = ps['social_group']['cluster_ID']
                psinfo = {
                                "idx": psid,
                                "gender": "useless",
                                "age": "useless"
                            }
                if groupid not in res.keys():
                    res[groupid]={
                        "group_id":groupid,
                        "group_type": "useless",
                        "person": [psinfo]
                    }
                    pslist.append(psid)
                elif psid not in pslist:
                    res[groupid]['person'].append(psinfo)
                    pslist.append(psid)
    ress = []
    for _ in res:
        ress.append(res[_])
    json_data = json.dumps(ress)
    final_name = storepath+imgfolder+'.json'
    with open(final_name, 'w') as f_six:
        f_six.write(json_data)
            