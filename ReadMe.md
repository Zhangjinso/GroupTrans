## Code for our paper "Towards Grouping in Large Scenes with Occlusion-aware Spatio-temporal Transformers"



### Install 

```
# the code is tested on the NVIDIA 2080Ti
conda env create -f baseline.yml
conda activate baseline
```

### Dataset

1. ##### **download [PANDA](https://gigavision.cn/) and [JRDB]() datasets.**

2. **crop the original image and extract appearance feature**

   ```
   #revise the "img_path0", "save_path" and "ann_path"
   python crop.py # for panda
   python jrdb_crop.py # for jrdb
   
   cd feature_extractor
   #revise the "img_path", "save_path" and "ann_path"
   python extract_panda.py
   #revise the "img_path", "save_path" and "ann_path" and "origin_img_path"
   python extract_jrdb.py
   # change the 'APP_PATH' in train_ours.py as "save_path"
   ```

The example of our file architecture is like:

```
-dataset
	--panda_annotation
		---grouping_annotation_train
			----01_University_Canteen.json
			----02_OCT_Habour.json
			---- 
		---video_annos
			----01_university_Canteen
				-----ann_pkl  #appearance feature
				-----seqinfo.json
				-----tracks_new.json
				-----tracks.json
			----
		---group_test.txt
		---group_train.txt
```



### train
```
python train_ours.py --gpu 0 --taskname traj_with_app --appearance --refresh_ana 1
python jrdb_train_ours.py --gpu 0 --taskname traj_with_app --appearance --refresh_ana 1
```


### test
python test_ours.py --gpu 0 --taskname oursmth --loading_tracjory_net ckpt/traj_with_app/full_traj_net_199.pt --appearance 