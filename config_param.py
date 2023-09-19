# # No spatial SOTA parameters
# lr = 0.2
# milestones = [50,100,150]
# lr_decay = 0.2
# TRAIN_SampleNumGP = 16
# TRAIN_SceneItrNum = 10
# TRAIN_MIN_DIS_THR=0.1
# TRAIN_MIN_TIMEOVERLAP_THR=0
# weight_decay = 0

# model_config = {
#     'appearance_feature_dim' : 0,  # 0 represent disabled appearance feature input
#     'temporal_hidden_dim1' : 128,
#     'temporal_hidden_layernum1' : 2,
#     'temporal_hidden_dim2' : 128,
#     'temporal_hidden_layernum2' : 2,
#     'spatial_transformer_dim1' : 128,
#     'spatial_transformer_dim2' : 128,
#     'prediction_embedding_dim' : 0,
#     'prediction_reduce_time' : 'mean',
#     'use_occusion_attention' : False,
#     'shortcut2predhead' : True,
#     'filter_edges':False
# }

# # test parameters
# TEST_MIN_DIS_THR = 0.2
# SCORE_TH=0.95
# LABEL_PROPAGATION_MAX_SIZE=6
# LABEL_PROPAGATION_STEP=1e-4
# LABEL_PROPAGATION_POOL='avg'

# TEST_MIN_TIMEOVERLAP_THR=0.3


#OverallModel
# lr = 0.10
# milestones = [50,100,150]
# lr_decay = 0.2
# TRAIN_SampleNumGP = 8
# TRAIN_SceneItrNum = 20
# TRAIN_MIN_DIS_THR=0.1

# model_config = {
#     'appearance_feature_dim' : 512,  # 0 represent disabled appearance feature input
#     'temporal_hidden_dim1' : 16,
#     'temporal_hidden_layernum1' : 2,
#     'temporal_hidden_dim2' : 32,
#     'temporal_hidden_layernum2' : 2,
#     'spatial_transformer_dim1' : 128,
#     'spatial_transformer_dim2' : 128,
#     'prediction_embedding_dim' : 64,
#     'prediction_reduce_time' : 'mean',
#     'use_occusion_attention' : True
# }

# # test parameters
# TEST_MIN_DIS_THR = 0.2
# SCORE_TH=0.65
# LABEL_PROPAGATION_MAX_SIZE=6
# LABEL_PROPAGATION_STEP=1e-4
# LABEL_PROPAGATION_POOL='avg'

#OverallModel2
# lr = 0.10
# milestones = [50,100,150]
# lr_decay = 0.2
# TRAIN_SampleNumGP = 8
# TRAIN_SceneItrNum = 20
# TRAIN_MIN_DIS_THR=0.1

# model_config = {
#     'appearance_feature_dim' : 512,  # 0 represent disabled appearance feature input
    
#     'temporal_hidden_dim1' : 16,#16+512
#     'temporal_hidden_layernum1' : 2,
#     'temporal_hidden_dim2' : 64,#64
#     'temporal_hidden_layernum2' : 2,
#     'spatial_transformer_dim1' : 128,
#     'spatial_transformer_dim2' : 128,
#     'prediction_embedding_dim' : 64,
#     'prediction_reduce_time' : 'mean',
#     'use_occusion_attention' : True
# }

# # test parameters
# TEST_MIN_DIS_THR = 0.2
# SCORE_TH=0.55
# LABEL_PROPAGATION_MAX_SIZE=6
# LABEL_PROPAGATION_STEP=1e-4
# LABEL_PROPAGATION_POOL='avg'

# OverallModel1 && #OverallModelADD
# lr = 0.10
# milestones = [50,100,150]
# lr_decay = 0.2
# TRAIN_SampleNumGP = 8
# TRAIN_SceneItrNum = 20
# TRAIN_MIN_DIS_THR=0.1

# model_config = {
#     'appearance_feature_dim' : 512,  # 0 represent disabled appearance feature input
    
#     'temporal_hidden_dim1' : 512,
#     'temporal_hidden_middle_dim1':64,
#     'temporal_hidden_layernum1' : 2,
#     'temporal_hidden_dim2' : 128,
#     'temporal_hidden_middle_dim2':64,
#     'temporal_hidden_layernum2' : 2,
#     'spatial_transformer_dim1' : 256,
#     'spatial_transformer_dim2' : 256,
#     'prediction_embedding_dim' : 64,
#     'prediction_reduce_time' : 'mean',
#     'use_occusion_attention' : True
# }

# # test parameters
# TEST_MIN_DIS_THR = 0.2
# SCORE_TH=0.6
# LABEL_PROPAGATION_MAX_SIZE=6
# LABEL_PROPAGATION_STEP=1e-4
# LABEL_PROPAGATION_POOL='avg'







# tuning appearance
lr = 0.10
milestones = [50,100,150]
lr_decay = 0.2
weight_decay = 0
TRAIN_SampleNumGP = 8
TRAIN_SceneItrNum = 20
TRAIN_MIN_DIS_THR=0.1
TRAIN_MIN_TIMEOVERLAP_THR=0

model_config = {
    'appearance_feature_dim' : 512,  # 0 represent disabled appearance feature input
    
    'temporal_hidden_dim1' : 128,
    'temporal_hidden_middle_dim1':64,
    'temporal_hidden_layernum1' : 2,
    'temporal_hidden_dim2' : 128,
    'temporal_hidden_middle_dim2':64,
    'temporal_hidden_layernum2' : 2,
    'spatial_transformer_dim1' : 128,
    'spatial_transformer_dim2' : 128,
    'prediction_embedding_dim' : -1,
    'prediction_reduce_time' : 'mean',
    'use_occusion_attention' : True,
    'shortcut2predhead' : True,
    'filter_edges':True
}

# test parameters
TEST_MIN_DIS_THR = 0.2
TEST_MIN_TIMEOVERLAP_THR=0.3
SCORE_TH=0.6
LABEL_PROPAGATION_MAX_SIZE=6
LABEL_PROPAGATION_STEP=1e-4
LABEL_PROPAGATION_POOL='avg'

