import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc,os,random
import time,datetime
from tqdm import tqdm

from utils import *
from model import *
root = args.root
seed = args.seed

df =  pd.read_feather('./input/nn_series.feather')
y = pd.read_csv('./input/train_labels.csv')

f = pd.read_feather('./input/nn_all_feature.feather')
df['idx'] = df.index
series_idx = df.groupby('customer_ID',sort=False).idx.agg(['min','max'])
series_idx['feature_idx'] = np.arange(len(series_idx))
df = df.drop(['idx'],axis=1)
#                                                      min   max  feature_idx
# customer_ID
# 0000099d6bd597052cdcda90ffabf56573fe9d7c79be5fb...     0    12            0
# 00000fd6641609c6ece5454664794f0340ad84dddce9a26...    13    25            1
# 00001b22f846c82c51f6e3958ccd81970162bae8b007e80...    26    38            2
# 000041bdba6ecadd89a52d11886e8eaaec9325906c97233...    39    51            3
# 00007889e4fcd2614b6cbe7f8f3d2e5c728eca32d9eb8ad...    52    64            4
#                                                   ...   ...          ...
# 00774c693a407e828b1d9d94d3e670944dacc856aa8f410...  9932  9944          820
# 00774cde32a3a8894c5274e49092252b0d78cc49bffbb81...  9945  9957          821
# 007756f6fbf1c36b946f3a1723cdcb4a755624c433488a1...  9958  9970          822
# 00777a52c3c78548ce384dbc412025582291370ac2790c4...  9971  9983          823
# 007793144e0eeef1e29a7aa93244815328beb0d46ccbe3d...  9984  9996          824
print(f.head())
nn_config = {
    'id_name':id_name,
    'feature_name':[],
    'label_name':label_name,
    'obj_max': 1,
    'epochs': 10,
    'smoothing': 0.001,
    'clipnorm': 1,
    'patience': 100,
    'lr': 3e-4,
    'batch_size': 256,
    'folds': 5,
    'seed': seed,
    'remark': args.remark
}
# train, test, model_class, config, use_series_oof, logit=False, output_root='./output/', run_id=None
NN_train_and_predict([df,f,y,series_idx.values[:y.shape[0]]],[df,f,series_idx.values[y.shape[0]:]],Amodel,nn_config,use_series_oof=False,run_id='NN_with_series')

NN_train_and_predict([df,f,y,series_idx.values[:y.shape[0]]],[df,f,series_idx.values[y.shape[0]:]],Amodel,nn_config,use_series_oof=True,run_id='NN_with_series_and_all_feature')
