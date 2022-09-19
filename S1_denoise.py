import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# 对训练集和测试集中D_63和D_64两个特征分别进行LabelEncoder
# 对训练集和测试集中除D_63和D_64以外的所有特征，降低特征的精度

def denoise(df):
    # 对D_63和D_64两个特征进行LabelEncoder
    df['D_63'] = df['D_63'].apply(lambda t: {'CR':0, 'XZ':1, 'XM':2, 'CO':3, 'CL':4, 'XL':5}[t]).astype(np.int8)
    df['D_64'] = df['D_64'].apply(lambda t: {np.nan:-1, 'O':0, '-1':1, 'R':2, 'U':3}[t]).astype(np.int8)
    # 对除D_63和D_64以外的所有特征，降低特征的精度
    for col in tqdm(df.columns):
        if col not in ['customer_ID','S_2','D_63','D_64']:
            df[col] = np.floor(df[col]*100)
    return df

train = pd.read_csv('./input/train_data.csv')
train = denoise(train)
train.to_feather('./input/train.feather')

del train

test = pd.read_csv('./input/test_data.csv')
test = denoise(test)
test.to_feather('./input/test.feather')
