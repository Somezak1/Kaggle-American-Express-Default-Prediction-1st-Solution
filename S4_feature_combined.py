import warnings
warnings.simplefilter('ignore')

import pandas as pd
from tqdm import tqdm

from utils import *

root = args.root

oof = pd.read_csv('./output/LGB_with_series_feature/oof.csv')  # 训练好的模型对于oof样本进行的预测结果
sub = pd.read_csv('./output/LGB_with_series_feature/submission.csv.zip')  # 训练好的模型对于测试集样本预测结果的平均

def pad_target(x):
    t = np.zeros(13)
    t[:-len(x)] = np.nan
    t[-len(x):] = x
    return list(t)

tmp1 = oof.groupby('customer_ID',sort=False)['target'].agg(lambda x:pad_target(x))
# 例如
# oof = pd.DataFrame({
#      "customer_ID": [A,A,A,A,A,B,B,B,B],
#      "target": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# })
# customer_ID
# A    [nan, 0.1, 0.2, 0.3, 0.4, 0.5]  # 以长度为6演示，其中概率值是我随机填充的
# B    [nan, nan, 0.6, 0.7, 0.8, 0.9]
# Name: target, dtype: object

tmp2 = sub.groupby('customer_ID',sort=False)['prediction'].agg(lambda x:pad_target(x))
# customer_ID
# C    [nan, 0.9, 0.8, 0.7, 0.6, 0.5]
# D    [nan, nan, 0.4, 0.3, 0.2, 0.1]
# Name: prediction, dtype: object

tmp = tmp1.append(tmp2)
# customer_ID
# A    [nan, 0.1, 0.2, 0.3, 0.4, 0.5]
# B    [nan, nan, 0.6, 0.7, 0.8, 0.9]
# C    [nan, 0.9, 0.8, 0.7, 0.6, 0.5]
# D    [nan, nan, 0.4, 0.3, 0.2, 0.1]
# dtype: object

tmp = pd.DataFrame(data=tmp.tolist(),columns=['target%s'%i for i in range(1,14)])
#    target1  target2  target3  target4  target5  target6   # 以长度为6演示
# 0      NaN      0.1      0.2      0.3      0.4      0.5
# 1      NaN      NaN      0.6      0.7      0.8      0.9
# 2      NaN      0.9      0.8      0.7      0.6      0.5
# 3      NaN      NaN      0.4      0.3      0.2      0.1


df = []
for fn in ['cat','num','diff','rank_num','last3_cat','last3_num','last3_diff', 'last6_num','ym_rank_num']:
    if len(df) == 0:
        df.append(pd.read_feather(f'{root}/{fn}_feature.feather'))
    else:
        df.append(pd.read_feather(f'{root}/{fn}_feature.feather').drop([id_name],axis=1))
    if 'last' in fn :
        df[-1] = df[-1].add_prefix('_'.join(fn.split('_')[:-1])+'_')  # 给前缀有last的几个宽表里面的特征，加上前缀

df.append(tmp)

df = pd.concat(df,axis=1)
print(df.shape)
# 假设df中除tmp以外的元素长下面这样
#     f1   f2
# 0    1    3
# 1    2    4
# 2    8    2
# 3    4    0

#     f1   f2  target1  target2  target3  target4  target5  target6
# 0  1.0  3.0      NaN      0.1      0.2      0.3      0.4      0.5
# 1  2.0  4.0      NaN      NaN      0.6      0.7      0.8      0.9
# 2  8.0  2.0      NaN      0.9      0.8      0.7      0.6      0.5
# 3  4.0  0.0      NaN      NaN      0.4      0.3      0.2      0.1
# 前两行是训练集样本，后两行是测试集样本
# 前两列是该用户统计得到的特征，后几列是lgb模型对于每个用户在第n个月违约的预测概率
df.to_feather(f'{root}/all_feature.feather')

del df

def one_hot_encoding(df,cols,is_drop=True):
    for col in cols:
        print('one hot encoding:',col)
        dummies = pd.get_dummies(pd.Series(df[col]),prefix='oneHot_%s'%col)
        df = pd.concat([df,dummies],axis=1)
    if is_drop:
        df.drop(cols,axis=1,inplace=True)
    return df

cat_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]

df = pd.read_feather(f'./input/train.feather').append(pd.read_feather(f'./input/test.feather')).reset_index(drop=True)
df = df.drop(['S_2'],axis=1)
df = one_hot_encoding(df,cat_features,True)  # 将train和test拼接，然后将类别型特征one-hot编码，随后丢弃掉这些类别特征
                                             # 此时，df中只剩原始的数值型特征和one-hot展开的类别型特征
for col in tqdm(df.columns):
    if col not in ['customer_ID','S_2']:
        df[col] /= 100
    df[col] = df[col].fillna(0)              # 将df中的特征值统一除以100，然后用0填充缺失值

df.to_feather('./input/nn_series.feather')

def GreedyFindBin(distinct_values, counts,num_distinct_values, max_bin, total_cnt, min_data_in_bin=3):
#INPUT:
#   distinct_values 保存特征取值的数组，特征取值单调递增
#   counts 特征的取值对应的样本数目
#   num_distinct_values 特征取值的数量
#   max_bin 分桶的最大数量
#   total_cnt 样本数量
#   min_data_in_bin 桶包含的最小样本数

# bin_upper_bound就是记录桶分界的数组
    bin_upper_bound=list();
    assert(max_bin>0)

    # 特征取值数比max_bin数量少，直接取distinct_values的中点放置
    if num_distinct_values <= max_bin:
        cur_cnt_inbin = 0
        for i in range(num_distinct_values-1):
            cur_cnt_inbin += counts[i]
            #若一个特征的取值比min_data_in_bin小，则累积下一个取值，直到比min_data_in_bin大，进入循环。
            if cur_cnt_inbin >= min_data_in_bin:
                #取当前值和下一个值的均值作为该桶的分界点bin_upper_bound
                bin_upper_bound.append((distinct_values[i] + distinct_values[i + 1]) / 2.0)
                cur_cnt_inbin = 0
        # 对于最后一个桶的上界则为无穷大
        cur_cnt_inbin += counts[num_distinct_values - 1];
        bin_upper_bound.append(float('Inf'))
        # 特征取值数比max_bin来得大，说明几个特征值要共用一个bin
    else:
        if min_data_in_bin>0:
            max_bin=min(max_bin,total_cnt//min_data_in_bin)
            max_bin=max(max_bin,1)
        #mean size for one bin
        mean_bin_size=total_cnt/max_bin
        rest_bin_cnt = max_bin
        rest_sample_cnt = total_cnt
        #定义is_big_count_value数组：初始设定特征每一个不同的值的数量都小（false）
        is_big_count_value=[False]*num_distinct_values
        #如果一个特征值的数目比mean_bin_size大，那么这些特征需要单独一个bin
        for i in range(num_distinct_values):
        #如果一个特征值的数目比mean_bin_size大，则设定这个特征值对应的is_big_count_value为真。。
            if counts[i] >= mean_bin_size:
                is_big_count_value[i] = True
                rest_bin_cnt-=1
                rest_sample_cnt -= counts[i]
        #剩下的特征取值的样本数平均每个剩下的bin：mean size for one bin
        mean_bin_size = rest_sample_cnt/rest_bin_cnt
        upper_bounds=[float('Inf')]*max_bin
        lower_bounds=[float('Inf')]*max_bin

        bin_cnt = 0
        lower_bounds[bin_cnt] = distinct_values[0]
        cur_cnt_inbin = 0
        #重新遍历所有的特征值（包括数目大和数目小的）
        for i in range(num_distinct_values-1):
            #如果当前的特征值数目是小的
            if not is_big_count_value[i]:
                rest_sample_cnt -= counts[i]
            cur_cnt_inbin += counts[i]

            # 若cur_cnt_inbin太少，则累积下一个取值，直到满足条件，进入循环。
            # need a new bin 当前的特征如果是需要单独成一个bin，或者当前几个特征计数超过了mean_bin_size，或者下一个是需要独立成桶的
            if is_big_count_value[i] or cur_cnt_inbin >= mean_bin_size or \
            is_big_count_value[i + 1] and cur_cnt_inbin >= max(1.0, mean_bin_size * 0.5):
                upper_bounds[bin_cnt] = distinct_values[i] # 第i个bin的最大就是 distinct_values[i]了
                bin_cnt+=1
                lower_bounds[bin_cnt] = distinct_values[i + 1] # 下一个bin的最小就是distinct_values[i + 1]，注意先++bin了
                if bin_cnt >= max_bin - 1:
                    break
                cur_cnt_inbin = 0
                if not is_big_count_value[i]:
                    rest_bin_cnt-=1
                    mean_bin_size = rest_sample_cnt / rest_bin_cnt
#             bin_cnt+=1
        # update bin upper bound 与特征取值数比max_bin数量少的操作类似，取当前值和下一个值的均值作为该桶的分界点
        for i in range(bin_cnt-1):
            bin_upper_bound.append((upper_bounds[i] + lower_bounds[i + 1]) / 2.0)
        bin_upper_bound.append(float('Inf'))
    return bin_upper_bound

cat_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]
eps = 1e-3

dfs = []
for fn in ['cat','num','diff','rank_num','last3_cat','last3_num','last3_diff', 'last6_num','ym_rank_num']:
    if len(dfs) == 0:
        dfs.append(pd.read_feather(f'{root}/{fn}_feature.feather'))
    else:
        dfs.append(pd.read_feather(f'{root}/{fn}_feature.feather').drop(['customer_ID'],axis=1))

    if 'last' in fn:
        dfs[-1] = dfs[-1].add_prefix('_'.join(fn.split('_')[:-1])+'_')

for df in dfs:
    for col in tqdm(df.columns):
        if col not in ['customer_ID','S_2']:
            # v_min = df[col].min()
            # v_max = df[col].max()
            # df[col] = (df[col]-v_min+eps) / (v_max-v_min+eps)

            vc = df[col].value_counts().sort_index()  # 按df[col]的值排序
            if len(vc) != 0:
                bins = GreedyFindBin(vc.index.values,vc.values,len(vc),255,vc.sum())
                df[col] = np.digitize(df[col],[-np.inf]+bins)  # 将col特征分到对应的箱里
                df.loc[df[col]==len(bins)+1,col] = 0  # 调整边界情况
                df[col] = df[col] / df[col].max()  # 将所属分箱值除以分箱总数
            else:
                df[col] = 0

tmp = tmp.fillna(0)
dfs.append(tmp)
df = pd.concat(dfs,axis=1)

df.to_feather('./input/nn_all_feature.feather')
