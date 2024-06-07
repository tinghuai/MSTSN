# 导入必要的库
import pandas as pd
import numpy as np
from skfeature.function.information_theoretical_based.MRMR import mrmr
from sklearn.feature_selection import mutual_info_regression

# 计算每个特征的mRMR得分
def mrmr_scores(X, Y):
    scores=pd.DataFrame(index=df_lagged.drop('SPEI_9_t', axis=1).columns)
    mi_scores = mutual_info_regression(X, Y)
    correlations = pd.DataFrame(X).corr()
    for i in range(X.shape[1]):
        mi = mi_scores[i]
        avg_corr = correlations[i].mean()
        mrmr = mi - avg_corr
        scores.loc[scores.index[i],'Score'] = mrmr
    return scores

# 读取数据
df = pd.read_excel('data/平稳时间序列/53898_new.xls')

n_lags = 9

# 移除你认为无关的列
df.drop(['SPEI_12','SPEI_1','SPEI_3','SPEI_6', '日期'], axis=1, inplace=True)

# 创建一个空的DataFrame用于存放历史公共数据集
df_lagged = pd.DataFrame()

# 产生滞后的自变量数据
for i in range(1, n_lags+1):
    shifted = df.shift(i)
    shifted.columns = [str.format("%s_t-%d" % (col, i)) for col in shifted.columns]
    df_lagged = pd.concat((df_lagged, shifted), axis=1)

# 保留当前时间步的SPEI_1
df_lagged['SPEI_9_t'] = df['SPEI_9']

# 删除含有NaN的行
df_lagged = df_lagged.dropna()

# 将'SPEI_1_t'这一列单独提取出来作为目标变量 Y
Y = df_lagged['SPEI_9_t'].values

# 将其余列作为自变量 X
X = df_lagged.drop('SPEI_9_t', axis=1).values

# 使用mRMR方法计算特征得分
selected_features = mrmr(X, Y, n_selected_features=6)

# 转换回原始特征名称
feature_names = df_lagged.drop('SPEI_9_t', axis=1).columns[selected_features]

# 计算特征的mRMR得分
scores = mrmr_scores(X, Y)

# 创建一个DataFrame，其中包含所选择的特征名称及其得分
output = pd.DataFrame({'Selected Features': feature_names, 'mRMR Score': scores.loc[feature_names, 'Score']})

# 设置DataFrame的索引为特征名称
output.set_index('Selected Features', inplace=True)

# 将DataFrame保存为Excel文件
output.to_excel("mRMR_scores.xlsx")