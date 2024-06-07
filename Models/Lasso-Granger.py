from sklearn.linear_model import Lasso
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
def fit_granger(data, lags=2, alpha=0.1):
    num_variables = data.shape[1]
    coefficients = np.zeros((num_variables, lags * num_variables + 1))
    for target_variable in range(num_variables):
        X = np.ones((data.shape[0] - lags, lags * num_variables + 1))
        for l in range(lags):
            X[:, 1 + l * num_variables:1 + (l + 1) * num_variables] = data[l:data.shape[0] - lags + l, :]
        y = data[lags:, target_variable]
        model = Lasso(alpha=alpha, fit_intercept=False)
        model.fit(X, y)
        coefficients[target_variable, :] = model.coef_
    return coefficients

df = pandas.read_excel('data/多源数据/53898.xls')
df = df.iloc[:, 1:]
data = df.values

num_variables = data.shape[1]

coefficients = fit_granger(df.values, lags=2, alpha=0.1)

# 系数矩阵的形状为 (变量数目，lags * 变量数目 + 1)
# 为了提取紧急相关的系数，我们需要将其变形
coefs_reshaped = coefficients[:, 1:].reshape(coefficients.shape[0], -1, num_variables)

# 计算每个变量对其他变量的总影响
sum_abs_coefs = np.sum(np.abs(coefs_reshaped), axis=1)

# 因果关系热力图
plt.figure(figsize=(10,10))
sns.heatmap(sum_abs_coefs, yticklabels=df.columns, xticklabels=df.columns, cmap='Blues')
plt.show()

