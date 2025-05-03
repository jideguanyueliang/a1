import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, classification_report
import time

# 1. 加载数据
file_path = 'Data.xlsx'  # 请根据实际路径修改
try:
    data = pd.read_excel(file_path)
    print("数据加载成功！")
    print(data.head())
except FileNotFoundError:
    print(f"文件 {file_path} 未找到，请检查路径是否正确！")
    exit()

# 可视化特征之间的关系
sns.pairplot(data)
plt.show()

# 2. 数据集基本信息
print("数据集基本信息:")
print(data.info())

# 显示前5行数据
print("\n数据集前6行:")
print(data.head(6))

# 3. 数据预处理
# 检查是否有缺失值
print("\n缺失值情况:")
print(data.isnull().sum())

# 填充缺失值（根据数据类型选择填充方式）
for column in data.columns:
    if data[column].dtype == 'object':
        data[column].fillna(data[column].mode()[0], inplace=True)
    else:
        data[column].fillna(data[column].mean(), inplace=True)

# 选择特征X和目标变量y，假设目标变量是 'y'
target_column = 'y'  # 确保目标变量列名正确
X = data.drop(columns=[target_column])
y = data[target_column]

# 特征标准化（仅对数值型特征进行标准化）
numeric_features = X.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# 4. 分离训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 训练和评估回归模型
def evaluate_regression_model(model, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2, training_time

# 回归模型
models = {
    "线性回归": LinearRegression(),
    "支持向量回归": SVR(),
    "决策树回归": DecisionTreeRegressor(random_state=42)
}

# 评估回归模型
results = {}
for name, model in models.items():
    mse, mae, r2, training_time = evaluate_regression_model(model, X_train, X_test, y_train, y_test)
    results[name] = (mse, mae, r2, training_time)

# 输出回归模型评估结果
for name, metrics in results.items():
    print(f"{name}的回归性能:")
    print(f"均方误差 (MSE): {metrics[0]:.4f}")
    print(f"平均绝对误差 (MAE): {metrics[1]:.4f}")
    print(f"决定系数 (R^2): {metrics[2]:.4f}")
    print(f"训练时间: {metrics[3]:.4f} 秒\n")

# 选择最优的回归模型
best_model = max(results.items(), key=lambda x: x[1][2])
print(f"最优的回归模型是{best_model[0]}，对应的R^2值为: {best_model[1][2]:.4f}。\n")

# 6. 数据离散化（分类任务）
# 离散化承重等级
quantiles = [0, 0.33, 0.66, 1]
bins = data[target_column].quantile(quantiles).tolist()
labels = ['Low', 'Medium', 'High']
data['weight_class'] = pd.cut(data[target_column], bins=bins, labels=labels, right=False)

# 添加模拟的预测列（如果没有真实的预测数据）
np.random.seed(42)  # 确保结果可重复
data['predicted_class'] = np.random.choice(labels, size=len(data))

# 7. 混淆矩阵和分类报告
true_labels = data['weight_class']
predicted_labels = data['predicted_class']

# 计算混淆矩阵
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# 输出分类报告
report = classification_report(true_labels, predicted_labels, target_names=labels, output_dict=True)
precision_recall_df = pd.DataFrame(report).transpose()
precision_recall_df = precision_recall_df.drop(columns=['support'])

# 筛选查准率（Precision）和查全率（Recall）
precision_recall = precision_recall_df[['precision', 'recall', 'f1-score']].drop(['accuracy', 'macro avg', 'weighted avg'])
print("Precision, Recall, and F1-Score:")
print(precision_recall)

# 从分类报告中提取汇总指标
accuracy = precision_recall_df.loc['accuracy', 'precision']  # 精度
weighted_precision = precision_recall_df.loc['weighted avg', 'precision']  # 加权查准率
weighted_recall = precision_recall_df.loc['weighted avg', 'recall']        # 加权查全率
weighted_f1 = precision_recall_df.loc['weighted avg', 'f1-score']          # 加权F1值

# 打印整体汇总指标
print(f"\n整体分类汇总指标:")
print(f"整体精度 (Accuracy): {accuracy:.4f}")
print(f"加权查准率 (Weighted Precision): {weighted_precision:.4f}")
print(f"加权查全率 (Weighted Recall): {weighted_recall:.4f}")
print(f"加权 F1 值 (Weighted F1-score): {weighted_f1:.4f}")

# 可视化查准率、查全率和 F1 值
precision_recall.plot(kind='bar', figsize=(10, 7))
plt.title('Precision, Recall, and F1-Score by Class')
plt.ylabel('Score')
plt.xlabel('Class')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 分析不同承重等级混凝土的指标
for label in labels:
    precision = precision_recall_df.loc[label, 'precision']
    recall = precision_recall_df.loc[label, 'recall']
    f1 = precision_recall_df.loc[label, 'f1-score']
    print(f"承重等级 {label} 的查准率 (Precision): {precision:.4f}, 查全率 (Recall): {recall:.4f}, F1值: {f1:.4f}")

# 8. 交叉验证和超参数调优
# 支持向量回归（SVR）模型及其超参数调优
svr_model = SVR()
param_grid_svr = {'C': [0.01, 0.1, 1, 10, 100], 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'linear']}
grid_search_svr = GridSearchCV(svr_model, param_grid_svr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_svr.fit(X_train, y_train)

# 打印SVR的最佳超参数
print(f"\n支持向量回归模型的最佳超参数：{grid_search_svr.best_params_}")
print(f"支持向量回归模型的最佳得分：{-grid_search_svr.best_score_}")

# 决策树回归模型及其超参数调优
dt_model = DecisionTreeRegressor(random_state=42)
param_grid_dt = {'max_depth': [3, 5, 7, 10, None], 'min_samples_split': [2, 5, 10, 20], 'min_samples_leaf': [1, 2, 4, 8]}
grid_search_dt = GridSearchCV(dt_model, param_grid_dt, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_dt.fit(X_train, y_train)

# 打印决策树回归的最佳超参数
print(f"\n决策树回归模型的最佳超参数：{grid_search_dt.best_params_}")
print(f"决策树回归模型的最佳得分：{-grid_search_dt.best_score_}")

# 9. 测试最佳超参数模型
# 选择最佳超参数的SVR模型
best_svr = grid_search_svr.best