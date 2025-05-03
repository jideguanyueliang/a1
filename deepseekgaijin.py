import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, classification_report
import os
from warnings import filterwarnings

filterwarnings('ignore')

# 设置随机种子保证结果可复现
np.random.seed(42)


# 1. 数据加载与检查
def load_data(file_path):
    """
    加载数据文件并进行基本检查

    参数:
        file_path (str): 数据文件路径

    返回:
        pd.DataFrame: 加载的数据
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在")

        if not file_path.endswith(('.xlsx', '.xls')):
            raise ValueError("仅支持Excel文件格式(.xlsx, .xls)")

        data = pd.read_excel(file_path)
        print("数据加载成功！")
        print(f"数据集形状: {data.shape}")
        return data

    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return None


# 加载数据
file_path = 'Data.xlsx'  # 请根据实际路径修改
data = load_data(file_path)

if data is None:
    exit()

# 2. 数据探索
print("\n数据集基本信息:")
print(data.info())

print("\n数据集描述统计:")
print(data.describe())

print("\n缺失值情况:")
print(data.isnull().sum())

# 可视化特征分布
plt.figure(figsize=(12, 8))
sns.pairplot(data)
plt.suptitle("特征分布与关系图", y=1.02)
plt.savefig('feature_distributions.png', bbox_inches='tight')
plt.close()


# 3. 数据预处理
def preprocess_data(data, target_col='y'):
    """
    数据预处理流程

    参数:
        data (pd.DataFrame): 原始数据
        target_col (str): 目标列名

    返回:
        tuple: (处理后的特征, 目标变量, 预处理对象)
    """
    # 处理缺失值 - 仅对数值列用中位数填充
    numeric_cols = data.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if data[col].isnull().sum() > 0:
            median_val = data[col].median()
            data[col].fillna(median_val, inplace=True)
            print(f"列 {col} 的 {data[col].isnull().sum()} 个缺失值已用中位数 {median_val:.2f} 填充")

    # 分离特征和目标
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


X, y, scaler = preprocess_data(data)


# 4. 回归模型训练与评估
def evaluate_regression_models(X, y, test_size=0.2):
    """
    评估多种回归模型并返回结果

    参数:
        X (np.array): 特征矩阵
        y (np.array): 目标变量
        test_size (float): 测试集比例

    返回:
        dict: 模型评估结果
        object: 最佳模型
    """
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # 初始化模型
    models = {
        "线性回归": LinearRegression(),
        "支持向量回归": SVR(),
        "决策树回归": DecisionTreeRegressor(random_state=42)
    }

    results = {}
    best_model = None
    best_r2 = -np.inf

    print("\n回归模型评估:")
    print("-" * 50)

    for name, model in models.items():
        # 交叉验证评估
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

        # 训练模型
        model.fit(X_train, y_train)

        # 预测与评估
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # 保存结果
        results[name] = {
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'CV_R2_mean': np.mean(cv_scores),
            'CV_R2_std': np.std(cv_scores)
        }

        # 更新最佳模型
        if r2 > best_r2:
            best_r2 = r2
            best_model = model

        # 打印结果
        print(f"{name}:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R2: {r2:.4f}")
        print(f"  交叉验证R2: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        print("-" * 50)

    return results, best_model


reg_results, best_reg_model = evaluate_regression_models(X, y)


# 5. 超参数调优
def tune_hyperparameters(X_train, y_train):
    """
    对模型进行超参数调优

    参数:
        X_train (np.array): 训练特征
        y_train (np.array): 训练目标

    返回:
        dict: 调优后的最佳模型
    """
    # 分割训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # SVR参数网格
    svr_param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 1],
        'kernel': ['rbf', 'linear']
    }

    # 决策树参数网格
    dt_param_grid = {
        'max_depth': [3, 5, 7, 9, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # 初始化模型
    svr = SVR()
    dt = DecisionTreeRegressor(random_state=42)

    # 网格搜索
    svr_grid = GridSearchCV(svr, svr_param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
    dt_grid = GridSearchCV(dt, dt_param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)

    print("\n开始超参数调优...")
    svr_grid.fit(X_train, y_train)
    dt_grid.fit(X_train, y_train)

    # 评估最佳模型
    best_models = {
        'SVR': {
            'model': svr_grid.best_estimator_,
            'params': svr_grid.best_params_,
            'score': svr_grid.best_score_
        },
        'DecisionTree': {
            'model': dt_grid.best_estimator_,
            'params': dt_grid.best_params_,
            'score': dt_grid.best_score_
        }
    }

    # 在验证集上评估
    for name, info in best_models.items():
        val_score = info['model'].score(X_val, y_val)
        print(f"\n{name}最佳模型:")
        print(f"  最佳参数: {info['params']}")
        print(f"  训练R2: {info['score']:.4f}")
        print(f"  验证R2: {val_score:.4f}")

    return best_models


# 获取调优后的最佳模型
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tuned_models = tune_hyperparameters(X_train, y_train)


# 6. 分类任务分析
def classification_analysis(X, y, n_bins=3, strategy='quantile'):
    """
    执行分类任务分析

    参数:
        X (np.array): 特征矩阵
        y (np.array): 连续目标变量
        n_bins (int): 分箱数量
        strategy (str): 分箱策略

    返回:
        dict: 分类评估结果
    """
    # 离散化目标变量
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    y_class = discretizer.fit_transform(y.values.reshape(-1, 1)).ravel()
    class_labels = [f'Class_{i}' for i in range(n_bins)]

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )

    # 训练分类模型
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # 预测
    y_pred = clf.predict(X_test)

    # 评估
    conf_mat = confusion_matrix(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred, target_names=class_labels, output_dict=True)

    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    plt.close()

    # 可视化分类报告
    report_df = pd.DataFrame(clf_report).transpose()
    metrics_df = report_df[['precision', 'recall', 'f1-score']].drop(['accuracy', 'macro avg', 'weighted avg'])

    plt.figure(figsize=(10, 6))
    metrics_df.plot(kind='bar', figsize=(10, 6))
    plt.title('分类指标 (Precision, Recall, F1-score)')
    plt.ylabel('分数')
    plt.xlabel('类别')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('classification_metrics.png', bbox_inches='tight')
    plt.close()

    # 打印重要特征
    feature_importances = pd.DataFrame({
        'Feature': data.drop(columns=['y']).columns,
        'Importance': clf.feature_importances_
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importances)
    plt.title('特征重要性')
    plt.tight_layout()
    plt.savefig('feature_importances.png', bbox_inches='tight')
    plt.close()

    return {
        'confusion_matrix': conf_mat,
        'classification_report': clf_report,
        'feature_importances': feature_importances
    }


# 执行分类分析
print("\n开始分类任务分析...")
class_results = classification_analysis(X, y)

# 打印分类报告
print("\n分类报告:")
print(pd.DataFrame(class_results['classification_report']).transpose())


# 7. 保存最佳模型和结果
def save_results(results, filename='results.txt'):
    """保存结果到文件"""
    with open(filename, 'w') as f:
        for model_name, metrics in results.items():
            f.write(f"{model_name} 结果:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value}\n")
            f.write("\n")
    print(f"\n结果已保存到 {filename}")


# 保存回归结果
save_results(reg_results)

# 打印最终总结
print("\n" + "=" * 50)
print("分析总结:")
print(f"- 最佳回归模型: {type(best_reg_model).__name__}")
print(f"- 最佳回归R2分数: {reg_results[type(best_reg_model).__name__]['R2']:.4f}")
print(f"- 分类准确率: {class_results['classification_report']['accuracy']:.4f}")
print("=" * 50)