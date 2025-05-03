import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings("ignore")  # 忽略警告信息

# 函数：模型训练与评估
def evaluate_regression_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2, y_pred

# 函数：绘制混淆矩阵
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

# 主程序
def main():
    # 1. 加载数据
    file_path = 'Data.xlsx'  # 根据实际路径修改
    try:
        data = pd.read_excel(file_path)
        print("数据加载成功！")
    except Exception as e:
        print(f"数据加载失败：{e}")
        return

    print(data.head())

    # 2. 数据可视化
    sns.pairplot(data)
    plt.show()

    # 3. 数据基本信息
    print("\n数据集基本信息:")
    print(data.info())
    print("\n前6行数据:")
    print(data.head(6))

    # 4. 数据预处理
    print("\n缺失值情况:")
    print(data.isnull().sum())

    data.fillna(data.mean(), inplace=True)

    X = data.drop(columns=['y'])
    y = data['y']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 5. 回归模型训练与评估
    models = {
        "线性回归": LinearRegression(),
        "支持向量回归": SVR(),
        "决策树回归": DecisionTreeRegressor(random_state=42)
    }

    results = {}
    for name, model in models.items():
        mse, mae, r2, _ = evaluate_regression_model(model, X_train, X_test, y_train, y_test)
        results[name] = (mse, mae, r2)

    # 输出各模型性能
    for name, metrics in results.items():
        print(f"\n{name}回归性能:")
        print(f"均方误差 (MSE): {metrics[0]:.4f}")
        print(f"平均绝对误差 (MAE): {metrics[1]:.4f}")
        print(f"决定系数 (R^2): {metrics[2]:.4f}")

    # 选择最优模型
    best_model_name = max(results.items(), key=lambda x: x[1][2])[0]
    print(f"\n最优模型是: {best_model_name}")

    # 6. 数据离散化（分类任务）
    bins = [0, 30, 60, 100]
    labels = ['Low', 'Medium', 'High']
    data['weight_class'] = pd.cut(data['y'], bins=bins, labels=labels, right=False)

    # 用决策树预测结果作为分类依据
    best_regressor = DecisionTreeRegressor(random_state=42)
    best_regressor.fit(X_scaled, y)
    y_pred_full = best_regressor.predict(X_scaled)
    data['predicted_class'] = pd.cut(y_pred_full, bins=bins, labels=labels, right=False)

    # 7. 混淆矩阵和分类报告
    true_labels = data['weight_class']
    predicted_labels = data['predicted_class']

    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)
    plot_confusion_matrix(conf_matrix, labels)

    report = classification_report(true_labels, predicted_labels, target_names=labels, output_dict=True)
    precision_recall_df = pd.DataFrame(report).transpose()

    # 筛选 Precision、Recall、F1
    metrics_to_plot = precision_recall_df[['precision', 'recall', 'f1-score']].drop(['accuracy', 'macro avg', 'weighted avg'])
    print("\n各类别 Precision, Recall, F1-score：")
    print(metrics_to_plot)

    # 输出整体汇总指标
    accuracy = report['accuracy']
    weighted_precision = report['weighted avg']['precision']
    weighted_recall = report['weighted avg']['recall']
    weighted_f1 = report['weighted avg']['f1-score']

    print(f"\n整体分类指标:")
    print(f"整体精度 (Accuracy): {accuracy:.4f}")
    print(f"加权查准率 (Weighted Precision): {weighted_precision:.4f}")
    print(f"加权查全率 (Weighted Recall): {weighted_recall:.4f}")
    print(f"加权 F1 值 (Weighted F1-score): {weighted_f1:.4f}")

    # 可视化
    metrics_to_plot.plot(kind='bar', figsize=(10, 7))
    plt.title('Precision, Recall, F1-Score by Class')
    plt.ylabel('Score')
    plt.xlabel('Class')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # 8. 超参数调优
    # SVR调优
    svr_model = SVR()
    param_grid_svr = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'linear']}
    grid_search_svr = GridSearchCV(svr_model, param_grid_svr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search_svr.fit(X_train, y_train)
    print(f"\n最佳SVR参数: {grid_search_svr.best_params_}")

    # 决策树调优
    dt_model = DecisionTreeRegressor(random_state=42)
    param_grid_dt = {'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
    grid_search_dt = GridSearchCV(dt_model, param_grid_dt, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search_dt.fit(X_train, y_train)
    print(f"\n最佳决策树参数: {grid_search_dt.best_params_}")

    # 9. 测试最佳模型
    best_svr = grid_search_svr.best_estimator_
    _, _, r2_svr, _ = evaluate_regression_model(best_svr, X_train, X_test, y_train, y_test)
    print(f"\nSVR测试集R²: {r2_svr:.4f}")

    best_dt = grid_search_dt.best_estimator_
    _, _, r2_dt, _ = evaluate_regression_model(best_dt, X_train, X_test, y_train, y_test)
    print(f"决策树测试集R²: {r2_dt:.4f}")

    # 10. 选择最佳最终模型
    if r2_svr > r2_dt:
        final_model = '支持向量回归（SVR）'
        final_r2 = r2_svr
    else:
        final_model = '决策树回归'
        final_r2 = r2_dt

    print(f"\n最终最佳模型: {final_model} (R²={final_r2:.4f})")

    # 可选：保存分类报告
    precision_recall_df.to_excel('classification_report.xlsx')
    print("\n分类报告已保存到 'classification_report.xlsx'。")


if __name__ == "__main__":
    main()
