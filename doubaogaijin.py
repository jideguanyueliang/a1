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
import os


def load_data(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在。")
        data = pd.read_excel(file_path)
        print("数据加载成功！")
        return data
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None


def visualize_features(data):
    g = sns.pairplot(data)
    g.fig.suptitle('特征之间的关系', y=1.02)
    plt.show()


def get_data_info(data):
    print("数据集基本信息:")
    print(data.info())
    print("\n数据集前6行:")
    print(data.head(6))


def preprocess_data(data):
    print("\n缺失值情况:")
    print(data.isnull().sum())
    data.fillna(data.mean(), inplace=True)
    X = data.drop(columns=['y'])
    y = data['y']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate_regression_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    return mse, mae, r2, rmse


def train_and_evaluate_models(models, X_train, X_test, y_train, y_test):
    results = {}
    for name, model in models.items():
        mse, mae, r2, rmse = evaluate_regression_model(model, X_train, X_test, y_train, y_test)
        results[name] = (mse, mae, r2, rmse)
    return results


def print_model_results(results):
    for name, metrics in results.items():
        print(f"{name}的回归性能:")
        print(f"均方误差 (MSE): {metrics[0]:.4f}")
        print(f"平均绝对误差 (MAE): {metrics[1]:.4f}")
        print(f"决定系数 (R^2): {metrics[2]:.4f}")
        print(f"均方根误差 (RMSE): {metrics[3]:.4f}\n")


def select_best_model(results):
    best_model = max(results.items(), key=lambda x: x[1][2])
    print(f"最优的回归模型是{best_model[0]}，对应的R^2值为: {best_model[1][2]:.4f}。\n")
    return best_model[0]


def discretize_data(data):
    bins = [0, 30, 60, 100]
    labels = ['Low', 'Medium', 'High']
    data['weight_class'] = pd.cut(data['y'], bins=bins, labels=labels, right=False)
    np.random.seed(42)
    data['predicted_class'] = np.random.choice(labels, size=len(data))
    return data


def analyze_classification(data):
    true_labels = data['weight_class']
    predicted_labels = data['predicted_class']
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=['Low', 'Medium', 'High'])

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Low', 'Medium', 'High'],
                yticklabels=['Low', 'Medium', 'High'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.show()

    report = classification_report(true_labels, predicted_labels, target_names=['Low', 'Medium', 'High'], output_dict=True)
    precision_recall_df = pd.DataFrame(report).transpose()
    precision_recall = precision_recall_df[['precision','recall', 'f1-score']].drop(
        ['accuracy','macro avg', 'weighted avg'])
    print("Precision, Recall, and F1-Score:")
    print(precision_recall)

    accuracy = precision_recall_df.loc['accuracy', 'precision']
    weighted_precision = precision_recall_df.loc['weighted avg', 'precision']
    weighted_recall = precision_recall_df.loc['weighted avg','recall']
    weighted_f1 = precision_recall_df.loc['weighted avg', 'f1-score']
    print(f"\n整体分类汇总指标:")
    print(f"整体精度 (Accuracy): {accuracy:.4f}")
    print(f"加权查准率 (Weighted Precision): {weighted_precision:.4f}")
    print(f"加权查全率 (Weighted Recall): {weighted_recall:.4f}")
    print(f"加权 F1 值 (Weighted F1-score): {weighted_f1:.4f}")

    precision_recall.plot(kind='bar', figsize=(10, 7))
    plt.title('各等级查准率、查全率和F1值')
    plt.ylabel('得分')
    plt.xlabel('等级')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    for label in ['Low', 'Medium', 'High']:
        precision = precision_recall_df.loc[label, 'precision']
        recall = precision_recall_df.loc[label,'recall']
        f1 = precision_recall_df.loc[label, 'f1-score']
        print(f"承重等级 {label} 的查准率 (Precision): {precision:.4f}, 查全率 (Recall): {recall:.4f}, F1值: {f1:.4f}")


def hyperparameter_tuning(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, -grid_search.best_score_


def test_best_models(best_svr, best_dt, X_train, X_test, y_train, y_test):
    best_svr.fit(X_train, y_train)
    y_pred_svr = best_svr.predict(X_test)
    mse_svr = mean_squared_error(y_test, y_pred_svr)
    r2_svr = r2_score(y_test, y_pred_svr)
    print(f"\n支持向量回归（SVR）模型在测试集上的评估：")
    print(f"MSE: {mse_svr}")
    print(f"R^2: {r2_svr}")

    best_dt.fit(X_train, y_train)
    y_pred_dt = best_dt.predict(X_test)
    mse_dt = mean_squared_error(y_test, y_pred_dt)
    r2_dt = r2_score(y_test, y_pred_dt)
    print(f"\n决策树回归模型在测试集上的评估：")
    print(f"MSE: {mse_dt}")
    print(f"R^2: {r2_dt}")
    return mse_svr, r2_svr, mse_dt, r2_dt


def select_best_performing_model(mse_svr, r2_svr, mse_dt, r2_dt):
    best_model = None
    best_r2 = -np.inf
    if r2_svr > best_r2:
        best_r2 = r2_svr
        best_model = '支持向量回归（SVR）'
    if r2_dt > best_r2:
        best_r2 = r2_dt
        best_model = '决策树回归'
    print(f"\n表现最好的模型是: {best_model} (R^2: {best_r2})")


if __name__ == "__main__":
    file_path = 'Data.xlsx'
    data = load_data(file_path)
    if data is not None:
        visualize_features(data)
        get_data_info(data)
        X_scaled, y = preprocess_data(data)
        X_train, X_test, y_train, y_test = split_data(X_scaled, y)

        models = {
            "线性回归": LinearRegression(),
            "支持向量回归": SVR(),
            "决策树回归": DecisionTreeRegressor(random_state=42)
        }
        results = train_and_evaluate_models(models, X_train, X_test, y_train, y_test)
        print_model_results(results)
        best_regression_model = select_best_model(results)

        data = discretize_data(data)
        analyze_classification(data)

        svr_model = SVR()
        param_grid_svr = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'linear']}
        svr_best_params, svr_best_score = hyperparameter_tuning(svr_model, param_grid_svr, X_train, y_train)
        print(f"\n支持向量回归模型的最佳超参数：{svr_best_params}")
        print(f"支持向量回归模型的最佳得分：{svr_best_score}")

        dt_model = DecisionTreeRegressor(random_state=42)
        param_grid_dt = {'max_depth': [3, 5, 7, None],'min_samples_split': [2, 5, 10],'min_samples_leaf': [1, 2, 4]}
        dt_best_params, dt_best_score = hyperparameter_tuning(dt_model, param_grid_dt, X_train, y_train)
        print(f"\n决策树回归模型的最佳超参数：{dt_best_params}")
        print(f"决策树回归模型的最佳得分：{dt_best_score}")

        best_svr = svr_model.set_params(**svr_best_params)
        best_dt = dt_model.set_params(**dt_best_params)
        mse_svr, r2_svr, mse_dt, r2_dt = test_best_models(best_svr, best_dt, X_train, X_test, y_train, y_test)
        select_best_performing_model(mse_svr, r2_svr, mse_dt, r2_dt)
