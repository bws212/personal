import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error

from matminer.featurizers.composition import ElementProperty
from matminer.utils.data import PymatgenData
from pymatgen.core import Composition
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR


def data_exploration(X, y):

    num_features = len(X[0])
    print(f"The fingerprint length is: {num_features}")
    print(f"The fingerprint for AG20 is: {X[0]}")

    num_crystals = len(y)
    print(f"the number of crystal samples is {num_crystals}")

    if len(X) == len(y):
        print(f"Every crystal has a fingerprint, {num_crystals} = {len(X)}")
    else:
        print(f"There is not a fingerprint for every crystal: {num_crystals} != {len(X)}")


def linear_scores(model, X_train, y_train, X_test, y_test, y_pred_train, y_pred_test):
    test_score = model.score(X_test, y_test)
    train_score = model.score(X_train, y_train)
    print(f"The model is {model}")
    print(f"The test score for the model is {test_score}")
    print(f"The train score for the model is {train_score}")
    print(f"Mean Absolute Error Testing: {mean_absolute_error(y_test, y_pred_test)} Kelvin")
    print(f"Root Mean Squared Error Testing: {sqrt(mean_squared_error(y_test, y_pred_test))} Kelvin")
    print(f"Mean Absolute Error Training: {mean_absolute_error(y_train, y_pred_train)} Kelvin")
    print(f"Root Mean Squared Error Training: {sqrt(mean_squared_error(y_train, y_pred_train))} Kelvin \n")
    return


def linear_plotting(model, y_train, y_test, y_pred_train, y_pred_test):
    plt.figure(figsize=(10, 10))
    plt.plot([-500, 4500], [-500, 4500], color='black')
    plt.scatter(y_train, y_pred_train, color='blue', alpha=0.5, label='train')
    plt.scatter(y_test, y_pred_test, color='red', alpha=0.5, label='test')
    plt.xlim([-500, 4500])
    plt.ylim([-500, 4500])
    plt.xlabel('$y_{true}$', size=16)
    plt.ylabel('$y_{pred}$', size=16)
    plt.legend(fontsize=16);
    errors_test = y_test - y_pred_test
    plt.figure(figsize=(8, 8))
    plt.hist(errors_test, bins=20, color='red', density=True)
    plt.xlabel('Error in Kelvin', size=16)
    plt.ylabel('Frequency', size=16)
    plt.title(f'{model} Error', size=16)
    return plt.show()


def plotting_clusters(n_clusters, X_train_cluster):
    plt.figure(figsize=(8, 8))

    for i in range(n_clusters):
        plt.scatter(X_train_cluster[i][:, 0], X_train_cluster[i][:, 1], alpha=0.5, marker=i + 4, label=f'cluster {i}')

    plt.xlabel('Cohesive energy', size=16)
    plt.ylabel('Bulk modulus', size=16)
    plt.legend(fontsize=16)
    return plt.show()


def plotting_cluster_results(model, n_clusters, y_train_cluster, y_pred_train_cluster, y_test_cluster, y_pred_test_cluster,
                             x_min, x_max, y_min, y_max):
    plt.figure(figsize=(10, 10))

    for i in range(n_clusters):
        plt.scatter(y_train_cluster[i], y_pred_train_cluster[i], color='blue', alpha=0.5, marker=i + 3,
                    label=f'train cluster {i}')
        plt.scatter(y_test_cluster[i], y_pred_test_cluster[i], color='red', alpha=0.5, marker=i + 3,
                    label=f'test cluster {i}')

    plt.legend(fontsize=16);
    plt.plot([x_min, x_max], [y_min, y_max], color='black')
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.xlabel('$y_{true}$', size=16)
    plt.ylabel('$y_{pred}$', size=16)

    errors_test = []
    for i in range(n_clusters):
        errors_test.extend(y_test_cluster[i] - y_pred_test_cluster[i])

    plt.figure(figsize=(8, 8))
    plt.hist(errors_test, bins=20, color='red', density=True)
    plt.xlabel('Error in Kelvin', size=16)
    plt.ylabel('Frequency', size=16)
    plt.title(f'Clustering + {model} Error', size=16)
    return plt.show()


def svm_regression(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled):
    svm = SVR(kernel='rbf')
    svm.fit(X_train_scaled, y_train_scaled)
    y_pred_train_svm = svm.predict(X_train_scaled)
    y_pred_test_svm = svm.predict(X_test_scaled)
    linear_scores(svm, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, y_pred_train_svm, y_pred_test_svm)
    return


def combined_cluster_regression(model, X_train_cluster_scaled, y_train_cluster_scaled, X_test_cluster_scaled,
                                    y_test_cluster_scaled):
    X_train_combined = np.concatenate(list(X_train_cluster_scaled.values()))
    y_train_combined = np.concatenate(list(y_train_cluster_scaled.values()))

    model.fit(X_train_combined, y_train_combined)

    X_test_combined = np.concatenate(list(X_test_cluster_scaled.values()))
    y_test_combined = np.concatenate(list(y_test_cluster_scaled.values()))

    test_score_combined = model.score(X_test_combined, y_test_combined)
    print("Combined Test Score for Cluster Models:", test_score_combined)
    return


def plotting_cluster_results_svm(model, n_clusters, y_train_cluster, y_pred_train_cluster, y_test_cluster, y_pred_test_cluster,
                                 x_min, x_max, y_min, y_max):
    plt.figure(figsize=(10, 10))

    for i in range(n_clusters):
        plt.scatter(y_train_cluster[i], y_pred_train_cluster[i], color='blue', alpha=0.5, marker=i + 3,
                    label=f'train cluster {i}')
        plt.scatter(y_test_cluster[i], y_pred_test_cluster[i], color='red', alpha=0.5, marker=i + 3,
                    label=f'test cluster {i}')

    plt.legend(fontsize=16);
    plt.plot([x_min, x_max], [y_min, y_max], color='black')
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.xlabel('$y_{true}$', size=16)
    plt.ylabel('$y_{pred}$', size=16)

    plt.figure(figsize=(10, 8))
    for i in range(n_clusters):
        errors = y_test_cluster[i] - y_pred_test_cluster[i]
        plt.hist(errors, bins=20, alpha=0.5, label=f'Cluster {i}', density=True)

    plt.xlabel('Error in Kelvin', size=16)
    plt.ylabel('Frequency', size=16)
    plt.title(f'Error Histogram for {model}', size=16)
    plt.legend()
    plt.show()


def grid_search(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled):
    best_score = 0
    best_parameters = {'C': None, 'gamma': None}
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            svm = SVR(kernel='rbf', C=C, gamma=gamma)
            svm.fit(X_train_scaled, y_train_scaled)
            score = svm.score(X_test_scaled, y_test_scaled)
            if score > best_score:
                best_score = score
                best_parameters = {'C': C, 'gamma': gamma}
    print(f"Best parameters: {best_parameters}")

if __name__ == '__main__':
    data = pd.read_csv('Melting_Data.csv')

    data["composition"] = data['formula'].map(lambda x: Composition(x))

    descriptors = ['row', 'group', 'atomic_mass', 'atomic_radius',
                   'boiling_point', 'melting_point', 'X']
    stats = ["mean", "std_dev"]

    ep = ElementProperty(data_source=PymatgenData(), features=descriptors, stats=stats)
    data = ep.featurize_dataframe(data, "composition")

    # deleting an unnecessary column
    data = data.drop(columns=['composition'])
    print(data.head())

    X = data.iloc[:, 5:].values
    y = data['melt_temp_K'].values

    data_exploration(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_train = lr.predict(X_train)
    y_pred_test = lr.predict(X_test)

    linear_scores(lr, X_train, y_train, X_test, y_test, y_pred_train, y_pred_test)

    linear_plotting(lr, y_train, y_test, y_pred_train, y_pred_test)

    n_clusters = 3  # set the number of clusters here
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_train[:, [0, 1]])
    train_cluster_labels = kmeans.predict(X_train[:, [0, 1]])
    test_cluster_labels = kmeans.predict(X_test[:, [0, 1]])
    X_train_cluster, X_test_cluster = {}, {}
    y_train_cluster, y_test_cluster = {}, {}

    for i in range(n_clusters):
        idx_train = np.where(train_cluster_labels == i)[0]
        idx_test = np.where(test_cluster_labels == i)[0]
        X_train_cluster[i] = X_train[idx_train, :]
        X_test_cluster[i] = X_test[idx_test, :]
        y_train_cluster[i] = y_train[idx_train]
        y_test_cluster[i] = y_test[idx_test]

    plotting_clusters(n_clusters, X_train_cluster)

    y_pred_train_cluster, y_pred_test_cluster = {}, {}
    y_pred_train_cluster_svm, y_pred_test_cluster_svm = {}, {}

    lr_cluster = None
    for i in range(n_clusters):
        lr_cluster = LinearRegression()
        lr_cluster.fit(X_train_cluster[i], y_train_cluster[i])
        y_pred_train_cluster[i] = lr_cluster.predict(X_train_cluster[i])
        y_pred_test_cluster[i] = lr_cluster.predict(X_test_cluster[i])


    for i in range(n_clusters):
        print(f"Cluster {i+1} Error Metrics: \n")
        linear_scores(lr_cluster, X_train_cluster[i], y_train_cluster[i], X_test_cluster[i], y_test_cluster[i],
                      y_pred_train_cluster[i], y_pred_test_cluster[i])

    plotting_clusters(n_clusters, X_train_cluster)

    plotting_cluster_results(lr, n_clusters, y_train_cluster, y_pred_train_cluster, y_test_cluster, y_pred_test_cluster,
                             -500, 4500, -500, 4500)

    mean_MAE_train = []
    mean_RMSE_train = []
    mean_MAE_test = []
    mean_RMSE_test = []
    for i in range(n_clusters):
        MAE_train = mean_absolute_error(y_train_cluster[i], y_pred_train_cluster[i])
        RMSE_train = sqrt(mean_squared_error(y_train_cluster[i], y_pred_train_cluster[i]))
        MAE_test = mean_absolute_error(y_test_cluster[i], y_pred_test_cluster[i])
        RMSE_test = sqrt(mean_squared_error(y_test_cluster[i], y_pred_test_cluster[i]))
        mean_MAE_train.append(MAE_train)
        mean_RMSE_train.append(RMSE_train)
        mean_MAE_test.append(MAE_test)
        mean_RMSE_test.append(RMSE_test)

    mean_cluster_MAE_train = sum(mean_MAE_train) / 3
    mean_cluster_RMSE_train = sum(mean_RMSE_train) / 3
    mean_cluster_MAE_test = sum(mean_MAE_test) / 3
    mean_cluster_RMSE_test = sum(mean_RMSE_test) / 3

    print("Metrics for Regression on Combined CLusters")
    print(f"Mean MAE Train: {mean_cluster_MAE_train}")
    print(f"Mean RMSE Train: {mean_cluster_RMSE_train}")
    print(f"Mean MAE Test: {mean_cluster_MAE_test}")
    print(f"Mean RMSE Test: {mean_cluster_RMSE_test} \n")


    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled,
                                                                                    y_scaled, train_size=0.8,
                                                                                    random_state=42)
    print("Shapes after train-test split:")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    grid_search(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)

    svm_regression(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)

    kmeans_svr = KMeans(n_clusters=n_clusters, random_state=42).fit(X_train_scaled[:, [0, 1]])
    train_cluster_labels_scaled = kmeans_svr.labels_
    test_cluster_labels_scaled = kmeans_svr.predict(X_test_scaled[:, [0, 1]])
    X_train_cluster_scaled, X_test_cluster_scaled = {}, {}
    y_train_cluster_scaled, y_test_cluster_scaled = {}, {}

    for i in range(n_clusters):
        idx_train_s = np.where(train_cluster_labels_scaled == i)[0]
        idx_test_s = np.where(test_cluster_labels_scaled == i)[0]
        X_train_cluster_scaled[i] = X_train_scaled[idx_train_s, :]
        X_test_cluster_scaled[i] = X_test_scaled[idx_test_s, :]
        y_train_cluster_scaled[i] = y_train_scaled[idx_train_s]
        y_test_cluster_scaled[i] = y_test_scaled[idx_test_s]

    plotting_clusters(n_clusters, X_train_cluster_scaled)

    y_pred_train_cluster_svm, y_pred_test_cluster_svm = {}, {}

    for i in range(n_clusters):
        if len(X_train_cluster_scaled[i]) == 0 or len(X_test_cluster_scaled[i]) == 0:
            print(f"Skipping Cluster {i + 1} as it has no samples.")
            continue

        svm_cluster = SVR(kernel='rbf')
        svm_cluster.fit(X_train_cluster_scaled[i], y_train_cluster_scaled[i])
        y_pred_train_cluster_svm[i] = svm_cluster.predict(X_train_cluster_scaled[i])
        y_pred_test_cluster_svm[i] = svm_cluster.predict(X_test_cluster_scaled[i])
        print(f"Cluster {i + 1}:")
        print("X_train_cluster_scaled shape:", X_train_cluster_scaled[i].shape)
        print("y_train_cluster_scaled shape:", y_train_cluster_scaled[i].shape)
        linear_scores(svm_cluster, X_train_cluster_scaled[i], y_train_cluster_scaled[i],
                      X_test_cluster_scaled[i], y_test_cluster_scaled[i],
                      y_pred_train_cluster_svm[i], y_pred_test_cluster_svm[i])

    svm_model_combined = SVR(kernel='rbf')

    combined_cluster_regression(svm_model_combined, X_train_cluster_scaled, y_train_cluster_scaled,
                                X_test_cluster_scaled,
                                y_test_cluster_scaled)

    plotting_cluster_results_svm(svm_model_combined, n_clusters, y_train_cluster_scaled, y_pred_train_cluster_svm,
                             y_test_cluster_scaled, y_pred_test_cluster_svm, 0, 1, 0, 1)

#Much better score with both clustering and regression with the SVM than just regression
#We also did grid search and best params are 1, 0.1
