from math import sqrt

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import BaggingClassifier
import seaborn as sns


def pca():
    path = "Hotel Reservations.csv"

    data = pd.read_csv(path)

    label_encoder = LabelEncoder()
    categorical_columns = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']

    for column in categorical_columns:
        data[column] = label_encoder.fit_transform(data[column])
    # plt.figure(figsize=(15, 15))
    # sns.heatmap(data.corr(), annot=True, fmt=".2f")
    # plt.show()
    data["booking_status"] = data["booking_status"].replace({"Not_Canceled": 0, "Canceled": 1})
    data.drop(columns=["Booking_ID", "room_type_reserved", "arrival_year", "arrival_date",
                       "arrival_month", "no_of_previous_cancellations", "type_of_meal_plan",
                       "no_of_previous_bookings_not_canceled", "required_car_parking_space"], inplace=True)

    pca = PCA(n_components=1)
    pca_no_people = pca.fit_transform(data[["no_of_adults", "no_of_children"]])
    data["no_of_people"] = pca_no_people
    data.drop(columns=["no_of_adults", "no_of_children"], inplace=True)

    pca_no_week = pca.fit_transform(data[["no_of_weekend_nights", "no_of_week_nights"]])
    data["no_of_week_days"] = pca_no_week
    data.drop(columns=["no_of_weekend_nights", "no_of_week_nights"], inplace=True)

    X = data.drop('booking_status', axis=1).values
    y = data['booking_status'].values

    # plt.figure(figsize=(15, 15))
    # sns.heatmap(data.corr(), annot=True, fmt=".2f")
    # plt.show()

    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=0)
    X, y = ros.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=987987)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA()
    pca.fit(X_train)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    optimal_n_components = np.argmax(cumulative_variance > 0.95) + 1
    print(f"optimal_n_components {optimal_n_components}")
    pca = PCA(n_components=optimal_n_components)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # from sklearn.decomposition import SparsePCA
    # sparse_pca = SparsePCA(n_components=3, alpha=0.5, random_state=3232)
    # X_train = sparse_pca.fit_transform(X_train)
    # X_test = sparse_pca.transform(X_test)

    knn_model = KNeighborsClassifier(
        n_neighbors=29,
        weights="distance",
        algorithm="kd_tree",
        leaf_size=4,
        p=1,
        metric="manhattan"
    )
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = sqrt(test_mse)
    print(f"rmse: {test_rmse:.4f}")
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"f1 is {f1:.4f}")


    # bagging_model = BaggingClassifier(knn_model, n_estimators=11, max_features=5, random_state=3920)
    #
    # bagging_model.fit(X_train, y_train)
    #
    # y_pred_bagging = bagging_model.predict(X_test)
    # accuracy_bagging = accuracy_score(y_test, y_pred_bagging)
    # print(f"Accuracy with bagging: {accuracy_bagging:.4f}")
    #
    # test_mse = mean_squared_error(y_test, y_pred_bagging)
    # test_rmse = sqrt(test_mse)
    # print(f"rmse with bagging: {test_rmse:.4f}")
    #
    # f1 = f1_score(y_test, y_pred_bagging, average='weighted')
    # print(f"f1 is {f1:.4f}")


def pca_pipeline():
    path = "Hotel Reservations.csv"

    data = pd.read_csv(path)

    data.drop('Booking_ID', axis=1, inplace=True)
    label_encoder = LabelEncoder()

    data['market_segment_type_encoded'] = label_encoder.fit_transform(data['market_segment_type'])
    data = data.drop('market_segment_type', axis=1)

    data["type_of_meal_plan_encoded"] = label_encoder.fit_transform(data["type_of_meal_plan"])
    data = data.drop("type_of_meal_plan", axis=1)

    data["room_type_reserved_encoded"] = label_encoder.fit_transform(data["room_type_reserved"])
    data = data.drop("room_type_reserved", axis=1)

    # Replace "Not_Canceled" with 0 and "Canceled" with 1
    data["booking_status"] = data["booking_status"].replace({"Not_Canceled": 0, "Canceled": 1})

    # Split the dataset into features and labels
    X = data.drop('booking_status', axis=1).values
    y = data['booking_status'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=987987)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=16)),
        ('knn', KNeighborsClassifier(n_neighbors=29,
                                     weights="distance",
                                     algorithm="kd_tree",
                                     leaf_size=4,
                                     p=1,
                                     metric="manhattan"))
    ])

    # Train the pipeline (preprocessing, PCA, and kNN)
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    accuracy = pipeline.score(X_test, y_test)
    print(f'Accuracy: {accuracy:.4f}')


if __name__ == "__main__":
    # pca_analysis("Hotel Reservations.csv", 'booking_status', test_size=0.2)

    pca()
    #
    # pca_pipeline()
