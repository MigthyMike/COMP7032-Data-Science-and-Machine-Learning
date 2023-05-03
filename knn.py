from math import sqrt

import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def knn_1():
    path = "Hotel datas.csv"

    datas = pd.read_csv(path)

    datas.drop('Booking_ID', axis=1, inplace=True)

    datas = pd.get_dummies(datas, columns=["type_of_meal_plan"])

    datas = pd.get_dummies(datas, columns=["room_type_reserved"])

    datas = pd.get_dummies(datas, columns=["market_segment_type"])
    X = datas.drop('booking_status', axis=1)
    y = datas['booking_status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4393)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    k = 5
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)


def knn_2():
    """
    It takes only a lead time to to create  a model
    :return:
    """
    path = "Hotel Reservations.csv"

    datas = pd.read_csv(path)

    datas.drop('Booking_ID', axis=1, inplace=True)

    datas = pd.get_dummies(datas, columns=["type_of_meal_plan"])

    datas = pd.get_dummies(datas, columns=["room_type_reserved"])

    datas = pd.get_dummies(datas, columns=["market_segment_type"])

    X = datas.drop('booking_status', axis=1)  
    y = datas['booking_status']

    lead_time = datas['lead_time']
    X['lead_time'] = lead_time

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4393)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    k = 7
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)


def knn_3():
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

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    knn_model = KNeighborsClassifier(n_neighbors=29,
                                     weights="distance",
                                     algorithm="kd_tree",
                                     leaf_size=4,
                                     p=1,
                                     metric="manhattan")

    knn_model.fit(X_train, y_train)

    test_preds = knn_model.predict(X_test)

    # accuracy
    accuracy = accuracy_score(y_test, test_preds)
    print('Accuracy:', accuracy)

    # rmse
    mse = mean_squared_error(y_test, test_preds)
    rmse = sqrt(mse)
    print(f"rmse is {rmse}")

    f1 = f1_score(y_test, test_preds, average='weighted')
    print("FINAL F1 Score:", f1)

    # print("starting GridSearchCV")
    #
    # parameters = {
    #     "n_neighbors": list(range(10, 30)),
    #     "weights": ['uniform', 'distance'],
    #     "algorithm": ['ball_tree', 'kd_tree', 'brute'],
    #     "leaf_size": [2,3,4,5],
    #     "p": [1, 2],
    #     "metric": ["euclidean", "manhattan", "chebyshev", "minkowski", "wminkowski", "seuclidean", "mahalanobis"],
    # }
    # gridsearch = RandomizedSearchCV(KNeighborsClassifier(), parameters, scoring="accuracy", cv=5)
    # gridsearch.fit(X_train, y_train)
    #
    # print(f"best_param is {gridsearch.best_params_}")
    # print(f"search best is {gridsearch.best_score_}")

    # bagging
    print("bagging start")
    # for i in range(1, 17):
    # bagging_model = BaggingClassifier(knn_model, n_estimators=10, max_features=12)
    bagging_model = BaggingClassifier(knn_model, n_estimators=10, max_features=12)
    bagging_model.fit(X_train, y_train)

    # accuracy
    y_pred_bagging = bagging_model.predict(X_test)
    accuracy_bagging = accuracy_score(y_test, y_pred_bagging)
    print(f"Accuracy with bagging: {accuracy_bagging}")

    # rmse
    test_mse = mean_squared_error(y_test, y_pred_bagging)
    test_rmse = sqrt(test_mse)
    print(f"rmse with bagging: {test_rmse}")

    f1 = f1_score(y_test, y_pred_bagging, average='weighted')
    print("FINAL F1 Score with bagging:", f1)


def knn_vanilla():
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

    knn_model = KNeighborsClassifier()

    knn_model.fit(X_train, y_train)

    test_preds = knn_model.predict(X_test)

    # accuracy
    accuracy = accuracy_score(y_test, test_preds)
    print('Vanilla accuracy:', accuracy)

    # rmse
    mse = mean_squared_error(y_test, test_preds)
    rmse = sqrt(mse)
    print(f"Vanilla rmse is {rmse}")

    f1 = f1_score(y_test, test_preds, average='weighted')
    print("vannila F1 Score:", f1)


if __name__ == "__main__":
    knn_vanilla()
    knn_3()
