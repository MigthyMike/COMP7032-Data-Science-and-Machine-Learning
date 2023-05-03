from math import sqrt

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import f1_score

path = "Hotel Reservations.csv"
data = pd.read_csv(path)

# drop booking id
data.drop('Booking_ID', axis=1, inplace=True)

label_encoder = LabelEncoder()
categorical_columns = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'booking_status']

for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

X = data.drop('booking_status', axis=1)
y = data['booking_status'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30921)


def rf():
    # Replace "Not_Canceled" with 0 and "Canceled" with 1
    data["booking_status"] = data["booking_status"].replace({"Not_Canceled": 0, "Canceled": 1})

    # select the most important features
    selected_columns = ['lead_time', 'avg_price_per_room', 'no_of_special_requests',
                        'arrival_date', 'arrival_month', 'no_of_week_nights',
                        'market_segment_type', 'no_of_weekend_nights', 'arrival_year',
                        'no_of_adults', 'type_of_meal_plan', 'room_type_reserved',
                        'required_car_parking_space', 'no_of_children', 'repeated_guest']

    X = data[selected_columns].values
    y = data['booking_status'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30921)

    rf_model = RandomForestClassifier(n_estimators=100,
                                      random_state=203982,
                                      max_features=2,
                                      max_depth=26,
                                      min_samples_split=3,
                                      min_samples_leaf=1,
                                      bootstrap=True,
                                      criterion="entropy",
                                      class_weight="balanced")

    rf_model.fit(X_train, y_train)

    # accuracy
    rf_y_pred = rf_model.predict(X_test)
    print('Accuracy Score:', accuracy_score(y_test, rf_y_pred))
    # rmse
    mse = mean_squared_error(y_test, rf_y_pred)
    rmse = sqrt(mse)
    print(f"rmse is {rmse}")

    ### GridSearchCV
    rf_param_grid = {
        'max_features': ['sqrt', 'log2'] + list(range(1, 10)),
        'n_estimators': list(range(30, 35)),
        'max_depth': [30,31,32,33,34,35, None],
        'min_samples_split': [3,4,5,6,7],
        'min_samples_leaf': [1,2,3],
        'bootstrap': [True, False],
        'criterion': ['entropy'],
    }

    rf_grid = RandomizedSearchCV(RandomForestClassifier(), rf_param_grid, cv=5)
    rf_grid.fit(X_train, y_train)

    print('RF best Parameters:', rf_grid.best_estimator_)
    print('RF best Score:', rf_grid.best_score_)


def importance_features():
    rf_model = rf_without_selected_column()

    importances = rf_model.feature_importances_
    print(importances)

    # Get the indices of the features sorted by importance
    sorted_idx = np.argsort(importances)[::-1]

    # Get the names of the most important features
    top_n_features = 10
    selected_columns = data.columns[sorted_idx[:top_n_features]]
    print(selected_columns)

    X_selected = data[selected_columns].values
    X_train_selected, X_test_selected, y_train, y_test = train_test_split(X_selected, y, test_size=0.2,
                                                                          random_state=30921)

    rf_model = RandomForestClassifier(n_estimators=100,
                                      random_state=203982)

    rf_model.fit(X_train_selected, y_train)

    rf_y_pred = rf_model.predict(X_test_selected)
    print('Accuracy Score with feature selection:', accuracy_score(y_test, rf_y_pred))


def final_rf():
    # select the most important features
    selected_columns = ['lead_time',
                        'avg_price_per_room',
                        'no_of_special_requests',
                        'arrival_date',
                        'arrival_month',
                        'no_of_week_nights',
                        'market_segment_type',
                        'no_of_weekend_nights',
                        'arrival_year',
                        'no_of_adults',
                        'type_of_meal_plan',
                        'room_type_reserved',
                        'required_car_parking_space',
                        'no_of_children',
                        'repeated_guest'
                        ]

    # X = data.drop('booking_status', axis=1)
    X = data[selected_columns].values
    y = data['booking_status'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30921)

    rf_model = RandomForestClassifier(n_estimators=31,
                                      random_state=203982,
                                      max_depth=33,
                                      max_features=6,
                                      min_samples_split=5,
                                      min_samples_leaf=2,
                                      bootstrap=False,
                                      criterion="entropy",
                                      class_weight=None)

    rf_model.fit(X_train, y_train)

    # accuracy
    rf_y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, rf_y_pred)
    print(f'FINAL Accuracy Score: {accuracy}')
    # rmse
    mse = mean_squared_error(y_test, rf_y_pred)
    rmse = sqrt(mse)
    print(f"FINAL rmse is {rmse}")

    f1 = f1_score(y_test, rf_y_pred, average='weighted')
    print("FINAL F1 Score:", f1)

    # tree_to_visualize = rf_model.estimators_[0]
    # plt.figure(figsize=(20, 10))
    # plot_tree(tree_to_visualize, feature_names=selected_columns, class_names=['Not Booked', 'Booked'], filled=True)
    # plt.show()


def rf_without_selected_column():
    rf_model = RandomForestClassifier(n_estimators=92,
                                      random_state=203982,
                                      max_depth=33,
                                      max_features=6,
                                      min_samples_split=5,
                                      min_samples_leaf=2,
                                      bootstrap=False,
                                      criterion="entropy",
                                      class_weight=None)

    rf_model.fit(X_train, y_train)

    # accuracy
    rf_y_pred = rf_model.predict(X_test)
    print(f"Accuracy Score without column selection: {accuracy_score(y_test, rf_y_pred)}")

    # rmse
    mse = mean_squared_error(y_test, rf_y_pred)
    rmse = sqrt(mse)
    print(f"rmse without column selection is: {rmse}")

    # f1 score
    f1 = f1_score(y_test, rf_y_pred, average='weighted')
    print("F1 without column selection Score:", f1)

    return rf_model


def rf_vanilla():
    rf_model = RandomForestClassifier(random_state=203982)

    rf_model.fit(X_train, y_train)

    # accuracy
    rf_y_pred = rf_model.predict(X_test)
    print(f"Accuracy Score vanila: {accuracy_score(y_test, rf_y_pred)}")

    # rmse
    mse = mean_squared_error(y_test, rf_y_pred)
    rmse = sqrt(mse)
    print(f"rmse is: {rmse}")

    f1 = f1_score(y_test, rf_y_pred, average='weighted')
    print("F1 baseline Score:", f1)

    return rf_model


def create_rf(max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, criterion, _class_weight,
              n_estimators):
    rf_model = RandomForestClassifier(n_estimators=n_estimators,
                                      random_state=203982,
                                      max_depth=max_depth,
                                      min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf,
                                      bootstrap=bootstrap,
                                      max_features=max_features,
                                      criterion=criterion,
                                      class_weight=_class_weight
                                      )

    rf_model.fit(X_train, y_train)

    # accuracy
    rf_y_pred = rf_model.predict(X_test)
    # print('Accuracy Score:', accuracy_score(y_test, rf_y_pred))

    # rmse
    mse = mean_squared_error(y_test, rf_y_pred)
    rmse = sqrt(mse)
    # print(f"rmse is {rmse}")

    return accuracy_score(y_test, rf_y_pred), rmse


if __name__ == "__main__":
    # print("Accuracy Score vanila: 0.9128876636802206 rmse without column selection is: 0.2951479905399653")

    rf_vanilla()
    rf_without_selected_column()
    final_rf()

    # importance_features()

    # rf()

    # for i in range(1, 50):
    #     accuracy, rmse = create_rf(33, 5, 2, 6,False, "entropy", None, i)
    #     result = f"n estimator: {i} accuracy: {accuracy}, rmse: {rmse}\n"
    #     print(result)

    # max_features = ['auto', 'sqrt', 'log2'] + list(range(1, 20))
    # n_estimators = [10, 50, 100, 200, 500]
    # max_depth = [29, 30, 31,32,33,34, None]
    # bootstrap = [True, False]
    # criterion = ['gini', 'entropy']
    # class_weight = [None, 'balanced']
    # n_estimators = [10, 50, 100, 200, 500]
    # with open("results.txt", "a") as file:
    #     for n_estimator in n_estimators:
    #         for _class_weight in class_weight:
    #             for _criterion in criterion:
    #                 for _bootstrap in bootstrap:
    #                     for feature in max_features:
    #                         for min_samples_split in range(2, 6):
    #                             for min_samples_leaf in range(1, 3):
    #                                 for depth in max_depth:
    #                                     accuracy, rmse = create_rf(depth, min_samples_split, min_samples_leaf, feature,
    #                                                                _bootstrap, _criterion, _class_weight, n_estimator)
    #                                     result = (f"max_depth: {depth}, min_samples_split: {min_samples_split}, "
    #                                               f"min_samples_leaf: {min_samples_leaf}, max feature: {feature}, "
    #                                               f"bootstrap: {_bootstrap}, _criterion: {_criterion}, "
    #                                               f"class_weight:{_class_weight}, n_estimators: {n_estimator} "
    #                                               f"accuracy: {accuracy}, rmse: {rmse}\n")
    #                                     print(result)
    #                                     file.write(result)
