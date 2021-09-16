import math

import pandas as pd

import Performance_class
import Song_Class
import numpy as np
import pretty_midi
import os
import shutil
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb

import pickle


def trainAndTest(train_one_dim, test, cnt, to_print=False, model_features="All"):
    one_dim_scores = test_algorithms_next_step(train_one_dim, test, True, cnt, "xgb", to_print)
    overall_scores = test_algorithms_scores(train_one_dim, test, "Overall", cnt, "rf_entropy", 4, to_print=to_print)

    tempo_scores = test_algorithms_scores(train_one_dim, test, "Tempo", cnt, "rf_gini", 4, to_print=to_print,
                                          model_features=model_features)
    rhythm_scores = test_algorithms_scores(train_one_dim, test, "Rhythm", cnt, "rf_gini", 4, to_print=to_print,
                                           model_features=model_features)
    a_d_scores = test_algorithms_scores(train_one_dim, test, "Articulation & Dynamics", cnt, "xgb", 4,
                                        to_print=to_print,
                                        model_features=model_features)

    if model_features == "All" or model_features == "Only":
        pitch_scores = test_algorithms_scores(train_one_dim, test, "Pitch", cnt, "xgb", 4, to_print=to_print,
                                              model_features=model_features)
    else:
        pitch_scores = test_algorithms_scores(train_one_dim, test, "Pitch", cnt, None, 4, to_print=to_print)

    return one_dim_scores, pitch_scores, tempo_scores, rhythm_scores, a_d_scores, overall_scores


def test_algorithms_next_step(labeled_data_train, labeled_data_test, with_tempo, cnt, chosen_model_name,
                                            to_print=True):
    if with_tempo:
        x_train = pd.DataFrame(labeled_data_train[["Pitch", "Tempo", 'Rhythm', 'Articulation', 'Dynamics']])
        y_train = labeled_data_train['label']

        x_test = pd.DataFrame(labeled_data_test[["Pitch", "Tempo", 'Rhythm', 'Articulation', 'Dynamics']])
        y_test = labeled_data_test['label']

    else:
        x_train = labeled_data_train.drop(columns=["Teacher's Pitch", "Teacher's Tempo", "Teacher's Rhythm",
                                                   "Teacher's Articulation & Dynamics", 'Tempo', "Teacher's Overall",
                                                   'label'])
        y_train = labeled_data_train['label']

        x_test = labeled_data_test.drop(columns=["Teacher's Pitch", "Teacher's Tempo", "Teacher's Rhythm",
                                                 "Teacher's Articulation & Dynamics", 'Tempo', "Teacher's Overall",
                                                 'label'])
        y_test = labeled_data_test['label']

    ### random forest

    model_rf_gini = RandomForestClassifier(criterion='gini')
    model_rf_gini.fit(x_train, y_train)
    random_forest_gini_score = model_score_main(model_rf_gini, x_test, y_test)

    model_rf_entropy = RandomForestClassifier(criterion='entropy')
    model_rf_entropy.fit(x_train, y_train)
    random_forest_entropy_score = model_score_main(model_rf_entropy, x_test, y_test)

    ### logistic regression (classification)

    model_lr = LogisticRegression(max_iter=500)
    model_lr.fit(x_train, y_train)
    logistic_regression_score = model_score_main(model_lr, x_test, y_test)

    ### knn (classification)
    max_knn_score = 0
    max_knn_val = 0
    knn_x = [x for x in range(3, 7)]
    knn_y = []
    for i in range(4, min(len(x_train) + 1, 7)):
        model_knn = KNeighborsClassifier(n_neighbors=i)
        model_knn.fit(x_train, y_train)
        knn_score = model_score_main(model_knn, x_test, y_test)
        knn_y.append(knn_score)
        if knn_score > max_knn_score:
            max_knn_score = knn_score
            max_knn_val = i
    # plt.plot(knn_x, knn_y)
    # plt.show()

    ### MLP (classification)
    model_mlp = MLPClassifier(max_iter=199)
    model_mlp.fit(x_train, y_train)
    mlp_score = model_score_main(model_mlp, x_test, y_test)

    ### gradient boosting (xgb)
    model_xgb = xgb.XGBClassifier()
    model_xgb.fit(x_train, y_train)
    predict = model_xgb.predict(x_test)
    xgb_score = accuracy_score(y_test, predict)

    if to_print:
        print(" ")
        print("###########")
        print("One dimension results:")
        print("Random Forest (gini) Score: " + str(random_forest_gini_score))
        print("Random Forest (entropy) Score: " + str(random_forest_entropy_score))
        print("Logistic Regression Score: " + str(logistic_regression_score))
        print("KNN with k = " + str(max_knn_val) + " Score: " + str(max_knn_score))
        print("Multi-layer Perceptron with Neural Networks score: " + str(mlp_score))
        print("XGB score: " + str(xgb_score))
        print("###########")
        print(" ")

    models = {"rf_gini": model_rf_gini, "rf_entropy": model_rf_entropy, "lr": model_lr, "knn": model_knn,
              "mlp": model_mlp, "xgb": model_xgb}
    if chosen_model_name is not None:
        chosen_model = models[chosen_model_name]
        filename = 'models/label_one_dim/model_' + str(cnt) + '.sav'
        with open(filename, 'wb') as f:
            pickle.dump(chosen_model, f)

    return random_forest_gini_score, random_forest_entropy_score, logistic_regression_score, max_knn_score, mlp_score, max_knn_val, xgb_score


def test_algorithms_scores(labeled_data_train, labeled_data_test, feature_name, cnt, chosen_model_name, chosen_k,
                           model_features="All", to_print=True):
    x_train, x_test = extract_features_for_model(labeled_data_train, labeled_data_test, feature_name, model_features)
    y_train = labeled_data_train["Teacher's " + feature_name]
    y_test = labeled_data_test["Teacher's " + feature_name]

    ### random forest

    model_rf_gini = RandomForestClassifier(criterion='gini')
    model_rf_gini.fit(x_train, y_train)
    random_forest_gini_score = model_score_main(model_rf_gini, x_test, y_test)

    model_rf_entropy = RandomForestClassifier(criterion='entropy')
    model_rf_entropy.fit(x_train, y_train)
    random_forest_entropy_score = model_score_main(model_rf_entropy, x_test, y_test)

    ### logistic regression (classification)

    model_lr = LogisticRegression(max_iter=500)
    model_lr.fit(x_train, y_train)
    logistic_regression_score = model_score_main(model_lr, x_test, y_test)

    ### knn (classification)
    max_knn_val = chosen_k
    model_knn = KNeighborsClassifier(n_neighbors=chosen_k)
    model_knn.fit(x_train, y_train)
    max_knn_score = model_score_main(model_knn, x_test, y_test)

    ### MLP (classification)

    model_mlp = MLPClassifier(max_iter=199)
    model_mlp.fit(x_train, y_train)
    mlp_score = model_score_main(model_mlp, x_test, y_test)

    ### gradient boosting (xgb)
    model_xgb = xgb.XGBClassifier()
    model_xgb.fit(x_train, y_train)
    predict = model_xgb.predict(x_test)
    xgb_score = accuracy_score(y_test, predict)

    if to_print:
        print(" ")
        print("###########")
        print("Results for feature: " + feature_name)
        print("Random Forest (gini) Score: " + str(random_forest_gini_score))
        print("Random Forest (entropy) Score: " + str(random_forest_entropy_score))
        print("Logistic Regression Score: " + str(logistic_regression_score))
        print("KNN with k = " + str(max_knn_val) + " Score: " + str(max_knn_score))
        print("Multi-layer Perceptron with Neural Networks score: " + str(mlp_score))
        print("XGB score: " + str(xgb_score))
        print("###########")
        print(" ")

    models = {"rf_gini": model_rf_gini, "rf_entropy": model_rf_entropy, "lr": model_lr, "knn": model_knn,
              "mlp": model_mlp, "xgb": model_xgb}
    if chosen_model_name is not None:
        chosen_model = models[chosen_model_name]
        filename = 'models/' + feature_name + '/model_' + str(cnt) + '.sav'
        with open(filename, 'wb') as f:
            pickle.dump(chosen_model, f)

    return random_forest_gini_score, random_forest_entropy_score, logistic_regression_score, max_knn_score, mlp_score, max_knn_val, xgb_score


def extract_features_for_model(labeled_data_train, labeled_data_test, feature_name, model_features):
    if feature_name == "Overall" or feature_name == "Articulation & Dynamics":
        x_train = pd.DataFrame(labeled_data_train[['Pitch', 'Tempo', 'Rhythm', 'Articulation', 'Dynamics']])
        x_test = pd.DataFrame(labeled_data_test[['Pitch', 'Tempo', 'Rhythm', 'Articulation', 'Dynamics']])
    elif feature_name == "Pitch":
        x_train = pd.DataFrame(labeled_data_train["Pitch"])
        x_test = pd.DataFrame(labeled_data_test["Pitch"])
    elif model_features == "Timing":
        x_train = pd.DataFrame(labeled_data_train[['Tempo', 'Rhythm', 'Articulation']])
        x_test = pd.DataFrame(labeled_data_test[['Tempo', 'Rhythm', 'Articulation']])
    elif model_features == "Only":
        x_train = pd.DataFrame(labeled_data_train[[feature_name]])
        x_test = pd.DataFrame(labeled_data_test[[feature_name]])
    else:
        if feature_name == "Rhythm":
            x_train = pd.DataFrame(labeled_data_train[['Pitch', 'Rhythm']])
            x_test = pd.DataFrame(labeled_data_test[['Pitch', 'Rhythm']])
        else:
            x_train = pd.DataFrame(labeled_data_train[['Pitch', 'Tempo']])
            x_test = pd.DataFrame(labeled_data_test[['Pitch', 'Tempo']])
    return x_train, x_test


def model_score_main(model, x_test, y_test):
    cnt = 0
    for i in range(len(y_test)):
        y_hat = int(model.predict(x_test.iloc[i].to_numpy().reshape(1, -1))[0])
        if y_hat == y_test[i]:
            cnt += 1
    return cnt / len(x_test)
