import math

import pandas as pd

import Performance_class
import Song_Class
import numpy as np
import pretty_midi
import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def change_midi_file_tempo(original_path, new_path, percentage=0.25):
    performance = Performance_class.Performance(original_path, " ", " ",
                                                original_path)

    if percentage > 0:
        performance.mistakes_generator("rhythm", noise=percentage)
        performance.mistakes_generator("duration", noise=percentage, original=False)
    else:
        performance.mistakes_generator("duration", noise=percentage)
        performance.mistakes_generator("rhythm", noise=percentage, original=False)

    np2mid(performance.midi_df, new_path, None, True)
    return new_path


def generate_random_mistakes_data(folder, n, create_midi_files):
    basepath = folder + '/'
    all_data = []
    if create_midi_files:
        fake_data_path = folder + ' - fake data/'
        Path(fake_data_path).mkdir(exist_ok=True)
    with os.scandir(basepath) as songs:
        for song in songs:
            song_name = song.name.split(".")[0]
            if song.is_file() and song.name != '.DS_Store':
                song_instance = Song_Class.Song(song_name)
                flawed_performances, original_midi_data = create_random_mistakes(basepath + song.name, song_name, n,
                                                                                 min_noise=0, max_noise=0.9,
                                                                                 min_percentage=0.4, max_percentage=1)
                if create_midi_files:
                    Path(fake_data_path + song_name).mkdir(exist_ok=True)
                    shutil.copy(basepath + song.name, fake_data_path + song_name)
                    Path(fake_data_path + song_name + '/fake performances/').mkdir(exist_ok=True)
                for i, data in enumerate(flawed_performances):
                    if create_midi_files:
                        path = fake_data_path + song_name + '/fake performances/' + song_name + str(i) + ".mid"
                        np2mid(data.midi_df, path, original_midi_file=None, write_midi_file=True)
                    else:
                        fake_data_performance = np2mid(data.midi_df, song_name, original_midi_data[i],
                                                       write_midi_file=False)
                    song_instance.fake_performances.append(fake_data_performance)
            all_data.append(song_instance)
    return all_data


def create_random_mistakes(path, name, n, max_noise, max_percentage, min_noise=0, min_percentage=0.5):
    flawed_performances = []
    original_midi_data = []
    for i in range(n):
        performance = Performance_class.Performance(path, name, name + " random mistakes: " + str(i), path)
        original_midi_data.append(performance.midi_data_original)
        performance.mistakes_generator("rhythm", np.random.uniform(min_noise, max_noise),
                                       np.random.uniform(min_percentage, max_percentage))
        performance.mistakes_generator("duration", np.random.uniform(min_noise, max_noise),
                                       np.random.uniform(min_percentage, max_percentage), False)
        performance.mistakes_generator("velocity", np.random.uniform(min_noise, max_noise),
                                       np.random.uniform(min_percentage, max_percentage), False)
        performance.mistakes_generator("pitch", np.random.uniform(min_noise, max_noise),
                                       np.random.uniform(min_percentage, max_percentage), False)
        flawed_performances.append(performance)

    return flawed_performances, original_midi_data


def np2mid(np_performance, midfilename, original_midi_file, write_midi_file):
    """
    Converts an numpy array  to a .mid file

    @param np_performance: np array with Midi values
    @param midfilename: full path to the mid output file
    @return: None
    """

    performance = pretty_midi.PrettyMIDI()

    piano = pretty_midi.Instrument(program=4)
    # Iterate over note names, which will be converted to note number later
    for m in np_performance:
        note = pretty_midi.Note(velocity=int(m[3]), pitch=int(m[2]), start=m[0], end=m[1])
        piano.notes.append(note)
    performance.instruments.append(piano)
    if write_midi_file:
        performance.write(midfilename)
    else:
        return_performance = Performance_class.Performance(path=None, name=midfilename, player_name="np2mid",
                                                           original_path=None,
                                                           prettyMidiFile_performance=performance,
                                                           prettyMidiFile_original=original_midi_file)
        return return_performance


def test_algorithms_next_step_one_dimension(labeled_data_train, labeled_data_test, with_tempo, to_print=True):
    if with_tempo:
        x_train = labeled_data_train.drop(columns=["Teacher's Pitch", "Teacher's Rhythm", "Teacher's Tempo",
                                                   "Teacher's Articulation & Dynamics", 'label'])
        y_train = labeled_data_train['label']

        x_test = labeled_data_test.drop(columns=["Teacher's Pitch", "Teacher's Rhythm", "Teacher's Tempo",
                                                 "Teacher's Articulation & Dynamics", 'label'])
        y_test = labeled_data_test['label']

    else:
        x_train = labeled_data_train.drop(columns=["Teacher's Pitch", "Teacher's Rhythm", "Teacher's Tempo",
                                                   "Teacher's Articulation & Dynamics", 'Tempo', 'label'])
        y_train = labeled_data_train['label']

        x_test = labeled_data_test.drop(columns=["Teacher's Pitch", "Teacher's Rhythm", "Teacher's Tempo",
                                                 "Teacher's Articulation & Dynamics", 'Tempo', 'label'])
        y_test = labeled_data_test['label']

    ### random forest

    model_rf_gini = RandomForestClassifier(criterion='gini')
    model_rf_gini.fit(x_train, y_train)
    random_forest_gini_score = score_test(model_rf_gini, x_test, y_test)

    model_rf_entropy = RandomForestClassifier(criterion='entropy')
    model_rf_entropy.fit(x_train, y_train)
    random_forest_entropy_score = score_test(model_rf_entropy, x_test, y_test)

    ### logistic regression (classification)

    model_lr = LogisticRegression(max_iter=500)
    model_lr.fit(x_train, y_train)
    logistic_regression_score = score_test(model_lr, x_test, y_test)

    ### knn (classification)
    max_knn_score = 0
    max_knn_val = 0
    knn_x = [x for x in range(1, 10)]
    knn_y = []
    for i in range(1, 10):
        model_knn = KNeighborsClassifier(n_neighbors=i)
        model_knn.fit(x_train, y_train)
        knn_score = score_test(model_knn, x_test, y_test)
        knn_y.append(knn_score)
        if knn_score > max_knn_score:
            max_knn_score = knn_score
            max_knn_val = i
    #plt.plot(knn_x, knn_y)
    #plt.show()

    model_mlp = MLPClassifier(max_iter=199)
    model_mlp.fit(x_train, y_train)
    mlp_score = score_test(model_mlp, x_test, y_test)
    ### MLP (classification)

    if to_print:
        print(" ")
        print("###########")
        print("One dimension results:")
        print("Random Forest (gini) Score: " + str(random_forest_gini_score))
        print("Random Forest (entropy) Score: " + str(random_forest_entropy_score))
        print("Logistic Regression Score: " + str(logistic_regression_score))
        print("KNN with k = " + str(max_knn_val) + " Score: " + str(max_knn_score))
        print("Multi-layer Perceptron with Neural Networks score: " + str(mlp_score))
        print("###########")
        print(" ")

    return [random_forest_gini_score, random_forest_entropy_score, logistic_regression_score, max_knn_score, mlp_score]


def test_algorithms_next_step_two_dimensions(labeled_data_train, labeled_data_test, with_tempo, to_print=True):
    if with_tempo:
        x_train = labeled_data_train.drop(columns=["Teacher's Pitch", "Teacher's Rhythm", "Teacher's Tempo",
                                                   "Teacher's Articulation & Dynamics",
                                                   'label dim 1', 'label dim 2'])
        y_train_1 = labeled_data_train['label dim 1']
        y_train_2 = labeled_data_train['label dim 2']

        x_test = labeled_data_test.drop(columns=["Teacher's Pitch", "Teacher's Rhythm", "Teacher's Tempo",
                                                 "Teacher's Articulation & Dynamics",
                                                 'label'])
        y_test = labeled_data_test['label']

        # if to_print:
        #     print("##############")
        #     print("With Tempo:")
        #     print("##############")
    else:
        x_train = labeled_data_train.drop(columns=["Teacher's Pitch", "Teacher's Rhythm", "Teacher's Tempo",
                                                   "Teacher's Articulation & Dynamics", 'Tempo',
                                                   'label dim 1', 'label dim 2'])
        y_train_1 = labeled_data_train['label dim 1']
        y_train_2 = labeled_data_train['label dim 2']

        x_test = labeled_data_test.drop(columns=["Teacher's Pitch", "Teacher's Rhythm", "Teacher's Tempo",
                                                 "Teacher's Articulation & Dynamics", 'Tempo',
                                                 'label'])
        y_test = labeled_data_test['label']

    ### random forest

    model_rf_gini_1 = RandomForestClassifier(criterion='gini')
    model_rf_gini_2 = RandomForestClassifier(criterion='gini')
    model_rf_gini_1.fit(x_train, y_train_1)
    model_rf_gini_2.fit(x_train, y_train_2)
    random_forest_gini_score = model_score(model_rf_gini_1, model_rf_gini_2, x_test, y_test)

    model_rf_entropy_1 = RandomForestClassifier(criterion='entropy')
    model_rf_entropy_2 = RandomForestClassifier(criterion='entropy')
    model_rf_entropy_1.fit(x_train, y_train_1)
    model_rf_entropy_2.fit(x_train, y_train_2)
    random_forest_entropy_score = model_score(model_rf_entropy_1, model_rf_entropy_2, x_test, y_test)

    ### logistic regression (classification)

    model_lr_1 = LogisticRegression(max_iter=500)
    model_lr_2 = LogisticRegression(max_iter=500)
    model_lr_1.fit(x_train, y_train_1)
    model_lr_2.fit(x_train, y_train_2)
    logistic_regression_score = model_score(model_lr_1, model_lr_2, x_test, y_test)

    ### knn (classification)
    max_knn_score = 0
    max_knn_val = 0
    for i in range(1, 10):
        model_knn_1 = KNeighborsClassifier(n_neighbors=i)
        model_knn_2 = KNeighborsClassifier(n_neighbors=i)
        model_knn_1.fit(x_train, y_train_1)
        model_knn_2.fit(x_train, y_train_2)
        knn_score = model_score(model_knn_1, model_knn_2, x_test, y_test)
        if knn_score > max_knn_score:
            max_knn_score = knn_score
            max_knn_val = i

    model_mlp_1 = MLPClassifier(max_iter=199)
    model_mlp_2 = MLPClassifier(max_iter=199)
    model_mlp_1.fit(x_train, y_train_1)
    model_mlp_2.fit(x_train, y_train_2)
    mlp_score = model_score(model_mlp_1, model_mlp_2, x_test, y_test)

    ### MLP (classification)

    if to_print:
        print(" ")
        print("###########")
        print("Two dimensions results:")
        print("Random Forest (gini) Score: " + str(random_forest_gini_score))
        print("Random Forest (entropy) Score: " + str(random_forest_entropy_score))
        print("Logistic Regression Score: " + str(logistic_regression_score))
        print("KNN with k = " + str(max_knn_val) + " Score: " + str(max_knn_score))
        print("Multi-layer Perceptron with Neural Networks score: " + str(mlp_score))
        print("###########")
        print(" ")

    return random_forest_gini_score, random_forest_entropy_score, logistic_regression_score, max_knn_score, mlp_score


def test_algorithms_scores(labeled_data_train, labeled_data_test, feature_name, to_print=True):
    if feature_name == "Articulation & Dynamics":
        x_train = labeled_data_train[['Articulation', 'Dynamics']]
        x_test = labeled_data_test[['Articulation', 'Dynamics']]
    # elif feature_name == "Rhythm" or feature_name == "Tempo":
    #     x_train = labeled_data_train[['Articulation', 'Rhythm', 'Tempo']]
    #     x_test = labeled_data_test[['Articulation', 'Rhythm', 'Tempo']]
    else:
        x_train = pd.DataFrame(labeled_data_train[feature_name])
        x_test = pd.DataFrame(labeled_data_test[feature_name])

    y_train = labeled_data_train["Teacher's " + feature_name]
    y_test = labeled_data_test["Teacher's " + feature_name]

    ### random forest

    model_rf_gini = RandomForestClassifier(criterion='gini')
    model_rf_gini.fit(x_train, y_train)
    random_forest_gini_score = score_test(model_rf_gini, x_test, y_test)

    model_rf_entropy = RandomForestClassifier(criterion='entropy')
    model_rf_entropy.fit(x_train, y_train)
    random_forest_entropy_score = score_test(model_rf_entropy, x_test, y_test)

    ### logistic regression (classification)

    model_lr = LogisticRegression(max_iter=500)
    model_lr.fit(x_train, y_train)
    logistic_regression_score = score_test(model_lr, x_test, y_test)

    ### knn (classification)
    max_knn_score = 0
    max_knn_val = 0
    for i in range(1, 10):
        model_knn = KNeighborsClassifier(n_neighbors=i)
        model_knn.fit(x_train, y_train)
        knn_score = score_test(model_knn, x_test, y_test)
        if knn_score > max_knn_score:
            max_knn_score = knn_score
            max_knn_val = i

    model_mlp = MLPClassifier(max_iter=199)
    model_mlp.fit(x_train, y_train)
    mlp_score = score_test(model_mlp, x_test, y_test)

    ### MLP (classification)

    if to_print:
        print(" ")
        print("###########")
        print("Results for feature: " + feature_name)
        print("Random Forest (gini) Score: " + str(random_forest_gini_score))
        print("Random Forest (entropy) Score: " + str(random_forest_entropy_score))
        print("Logistic Regression Score: " + str(logistic_regression_score))
        print("KNN with k = " + str(max_knn_val) + " Score: " + str(max_knn_score))
        print("Multi-layer Perceptron with Neural Networks score: " + str(mlp_score))
        print("###########")
        print(" ")

    return random_forest_gini_score, random_forest_entropy_score, logistic_regression_score, max_knn_score, mlp_score


def model_score(model_1, model_2, x_test, y_test):
    """
    :param model_1:
    :param model_2:
    :param x_test:
    :param y_test:
    :return:
    max distance = (1-(-1))^2 + (1-0)^2 = 4 + 1 = 5
    min distance = 0
    """
    # label_mapping_two_to_one = {"0": {"-1": "0", "0": "1", "1": "2"}, "1": {"-1": "3", "0": "4", "1": "5"}}
    label_mapping_one_to_two = {"0": [0, -1], "3": [1, -1], "1": [0, 0], "2": [0, 1], "4": [1, 0], "5": [1, 1]}
    err = 0
    max_distance = math.sqrt(5)
    min_distance = 0

    for i in range(len(x_test)):
        label_1 = int(model_1.predict(x_test.iloc[i].to_numpy().reshape(1, -1))[0])
        label_2 = int(model_2.predict(x_test.iloc[i].to_numpy().reshape(1, -1))[0])

        test_label = label_mapping_one_to_two[str(int(y_test.iloc[i]))]

        distance_i = math.sqrt((math.pow((label_1 - test_label[0]), 2) + math.pow((label_2 - test_label[1]), 2)))
        err += ((distance_i-min_distance)/(max_distance-min_distance))

    return 1 - (err/len(x_test))


def trainAndTest(train_one_dim, train_two_dim, test, to_print=False):
    one_dim_scores = test_algorithms_next_step_one_dimension(train_one_dim, test, True, to_print)
    two_dim_scores = test_algorithms_next_step_two_dimensions(train_two_dim, test, True, to_print)
    pitch_scores = test_algorithms_scores(train_one_dim, test, "Pitch", to_print)
    tempo_scores = test_algorithms_scores(train_one_dim, test, "Tempo", to_print)
    rhythm_scores = test_algorithms_scores(train_one_dim, test, "Rhythm", to_print)
    a_d_scores = test_algorithms_scores(train_one_dim, test, "Articulation & Dynamics", to_print)
    return one_dim_scores, two_dim_scores, pitch_scores, tempo_scores, rhythm_scores, a_d_scores


def score_test(model, x_test, y_test):
    cnt = 0
    for i in range(len(x_test)):
        y_hat = int(model.predict(x_test.iloc[i].to_numpy().reshape(1, -1))[0])
        if y_hat == y_test[i]:
            cnt += 1
    return cnt / len(x_test)
