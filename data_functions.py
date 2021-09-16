import pickle
import random
import testing_algorithms
from math import ceil
try:
    from math import comb
except:
    1
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics
from sklearn.metrics import ConfusionMatrixDisplay
import Performance_class
import Song_Class
import numpy as np
from sklearn.model_selection import KFold, RepeatedKFold
from statsmodels.stats.proportion import proportion_confint

from Automated_teacher import fake_teachers_algorithm

SurveyPerformanceList = [{"name": "HaKova Sheli", "player_name": "Student 12"},
                         {"name": "Bnu Gesher", "player_name": "Student 3"},
                         {"name": "Yom Huledet Sameach", "player_name": "Student 23"},
                         {"name": "HaAviv", "player_name": "Student 10"},
                         {"name": "Emek Hanahar Haadom", "player_name": "Student 6"},
                         {"name": "Lifnei Shanim Rabot", "player_name": "Student 18"},
                         {"name": "Achbar Hizaher", "player_name": "Student 1"},
                         {"name": "Yom Huledet Sameach", "player_name": "Student 22"},
                         {"name": "HaAviv", "player_name": "Student 11"},
                         {"name": "Shir Eres", "player_name": "Student 19"},
                         {"name": "Achbar Hizaher", "player_name": "Student 2"},
                         {"name": "Bnu Gesher", "player_name": "Student 4"},
                         {"name": "Emek Hanahar Haadom", "player_name": "Student 7"},
                         {"name": "Gina Li", "player_name": "Student 8"},
                         {"name": "Hatul Al Hagag", "player_name": "Student 15"},
                         {"name": "Shir Eres", "player_name": "Student 20"},
                         {"name": "Yom Huledet Sameach", "player_name": "Student 24"},
                         {"name": "HaKova Sheli", "player_name": "Student 14"},
                         {"name": "Hatul Al Hagag", "player_name": "Student 16"},
                         {"name": "HaAviv", "player_name": "Student 9"},
                         {"name": "Shir Eres", "player_name": "Student 21"},
                         {"name": "Emek Hanahar Haadom", "player_name": "Student 5"},
                         {"name": "Lifnei Shanim Rabot", "player_name": "Student 17"},
                         {"name": "Yom Huledet Sameach", "player_name": "Student 25"},
                         {"name": "HaKova Sheli", "player_name": "Student 13"}]
maxNumberOfTeachers = 10


def teacherGrades(teacher_grades_df: pd.Series):
    teacher_grades_array = teacher_grades_df.array
    teacher_grades = {"Pitch": 5 - int(teacher_grades_array[0]), "Tempo": 5 - int(teacher_grades_array[1]),
                      "Rhythm": 5 - int(teacher_grades_array[2]),
                      "A_D": 5 - int(teacher_grades_array[3]),
                      "Overall": int(teacher_grades_array[4])}
    next_step = int(teacher_grades_array[5])
    if next_step == 1:
        teacher_grades["Next Step"] = int(teacher_grades_array[6]) - 1
    else:
        teacher_grades["Next Step"] = int(teacher_grades_array[7]) + 2
    return teacher_grades


def get_performance_grades(performance_grades_df: pd.DataFrame):
    grades = []
    for i in range(0, len(performance_grades_df.index)):
        teacher_df = performance_grades_df.iloc[i, :]
        grades.append(teacherGrades(teacher_df))
    grades_df = pd.DataFrame(grades)

    return grades_df


def getPerformance(path, name, player_name):
    performance = None
    original_path = path + "/" + "original songs/" + name + ".midi"
    try:
        performance_path = path + "/real data/" + name + "/" + player_name + ".mid"
        performance = Performance_class.Performance(
            path=performance_path, name=name, player_name=player_name, original_path=original_path)

    except:
        performance_path = path + "/real data/" + name + "/" + player_name + ".midi"
        performance = Performance_class.Performance(
            path=performance_path, name=name, player_name=player_name, original_path=original_path)

    finally:
        return performance


def processSurveyResults(csv_path, folder_path):
    results_df = pd.read_csv(csv_path, dtype={
        'string_col': 'int32',
        'int_col': 'int32'
    }).fillna(-1)
    song_dict = {}
    i = 19
    for performance in SurveyPerformanceList:
        song_name = performance["name"]
        performance_class = getPerformance(folder_path, song_name, performance["player_name"])
        if performance_class is None:
            continue
        performance_grades_df = results_df.iloc[:, i:i + 8]
        grades_df = get_performance_grades(performance_grades_df)

        performance_class.teachers_grades += grades_df.values.tolist()

        pitch_feature, tempo_feature, rhythm_feature, articulation_feature, dynamics_feature = \
            performance_class.get_features()

        performance_class.give_labels()
        labels = performance_class.labels

        performance_attributes = [pitch_feature, tempo_feature, rhythm_feature, articulation_feature, dynamics_feature,
                                  labels[0], labels[1], labels[2], labels[3], labels[4], labels[5]]

        # performance_attributes_df = pd.Series(performance_attributes)

        if song_name not in song_dict:
            new_song = Song_Class.Song(performance["name"])
            song_dict[song_name] = new_song

        song_dict[song_name].performances.append(performance_attributes)
        i += 8
    return song_dict


def print_graph(scores, name, index, xlabel):
    scores_df = pd.DataFrame(scores, columns=["Pitch", "Tempo", "Rhythm",
                                              "Articulation & Dynamics", "Overall", 'Next Step'],
                             index=index)

    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=scores_df, colLabels=scores_df.columns, rowLabels=index, loc='center')

    fig.tight_layout()
    plt.show()


def plot_data_by_real_teachers(csv_path, folder_path):
    one_dim_final = []
    pitch_final = []
    tempo_final = []
    rhythm_final = []
    a_d_final = []
    overall_final = []

    for i in range(3):
        one_dim_i, pitch_i, tempo_i, rhythm_i, a_d_i, overall_i = train_test_real(csv_path, folder_path, True)

        one_dim_final.append(one_dim_i)
        pitch_final.append(pitch_i)
        tempo_final.append(tempo_i)
        rhythm_final.append(rhythm_i)
        a_d_final.append(a_d_i)
        overall_final.append(overall_i)

    one_dim_final = np.mean(one_dim_final, axis=0)
    pitch_final = np.mean(pitch_final, axis=0)
    tempo_final = np.mean(tempo_final, axis=0)
    rhythm_final = np.mean(rhythm_final, axis=0)
    a_d_final = np.mean(a_d_final, axis=0)
    overall_final = np.mean(overall_final, axis=0)

    print(" ")
    print("###########")
    print("One dimension results:")
    print("Random Forest (gini) Score: " + str(one_dim_final[0]))
    print("Random Forest (entropy) Score: " + str(one_dim_final[1]))
    print("Logistic Regression Score: " + str(one_dim_final[2]))
    print("KNN Score: " + str(one_dim_final[3]))
    print("Multi-layer Perceptron with Neural Networks score: " + str(one_dim_final[4]))
    print("XGB score: " + str(one_dim_final[6]))
    print("###########")
    print(" ")

    print(" ")
    print("###########")
    print("Pitch results:")
    print("Random Forest (gini) Score: " + str(pitch_final[0]))
    print("Random Forest (entropy) Score: " + str(pitch_final[1]))
    print("Logistic Regression Score: " + str(pitch_final[2]))
    print("KNN Score: " + str(pitch_final[3]))
    print("Multi-layer Perceptron with Neural Networks score: " + str(pitch_final[4]))
    print("XGB score: " + str(pitch_final[6]))
    print("###########")
    print(" ")

    print(" ")
    print("###########")
    print("Tempo results:")
    print("Random Forest (gini) Score: " + str(tempo_final[0]))
    print("Random Forest (entropy) Score: " + str(tempo_final[1]))
    print("Logistic Regression Score: " + str(tempo_final[2]))
    print("KNN Score: " + str(tempo_final[3]))
    print("Multi-layer Perceptron with Neural Networks score: " + str(tempo_final[4]))
    print("XGB score: " + str(tempo_final[6]))
    print("###########")
    print(" ")

    print(" ")
    print("###########")
    print("Rhythm results:")
    print("Random Forest (gini) Score: " + str(rhythm_final[0]))
    print("Random Forest (entropy) Score: " + str(rhythm_final[1]))
    print("Logistic Regression Score: " + str(rhythm_final[2]))
    print("KNN Score: " + str(rhythm_final[3]))
    print("Multi-layer Perceptron with Neural Networks score: " + str(rhythm_final[4]))
    print("XGB score: " + str(rhythm_final[6]))
    print("###########")
    print(" ")

    print(" ")
    print("###########")
    print("A&D results:")
    print("Random Forest (gini) Score: " + str(a_d_final[0]))
    print("Random Forest (entropy) Score: " + str(a_d_final[1]))
    print("Logistic Regression Score: " + str(a_d_final[2]))
    print("KNN Score: " + str(rhythm_final[3]))
    print("Multi-layer Perceptron with Neural Networks score: " + str(a_d_final[4]))
    print("XGB score: " + str(a_d_final[6]))
    print("###########")
    print(" ")

    print(" ")
    print("###########")
    print("Overall results:")
    print("Random Forest (gini) Score: " + str(overall_final[0]))
    print("Random Forest (entropy) Score: " + str(overall_final[1]))
    print("Logistic Regression Score: " + str(overall_final[2]))
    print("KNN Score: " + str(overall_final[3]))
    print("Multi-layer Perceptron with Neural Networks score: " + str(overall_final[4]))
    print("XGB score: " + str(overall_final[6]))
    print("###########")
    print(" ")


def train_test_real(csv_path, folder_path, to_print, del_songs=0):
    if csv_path == "Fake":
        song_dict = fake_teachers_algorithm(True, 10, folder=folder_path)
    else:
        song_dict = processSurveyResults(csv_path, folder_path)
    del song_dict["HaKova Sheli"]  # for test in the end
    del song_dict["Shir Eres"]  # for test in the end
    song_lst = list(song_dict.keys())
    for j in range(del_songs):
        del song_dict[song_lst[-1]]
        del song_lst[-1]
    n_splits = 4 - ceil(del_songs / 2)
    n_repeats = 50
    n_total = 28
    try:
        n_total = comb(8 - del_songs, 2)
    except:
        1
    # kf = KFold(n_splits=10)

    validation_dict = {}

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)

    one_dim_scores = [0, 0, 0, 0, 0, 0, 0]
    pitch_scores = [0, 0, 0, 0, 0, 0, 0]
    tempo_scores = [0, 0, 0, 0, 0, 0, 0]
    rhythm_scores = [0, 0, 0, 0, 0, 0, 0]
    a_d_scores = [0, 0, 0, 0, 0, 0, 0]
    overall_scores = [0, 0, 0, 0, 0, 0, 0]

    one_dim_k = []
    pitch_k = []
    tempo_k = []
    rhythm_k = []
    a_d_k = []
    overall_k = []

    label_mapping = {0: ["0", "-1"], 3: ["1", "-1"], 1: ["0", "0"], 2: ["0", "1"], 4: ["1", "0"],
                     5: ["1", "1"]}
    cnt = 0
    for train, test in rkf.split(song_lst):
        labeled_data_train_one_dimension = []
        labeled_data_test = []

        song_test_1 = song_dict[song_lst[test[0]]]
        song_test_2 = song_dict[song_lst[test[1]]]
        if song_test_1.name in validation_dict:
            if song_test_2.name in validation_dict[song_test_1.name]:
                continue
            else:
                validation_dict[song_test_1.name] += (song_test_2.name,)
                if song_test_2.name in validation_dict:
                    validation_dict[song_test_2.name] += (song_test_1.name,)
        else:
            validation_dict[song_test_1.name] = (song_test_2.name,)
            if song_test_2.name in validation_dict:
                validation_dict[song_test_2.name] += (song_test_1.name,)
            else:
                validation_dict[song_test_2.name] = (song_test_1.name,)

        cnt += 1

        labeled_data_test += song_test_1.performances
        labeled_data_test += song_test_2.performances

        for i in train:
            song_i = song_dict[song_lst[i]]
            labeled_data_train_one_dimension += song_i.performances


        train_one_dimension = pd.DataFrame(labeled_data_train_one_dimension,
                                           columns=['Pitch', 'Tempo', 'Rhythm', 'Articulation', 'Dynamics',
                                                    "Teacher's Pitch", "Teacher's Tempo", "Teacher's Rhythm",
                                                    "Teacher's Articulation & Dynamics", "Teacher's Overall", 'label'])

        test = pd.DataFrame(labeled_data_test, columns=['Pitch', 'Tempo', 'Rhythm', 'Articulation', 'Dynamics',
                                                        "Teacher's Pitch", "Teacher's Tempo", "Teacher's Rhythm",
                                                        "Teacher's Articulation & Dynamics", "Teacher's Overall",
                                                        'label'])

        one_dim_score_i, pitch_score_i, tempo_score_i, rhythm_score_i, a_d_score_i, overall_score_i = \
            testing_algorithms.trainAndTest(train_one_dimension, test, cnt, to_print=False, model_features="All")

        one_dim_scores = [x + y for x, y in zip(one_dim_scores, one_dim_score_i)]
        pitch_scores = [x + y for x, y in zip(pitch_scores, pitch_score_i)]
        tempo_scores = [x + y for x, y in zip(tempo_scores, tempo_score_i)]
        rhythm_scores = [x + y for x, y in zip(rhythm_scores, rhythm_score_i)]
        a_d_scores = [x + y for x, y in zip(a_d_scores, a_d_score_i)]
        overall_scores = [x + y for x, y in zip(overall_scores, overall_score_i)]

        one_dim_k.append(one_dim_score_i[5])
        pitch_k.append(pitch_score_i[5])
        tempo_k.append(tempo_score_i[5])
        rhythm_k.append(rhythm_score_i[5])
        a_d_k.append(a_d_score_i[5])
        overall_k.append(overall_score_i[5])

        print(str(cnt) + " is finished!")
        if cnt == n_total:
            break

    one_dim_final = [x / n_total for x in one_dim_scores]
    pitch_final = [x / n_total for x in pitch_scores]
    tempo_final = [x / n_total for x in tempo_scores]
    rhythm_final = [x / n_total for x in rhythm_scores]
    a_d_final = [x / n_total for x in a_d_scores]
    overall_final = [x / n_total for x in overall_scores]

    one_dim_k_final = max(set(one_dim_k), key=one_dim_k.count)
    rhythm_k_final = max(set(rhythm_k), key=rhythm_k.count)
    pitch_k_final = max(set(pitch_k), key=pitch_k.count)
    tempo_k_final = max(set(tempo_k), key=tempo_k.count)
    a_d_k_final = max(set(a_d_k), key=a_d_k.count)
    overall_k_final = max(set(overall_k), key=a_d_k.count)

    if to_print:
        print(" ")
        print("###########")
        print("One dimension results:")
        print("Random Forest (gini) Score: " + str(one_dim_final[0]))
        print("Random Forest (entropy) Score: " + str(one_dim_final[1]))
        print("Logistic Regression Score: " + str(one_dim_final[2]))
        print("KNN Score: " + str(one_dim_final[3]) + " with k = " + str(one_dim_k_final))
        print("Multi-layer Perceptron with Neural Networks score: " + str(one_dim_final[4]))
        print("XGB score: " + str(one_dim_final[6]))
        print("###########")
        print(" ")

        print(" ")
        print("###########")
        print("Pitch results:")
        print("Random Forest (gini) Score: " + str(pitch_final[0]))
        print("Random Forest (entropy) Score: " + str(pitch_final[1]))
        print("Logistic Regression Score: " + str(pitch_final[2]))
        print("KNN Score: " + str(pitch_final[3]) + " with k = " + str(pitch_k_final))
        print("Multi-layer Perceptron with Neural Networks score: " + str(pitch_final[4]))
        print("XGB score: " + str(pitch_final[6]))
        print("###########")
        print(" ")

        print(" ")
        print("###########")
        print("Tempo results:")
        print("Random Forest (gini) Score: " + str(tempo_final[0]))
        print("Random Forest (entropy) Score: " + str(tempo_final[1]))
        print("Logistic Regression Score: " + str(tempo_final[2]))
        print("KNN Score: " + str(tempo_final[3]) + " with k = " + str(tempo_k_final))
        print("Multi-layer Perceptron with Neural Networks score: " + str(tempo_final[4]))
        print("XGB score: " + str(tempo_final[6]))
        print("###########")
        print(" ")

        print(" ")
        print("###########")
        print("Rhythm results:")
        print("Random Forest (gini) Score: " + str(rhythm_final[0]))
        print("Random Forest (entropy) Score: " + str(rhythm_final[1]))
        print("Logistic Regression Score: " + str(rhythm_final[2]))
        print("KNN Score: " + str(rhythm_final[3]) + " with k = " + str(rhythm_k_final))
        print("Multi-layer Perceptron with Neural Networks score: " + str(rhythm_final[4]))
        print("XGB score: " + str(rhythm_final[6]))
        print("###########")
        print(" ")

        print(" ")
        print("###########")
        print("A&D results:")
        print("Random Forest (gini) Score: " + str(a_d_final[0]))
        print("Random Forest (entropy) Score: " + str(a_d_final[1]))
        print("Logistic Regression Score: " + str(a_d_final[2]))
        print("KNN Score: " + str(rhythm_final[3]) + " with k = " + str(a_d_k_final))
        print("Multi-layer Perceptron with Neural Networks score: " + str(a_d_final[4]))
        print("XGB score: " + str(a_d_final[6]))
        print("###########")
        print(" ")

        print(" ")
        print("###########")
        print("Overall results:")
        print("Random Forest (gini) Score: " + str(overall_final[0]))
        print("Random Forest (entropy) Score: " + str(overall_final[1]))
        print("Logistic Regression Score: " + str(overall_final[2]))
        print("KNN Score: " + str(overall_final[3]) + " with k = " + str(overall_k_final))
        print("Multi-layer Perceptron with Neural Networks score: " + str(overall_final[4]))
        print("XGB score: " + str(overall_final[6]))
        print("###########")
        print(" ")

    return one_dim_final, pitch_final, tempo_final, rhythm_final, a_d_final, overall_final


def songs_to_csv(song_dict):
    for song in song_dict.values():
        song_pd = pd.DataFrame(song.performances,
                               columns=['Pitch', 'Tempo', 'Rhythm', 'Articulation', 'Dynamics',
                                        "Teacher's Pitch", "Teacher's Tempo", "Teacher's Rhythm",
                                        "Teacher's Articulation & Dynamics", "Teacher's Overall", 'label'])
        song_pd.to_csv(song.name + '.csv')


def final_tests(csv_path, folder_path, del_songs=0):
    n_total = 28
    try:
        n_total = comb(8 - del_songs, 2)
    except:
        1
    if csv_path == "Fake":
        song_dict = fake_teachers_algorithm(True, 10, folder=folder_path)
    else:
        song_dict = processSurveyResults(csv_path, folder_path)
    song_1 = song_dict["HaKova Sheli"]  # for test in the end
    song_2 = song_dict["Shir Eres"]  # for test in the end

    labeled_data_performances = song_1.performances
    labeled_data_performances += song_2.performances

    labeled_data = pd.DataFrame(labeled_data_performances,
                                columns=['Pitch', 'Tempo', 'Rhythm', 'Articulation', 'Dynamics',
                                         "Teacher's Pitch", "Teacher's Tempo", "Teacher's Rhythm",
                                         "Teacher's Articulation & Dynamics", "Teacher's Overall", 'label'])

    ### one_dim
    x_one_dim = pd.DataFrame(labeled_data[["Pitch", "Tempo", 'Rhythm', 'Articulation', 'Dynamics']])
    y_one_dim = labeled_data["label"]

    models = load_models("label_one_dim", n_total)
    one_dim_final_score = predict_all(models, x_one_dim, y_one_dim)
    one_dim_confidence_interval = proportion_confint(count=(one_dim_final_score * 6), nobs=6, alpha=0.1)
    print_confusion_matrix(models, x_one_dim, y_one_dim, labels=['0', '1', '2', '3', '4', '5'])

    ### Pitch
    x_pitch = pd.DataFrame(labeled_data["Pitch"])
    y_pitch = labeled_data["Teacher's Pitch"]

    models = load_models("Pitch", n_total)
    pitch_final_score = predict_all(models, x_pitch, y_pitch)
    pitch_confidence_interval = proportion_confint(count=(pitch_final_score * 6), nobs=6, alpha=0.1)
    print_confusion_matrix(models, x_pitch, y_pitch)

    ### Tempo
    x_tempo = pd.DataFrame(labeled_data[["Pitch", "Tempo"]])
    y_tempo = labeled_data["Teacher's Tempo"]

    models = load_models("Tempo", n_total)
    tempo_final_score = predict_all(models, x_tempo, y_tempo)
    tempo_confidence_interval = proportion_confint(count=(tempo_final_score * 6), nobs=6, alpha=0.1)
    print_confusion_matrix(models, x_tempo, y_tempo)

    ### Rhythm
    x_rhythm = pd.DataFrame(labeled_data[["Pitch", 'Rhythm']])
    y_rhythm = labeled_data["Teacher's Rhythm"]

    models = load_models("Rhythm", n_total)
    rhythm_final_score = predict_all(models, x_rhythm, y_rhythm, False)
    rhythm_confidence_interval = proportion_confint(count=(rhythm_final_score * 6), nobs=6, alpha=0.1)
    print_confusion_matrix(models, x_rhythm, y_rhythm)

    ### A&D
    x_a_d = pd.DataFrame(labeled_data[["Pitch", "Tempo", 'Rhythm', 'Articulation', 'Dynamics']])
    y_a_d = labeled_data["Teacher's Articulation & Dynamics"]

    models = load_models("Articulation & Dynamics", n_total)
    a_d_final_score = predict_all(models, x_a_d, y_a_d, False)
    a_d_confidence_interval = proportion_confint(count=((1 - a_d_final_score) * 6), nobs=6, alpha=0.1)
    print_confusion_matrix(models, x_a_d, y_a_d)

    ### Overall
    x_overall = pd.DataFrame(labeled_data[['Pitch', 'Tempo', 'Rhythm', 'Articulation', 'Dynamics']])
    y_overall = labeled_data["Teacher's Overall"]

    models = load_models("Overall", n_total)
    overall_final_score = predict_all(models, x_overall, y_overall)
    overall_confidence_interval = proportion_confint(count=((1 - overall_final_score) * 6), nobs=6, alpha=0.1)
    print_confusion_matrix(models, x_overall, y_overall, labels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])

    print(" ")
    print("###########")
    print("Final Results:")
    print("One Dim label score: " + str(one_dim_final_score) + ", error confidence interval: " + str(
        one_dim_confidence_interval))
    print("Pitch Score: " + str(pitch_final_score) + ", error confidence interval: " + str(pitch_confidence_interval))
    print("Tempo Score: " + str(tempo_final_score) + ", error confidence interval: " + str(tempo_confidence_interval))
    print(
        "Rhythm Score: " + str(rhythm_final_score) + ", error confidence interval: " + str(rhythm_confidence_interval))
    print("A&D score: " + str(a_d_final_score) + ", error confidence interval: " + str(a_d_confidence_interval))
    print("Overall score: " + str(overall_final_score) + ", error confidence interval: " + str(
        overall_confidence_interval))
    print("###########")
    print(" ")

    return pitch_final_score, tempo_final_score, rhythm_final_score, a_d_final_score, overall_final_score, one_dim_final_score


def get_correlation_matrix(csv_path, folder_path):
    song_dict = processSurveyResults(csv_path, folder_path)
    song_lst = list(song_dict.keys())
    labeled_data = []
    for song in song_lst:
        song_i = song_dict[song]
        labeled_data += song_i.performances
    performances_df = pd.DataFrame(labeled_data,
                                   columns=['Pitch', 'Tempo', 'Rhythm', 'Articulation', 'Dynamics',
                                            "Teacher's Pitch", "Teacher's Tempo", "Teacher's Rhythm",
                                            "Teacher's Articulation & Dynamics", "Teacher's Overall", 'label'])
    data_corr = performances_df.corr(method="spearman")
    data_corr = data_corr[["Teacher's Pitch", "Teacher's Tempo", "Teacher's Rhythm",
                           "Teacher's Articulation & Dynamics", "Teacher's Overall", 'label']][0:5]
    data_corr.to_csv("correlation_matrix.csv")


def load_models(name, num_of_models=29):
    models = []
    for i in range(1, num_of_models):
        loaded_model = pickle.load(open('models/' + name + '/model_' + str(i) + '.sav', 'rb'))
        models.append(loaded_model)
    return models


def predict_all(models, x, y, majority=True):
    cnt = 0
    for i in range(len(y)):
        y_hat_final = predict_from_models(models, pd.DataFrame(x.iloc[i]), majority)
        if y_hat_final == y[i]:
            cnt += 1

    return cnt / len(y)


def print_confusion_matrix(models, x, y, labels=None):
    if labels is None:
        labels = ['0', '1', '2', '3', '4']
    y_pred = []
    y_true = y.to_numpy(dtype=str)
    for i in range(len(y)):
        y_pred.append(str(predict_from_models(models, pd.DataFrame(x.iloc[i]))))
    co_mat = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(co_mat)
    disp.plot()
    plt.show()


def predict_from_models(models, x, majority=True):
    y_hat = []
    for model in models:
        try:
            y_hat.append(int(model.predict(x.to_numpy().reshape(1, -1))[0]))
        except:
            x_i = x.transpose()
            y_hat.append(int(model.predict(x_i)))
    if majority:
        y_hat_final = max(set(y_hat), key=y_hat.count)
    else:
        y_hat_final = round((sum(list(map(int, y_hat))) / len(y_hat)))
    return y_hat_final


if __name__ == "__main__":
    # plot_data_by_real_teachers("Music+evaluation_September+7%2C+2021_07.06.csv", "songs")
    scores = []

    for del_songs in range(4, -1, -1):
        train_test_real("Music+evaluation_September+7%2C+2021_07.06.csv", "songs", to_print=True, del_songs=del_songs)
        pitch_final_score, tempo_final_score, rhythm_final_score, a_d_final_score, overall_final_score, one_dim_final_score = \
            final_tests("Music+evaluation_September+7%2C+2021_07.06.csv", "songs", del_songs=del_songs)
        scores.append([pitch_final_score, tempo_final_score, rhythm_final_score, a_d_final_score, overall_final_score,
                       one_dim_final_score])
    print(scores)



    '''
    generated_data = auxiliary.generate_random_mistakes_data('songs/original songs', 20, True)
    train_test_real("Fake", "songs/", to_print=True)
    pitch_final_score, tempo_final_score, rhythm_final_score, a_d_final_score, overall_final_score, one_dim_final_score = \
        final_tests("Fake", "songs/")
    '''
    # get_correlation_matrix("Music+evaluation_September+7%2C+2021_07.06.csv", "songs")
