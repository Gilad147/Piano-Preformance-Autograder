import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from auxiliary import test_algorithms
import Performance_class
import shutil
from pathlib import Path
import Automated_teacher
import auxiliary

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


def teacherGrades(teacher_grades_df: pd.Series):
    teacher_grades_array = teacher_grades_df.array
    teacher_grades = []
    i = 19
    for song in SurveyPerformanceList:

        performance_grades = {"name": song["name"], "player_name": song["player_name"],
                              "pitch": int(teacher_grades_array[i]), "tempo": int(teacher_grades_array[i + 1]),
                              "rhythm": int(teacher_grades_array[i + 2]), "a_d": int(teacher_grades_array[i + 3]),
                              "overall": int(teacher_grades_array[i + 4])}
        next_step = int(teacher_grades_array[i + 5])
        if next_step == 1:

            performance_grades["next_step"] = int(teacher_grades_array[i + 6]) - 1
        else:
            performance_grades["next_step"] = int(teacher_grades_array[i + 7]) + 2
        teacher_grades.append(performance_grades)
        i += 8
    return teacher_grades


def processSurveyResults(path, n):
    results_df = pd.read_csv(path, dtype={
        'string_col': 'float16',
        'int_col': 'float16'
    }).fillna(0)
    results = []
    for i in range(1, n + 1):
        teacher_df = results_df.iloc[i, :]
        results.append(teacherGrades(teacher_df))
    return results


def getFeatures(path, name, player_name):
    performance = None
    rhythm_feature, velocity_feature, duration_feature, pitch_feature, tempo_feature = -1, 0, 0, 0, 0
    original_path = path + "/" + "original songs/" + name + ".midi"
    try:
        performance_path = path + "/real data/" + name + "/" + player_name + ".mid"
        performance = Performance_class.Performance(
            path=performance_path, name=name, player_name=player_name, original_path=original_path)
        rhythm_feature, velocity_feature, duration_feature, pitch_feature, tempo_feature = performance.get_features()

    except:
        performance_path = path + "/real data/" + name + "/" + player_name + ".midi"
        performance = Performance_class.Performance(
            path=performance_path, name=name, player_name=player_name, original_path=original_path)
        rhythm_feature, velocity_feature, duration_feature, pitch_feature, tempo_feature = performance.get_features()


    finally:
        return performance, rhythm_feature, velocity_feature, duration_feature, pitch_feature, tempo_feature


def getDataForSL(csv_path, number_of_teachers, folder_path):
    all_performances_predications = []
    performances = []
    features_predications = []
    results = processSurveyResults(csv_path, number_of_teachers)
    for teacher in results:
        for song in teacher:
            performance, rhythm_feature, velocity_feature, duration_feature, pitch_feature, tempo_feature = \
                getFeatures(folder_path, song["name"], song["player_name"])
            if rhythm_feature != -1:
                all_performances_predications.append(
                    [rhythm_feature, velocity_feature, duration_feature, pitch_feature,
                     tempo_feature, song["next_step"]])
                features_predications.append(
                    [rhythm_feature, velocity_feature, duration_feature, pitch_feature,
                     tempo_feature, song["rhythm"], song["a_d"], song["pitch"], song["tempo"]])
                performances.append(performance)

    return all_performances_predications, features_predications, performances


def trainAndTest(csv_path, number_of_teachers, folder_path, train_ratio, random_teachers=True, features=False,
                 fake_teachers=0, fake_songs=0):
    all_performances_predications, features_predications, performances = getDataForSL(csv_path, number_of_teachers,
                                                                                      folder_path)
    labeled_data_train = []  # explain that in order to prevent data leakeage (group leakage), we split *teachers* randomly into train-test
    labeled_data_test = []
    numOfPerformances = len(all_performances_predications)
    numOfSongs = int(numOfPerformances / number_of_teachers)
    if random_teachers:
        train_number = round(number_of_teachers * train_ratio)
        train_tuple = tuple(random.sample(range(0, number_of_teachers), train_number))
        for i in range(number_of_teachers):
            if i in train_tuple:
                for j in range(numOfSongs):
                    labeled_data_train.append([all_performances_predications[i * numOfSongs + j][0],
                                               all_performances_predications[i * numOfSongs + j][1],
                                               all_performances_predications[i * numOfSongs + j][2],
                                               all_performances_predications[i * numOfSongs + j][3],
                                               all_performances_predications[i * numOfSongs + j][4],
                                               all_performances_predications[i * numOfSongs + j][5]])
            else:
                for j in range(numOfSongs):
                    labeled_data_test.append([all_performances_predications[i * numOfSongs + j][0],
                                              all_performances_predications[i * numOfSongs + j][1],
                                              all_performances_predications[i * numOfSongs + j][2],
                                              all_performances_predications[i * numOfSongs + j][3],
                                              all_performances_predications[i * numOfSongs + j][4],
                                              all_performances_predications[i * numOfSongs + j][5]])
    else:
        train_number = round(numOfPerformances * train_ratio)
        train_tuple = tuple(random.sample(range(0, numOfPerformances), train_number))
        for i in range(numOfPerformances):
            if i in train_tuple:
                labeled_data_train.append([all_performances_predications[i][0],
                                           all_performances_predications[i][1],
                                           all_performances_predications[i][2],
                                           all_performances_predications[i][3],
                                           all_performances_predications[i][4],
                                           all_performances_predications[i][5]])
            else:
                labeled_data_test.append([all_performances_predications[i][0],
                                          all_performances_predications[i][1],
                                          all_performances_predications[i][2],
                                          all_performances_predications[i][3],
                                          all_performances_predications[i][4],
                                          all_performances_predications[i][5]])

    train = pd.DataFrame(labeled_data_train, columns=['Rhythm', 'Dynamics', 'Articulation', 'Pitch', 'Tempo', 'label'])
    test = pd.DataFrame(labeled_data_test, columns=['Rhythm', 'Dynamics', 'Articulation', 'Pitch', 'Tempo', 'label'])

    if fake_teachers:
        generated_data = auxiliary.generate_random_mistakes_data('songs/original songs', fake_songs, False)
        generated_data = performances + generated_data
        fake_train = Automated_teacher.fake_teachers_algorithm(False, performances_data=generated_data,
                                                               number_of_teachers=fake_teachers,
                                                               train_ratio=1, is_testing=False)
        train = pd.concat(objs=[train, fake_train])
        train = train.astype("float32")
    # test_algorithms(train, test, False)
    return test_algorithms(train, test, True, to_print=False)


def data_by_size_graph(csv_path, max_number_of_teachers, folder_path, train_ratio, random_teachers=True):
    scores = []
    for i in range(2, max_number_of_teachers + 1):
        scores.append(trainAndTest(csv_path, i, folder_path, train_ratio, random_teachers))
    scores_df = pd.DataFrame(scores, columns=["Random Forest (gini)", "Random Forest (entropy)", "Logistic Regression",
                                              "KNN(max)", "Multi-layer Perceptron"],
                             index=[str(i) for i in range(2, max_number_of_teachers + 1)])

    plt.figure()
    scores_df.plot()
    plt.ylim(0, 1)
    plt.xlabel = "Number of Teachers"
    plt.ylabel = "Model score"
    plt.show()


def fake_data_graph(csv_path, number_of_teachers, folder_path, train_ratio, fake_songs_max, fake_teachers_max,
                    random_teachers=False):
    scores = []
    x = []
    scores.append(trainAndTest(csv_path, number_of_teachers, folder_path, train_ratio, random_teachers, fake_teachers=0,
                               fake_songs=0))
    x.append((0, 0))
    for i in range(1, fake_teachers_max):
        for j in range(1, fake_songs_max + 1):
            scores.append(
                trainAndTest(csv_path, number_of_teachers, folder_path, train_ratio, random_teachers, fake_teachers=i,
                             fake_songs=j - 1))
            x.append((i, j-1))
            print("number of teachers: " + str(i) + " number of songs:" + str(j - 1) + "\nDone")
    index = pd.MultiIndex.from_tuples(x, names=["Teachers", "Fake Songs"])
    scores_df = pd.DataFrame(scores, columns=["Random Forest (gini)", "Random Forest (entropy)", "Logistic Regression",
                                              "KNN(max)", "Multi-layer Perceptron"],
                             index=index)

    print(scores_df.to_numpy().max())
    print(scores_df.to_numpy().argmax())

    plt.figure()
    scores_df.plot()
    plt.ylim(0, 1)
    plt.xlabel = "Number of Fake Teachers and Fake Songs"
    plt.ylabel = "Model score"
    plt.show()


if __name__ == "__main__":
    fake_data_graph("Music+evaluation_August+30,+2021_03.17.csv", 7, "songs", train_ratio=0.7, random_teachers=True,
                    fake_teachers_max=5, fake_songs_max=3)
