import pickle
import random
import matplotlib.pyplot as plt
import pandas as pd
import Performance_class
import Automated_teacher
import Song_Class
import auxiliary
import numpy as np
from sklearn.model_selection import KFold, RepeatedKFold
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint

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


def processSurveyResults(csv_path, folder_path, teachers):
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

        pitch_feature, tempo_feature, rhythm_feature, articulation_feature, dynamics_feature = performance_class.get_features()

        # if len(teachers) > 0:
        #     Automated_teacher.fake_teachers_feedback(performance_class, teachers, pitch_feature, tempo_feature, rhythm_feature, articulation_feature, dynamics_feature)

        performance_class.give_labels(majority_or_avg=True)
        labels = performance_class.labels

        performance_attributes = [pitch_feature, tempo_feature, rhythm_feature, articulation_feature, dynamics_feature,
                                  labels[0], labels[1], labels[2], labels[3], labels[4], labels[5]]

        #performance_attributes_df = pd.Series(performance_attributes)

        if song_name not in song_dict:
            new_song = Song_Class.Song(performance["name"])
            song_dict[song_name] = new_song

        song_dict[song_name].performances.append(performance_attributes)
        i += 8
    return song_dict


def getDataForSL(csv_path, folder_path, train_ratio, fake_teachers=[], train_tuple=None):
    song_dict = processSurveyResults(csv_path, folder_path, fake_teachers)
    #songs_to_csv(song_dict)
    song_set = set(song_dict.values())
    song_lst = list(song_set)
    song_lst.sort(key=lambda x: x.name)

    train_number = round(10 * train_ratio)
    if train_tuple is None:
        train_tuple = tuple(random.sample(range(0, 10), train_number))
    print(train_tuple)
    label_mapping = {0: ["0", "-1"], 3: ["1", "-1"], 1: ["0", "0"], 2: ["0", "1"], 4: ["1", "0"],
                     5: ["1", "1"]}

    labeled_data_train_one_dimension = []  # explain that in order to prevent data leakage (group leakage), we split *performances* randomly into train-test
    labeled_data_train_two_dimensions = []
    labeled_data_test = []

    if train_ratio == 0:
        song_lst = list(song_set)
        song_lst.sort(key=lambda x: x.name)
        for song in song_lst:
            labeled_data_test += song.performances

    else:
        for i in range(len(song_lst)):
            print(song_lst[i].name)
            if i in train_tuple:
                labeled_data_train_one_dimension += song_lst[i].performances

                for performance in song_lst[i].performances:
                    labeled_data_train_two_dimensions.append(performance.copy())
                    dim1, dim2 = label_mapping[labeled_data_train_two_dimensions[-1][9]][0], \
                                 label_mapping[labeled_data_train_two_dimensions[-1][9]][1]

                    labeled_data_train_two_dimensions[-1][9] = dim1
                    labeled_data_train_two_dimensions[-1].append(dim2)

            else:
                labeled_data_test += song_lst[i].performances

    train_one_dimension = pd.DataFrame(labeled_data_train_one_dimension,
                                       columns=['Pitch', 'Tempo', 'Rhythm', 'Articulation', 'Dynamics',
                                                "Teacher's Pitch", "Teacher's Tempo", "Teacher's Rhythm",
                                                "Teacher's Articulation & Dynamics", "Teacher's Overall", 'label'])
    train_two_dimensions = pd.DataFrame(labeled_data_train_two_dimensions,
                                        columns=['Pitch', 'Tempo', 'Rhythm', 'Articulation', 'Dynamics',
                                                 "Teacher's Pitch", "Teacher's Tempo", "Teacher's Rhythm",
                                                 "Teacher's Articulation & Dynamics", "Teacher's Overall", 'label dim 1', 'label dim 2'])
    test = pd.DataFrame(labeled_data_test, columns=['Pitch', 'Tempo', 'Rhythm', 'Articulation', 'Dynamics',
                                                    "Teacher's Pitch", "Teacher's Tempo", "Teacher's Rhythm",
                                                    "Teacher's Articulation & Dynamics", "Teacher's Overall", 'label'])

    return train_one_dimension, train_two_dimensions, test


def print_graph(scores, name, index, xlabel="Number of Fake Teachers and Fake Songs"):
    scores_df = pd.DataFrame(scores, columns=["Random Forest (gini)", "Random Forest (entropy)", "Logistic Regression",
                                              "KNN(max)", "Multi-layer Perceptron"],
                             index=index)

    plt.figure()
    scores_df.plot(xlabel=xlabel, ylabel="Model score", title=name + " predication scores",
                   ylim=(0, 1), figsize=(12, 12), grid=True)
    plt.show()


def plot_data_by_real_teachers(csv_path, folder_path, number_of_fake_teachers):
    final_one_dim = []
    final_two_dim = []
    final_pitch = []
    final_tempo = []
    final_rhythm = []
    final_a_d = []

    rf_gini = []
    rf_entropy = []
    lr = []
    knn = []
    knn_k = []
    perceptron = []
    xgb = []

    for i in range(3):
        rf_gini_i, rf_entropy_i, lr_i, knn_i, knn_k_i, perceptron_i, xgb_i = train_test_real(csv_path, folder_path, True)
        rf_gini.append(rf_gini_i)
        rf_entropy.append(rf_entropy_i)
        lr.append(lr_i)
        knn.append(knn_i)
        knn_k.append(knn_k_i)
        perceptron.append(perceptron_i)
        xgb.append(xgb_i)

    rf_gini_avg = sum(rf_gini) / 3
    rf_entropy_avg = sum(rf_entropy) / 3
    lr_avg = sum(lr) / 3
    knn_avg = sum(knn) / 3
    perceptron_avg = sum(perceptron) / 3
    xgb_avg = sum(xgb) / 3
    k_final = max(set(knn_k), key=knn_k.count)

    print(" ")
    print("###########")
    print("Overall results:")
    print("Random Forest (gini) Score: " + str(rf_gini_avg))
    print("Random Forest (entropy) Score: " + str(rf_entropy_avg))
    print("Logistic Regression Score: " + str(lr_avg))
    print("KNN Score: " + str(knn_avg) + " with k = " + str(k_final))
    print("Multi-layer Perceptron with Neural Networks score: " + str(perceptron_avg))
    print("XGB score: " + str(xgb_avg))
    print("###########")
    print(" ")

    # sum_one_dim = [0, 0, 0, 0, 0]
    # train_tuples = [(2,3,4,5,6,7,8,9), (0,1,4,5,6,7,8,9), (0,1,2,3,6,7,8,9), (0,1,2,3,4,5,8,9), (0,1,2,3,4,5,6,7)]
    # for tuple_i in train_tuples:
    #     train_one_dimension, train_two_dimensions, test = getDataForSL(csv_path, folder_path, train_ratio=0.9, train_tuple=tuple_i)
    #     one_dim_score_i, two_dim_score_i, pitch_score_i, tempo_score_i, rhythm_score_i, a_d_score_i = \
    #          auxiliary.trainAndTest(train_one_dimension, train_two_dimensions, test, True)
    #     sum_one_dim[0] += one_dim_score_i[0]
    #     sum_one_dim[1] += one_dim_score_i[1]
    #     sum_one_dim[2] += one_dim_score_i[2]
    #     sum_one_dim[3] += one_dim_score_i[3]
    #     sum_one_dim[4] += one_dim_score_i[4]
    #
    # sum_one_dim[0] /= 5
    # sum_one_dim[1] /= 5
    # sum_one_dim[2] /= 5
    # sum_one_dim[3] /= 5
    # sum_one_dim[4] /= 5
    #
    # print(" ")
    # print("###########")
    # print("One dimension results:")
    # print("Random Forest (gini) Score: " + str(sum_one_dim[0]))
    # print("Random Forest (entropy) Score: " + str(sum_one_dim[1]))
    # print("Logistic Regression Score: " + str(sum_one_dim[2]))
    # print("KNN Score: " + str(sum_one_dim[3]))
    # print("Multi-layer Perceptron with Neural Networks score: " + str(sum_one_dim[4]))
    # print("###########")
    # print(" ")



    # one_dim_scores, two_dim_scores, pitch_scores, tempo_scores, rhythm_scores, a_d_scores = \
    #     auxiliary.trainAndTest(train_one_dim_only_fake, train_two_dim_only_fake, test_train_is_only_fake, True)

    # train_one_dim_only_fake, train_two_dim_only_fake, test_train_is_only_fake = train_is_only_fake('songs/additional songs',
    #                                                                                       number_of_performances=3,
    #                                                                                       create_midi_files_for_fake_performances=False,
    #                                                                                       number_of_teachers=10,
    #                                                                                       majority_or_avg=True,
    #                                                                                       print=False, csv_path=csv_path,
    #                                                                                       folder_path=folder_path)
    #
    # one_dim_scores, two_dim_scores, pitch_scores, tempo_scores, rhythm_scores, a_d_scores = \
    #     auxiliary.trainAndTest(train_one_dim_only_fake, train_two_dim_only_fake, test_train_is_only_fake, True)


    # fake_teachers = Automated_teacher.create_fake_teachers(number_of_fake_teachers)
    # train_one_dim_mixed, train_two_dim_mixed, test_mixed = train_is_mixed(csv_path, folder_path, train_ratio=0.7,
    #                                                                       teachers=fake_teachers)
    #
    # one_dim_scores, two_dim_scores, pitch_scores, tempo_scores, rhythm_scores, a_d_scores = \
    #     auxiliary.trainAndTest(train_one_dim_mixed, train_two_dim_mixed, test_mixed, True)

    # x = [i / 10 for i in range(5, 10)]
    # for j in range(5):
    #     one_dim = []
    #     two_dim = []
    #     pitch = []
    #     tempo = []
    #     rhythm = []
    #     a_d = []
    #     for i in range(5, 7):
    #         train_one_dim, train_two_dim, test = getDataForSL(csv_path,
    #                                                           folder_path,
    #                                                           train_ratio=i / 10, fake_teachers = fake_teachers)
    #         one_dim_scores, two_dim_scores, pitch_scores, tempo_scores, rhythm_scores, a_d_scores = \
    #             auxiliary.trainAndTest(train_one_dim, train_two_dim, test, True)
    #         one_dim.append(one_dim_scores)
    #         two_dim.append(two_dim_scores)
    #         pitch.append(pitch_scores)
    #         tempo.append(tempo_scores)
    #         rhythm.append(rhythm_scores)
    #         a_d.append(a_d_scores)
    #         print("scoring done for train ratio: " + str(x[i - 5]))
    #     final_one_dim.append(one_dim)
    #     final_two_dim.append(two_dim)
    #     final_pitch.append(pitch)
    #     final_tempo.append(tempo)
    #     final_rhythm.append(rhythm)
    #     final_a_d.append(a_d)
    # final_one_dim = np.array(final_one_dim).mean(axis=0)
    # final_two_dim = np.array(final_two_dim).mean(axis=0)
    # final_pitch = np.array(final_pitch).mean(axis=0)
    # final_tempo = np.array(final_tempo).mean(axis=0)
    # final_rhythm = np.array(final_rhythm).mean(axis=0)
    # final_a_d = np.array(final_a_d).mean(axis=0)
    # xlabel = "Train Ratio"
    # print_graph(final_one_dim, "One dimension Next step", x, xlabel=xlabel)
    # print_graph(final_two_dim, "Two dimensions Next step", x, xlabel=xlabel)
    # print_graph(final_pitch, "Pitch feature", x, xlabel=xlabel)
    # print_graph(final_tempo, "Tempo feature", x, xlabel=xlabel)
    # print_graph(final_rhythm, "Rhythm feature", x, xlabel=xlabel)
    # print_graph(final_a_d, "Articulation and Dynamics feature", x, xlabel=xlabel)


def train_test_real(csv_path, folder_path, to_print):
    song_dict = processSurveyResults(csv_path, folder_path, [])
    del song_dict["HaKova Sheli"] # for test in the end
    del song_dict["Shir Eres"] # for test in the end
    song_lst = list(song_dict.keys())

    n_splits = 4
    n_repeats = 50
    n_total = 28
    #kf = KFold(n_splits=10)

    validation_dict = {}

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)

    one_dim_scores = [0, 0, 0, 0, 0, 0, 0]
    two_dim_scores = [0, 0, 0, 0, 0, 0, 0]
    pitch_scores = [0, 0, 0, 0, 0, 0, 0]
    tempo_scores = [0, 0, 0, 0, 0, 0, 0]
    rhythm_scores = [0, 0, 0, 0, 0, 0, 0]
    a_d_scores = [0, 0, 0, 0, 0, 0, 0]
    overall_scores = [0, 0, 0, 0, 0, 0, 0]

    one_dim_k = []
    two_dim_k = []
    pitch_k = []
    tempo_k = []
    rhythm_k = []
    a_d_k = []
    overall_k = []

    label_mapping = {0: ["0", "-1"], 3: ["1", "-1"], 1: ["0", "0"], 2: ["0", "1"], 4: ["1", "0"],
                     5: ["1", "1"]}
    cnt = 0
    for train, test in rkf.split(song_lst):
        labeled_data_train_one_dimension = []  # explain that in order to prevent data leakage (group leakage), we split *songs!!* randomly into train-test
        labeled_data_train_two_dimensions = []
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

            for performance in song_i.performances:
                labeled_data_train_two_dimensions.append(performance.copy())
                dim1, dim2 = label_mapping[labeled_data_train_two_dimensions[-1][10]][0], \
                             label_mapping[labeled_data_train_two_dimensions[-1][10]][1]

                labeled_data_train_two_dimensions[-1][10] = dim1
                labeled_data_train_two_dimensions[-1].append(dim2)

        train_one_dimension = pd.DataFrame(labeled_data_train_one_dimension,
                                           columns=['Pitch', 'Tempo', 'Rhythm', 'Articulation', 'Dynamics',
                                                    "Teacher's Pitch", "Teacher's Tempo", "Teacher's Rhythm",
                                                    "Teacher's Articulation & Dynamics", "Teacher's Overall", 'label'])
        train_two_dimensions = pd.DataFrame(labeled_data_train_two_dimensions,
                                            columns=['Pitch', 'Tempo', 'Rhythm', 'Articulation', 'Dynamics',
                                                     "Teacher's Pitch", "Teacher's Tempo", "Teacher's Rhythm",
                                                     "Teacher's Articulation & Dynamics", "Teacher's Overall", 'label dim 1', 'label dim 2'])
        test = pd.DataFrame(labeled_data_test, columns=['Pitch', 'Tempo', 'Rhythm', 'Articulation', 'Dynamics',
                                                        "Teacher's Pitch", "Teacher's Tempo", "Teacher's Rhythm",
                                                        "Teacher's Articulation & Dynamics", "Teacher's Overall", 'label'])

        one_dim_score_i, two_dim_score_i, pitch_score_i, tempo_score_i, rhythm_score_i, a_d_score_i, overall_score_i = \
            auxiliary.trainAndTest(train_one_dimension, train_two_dimensions, test, cnt, to_print=False)

        one_dim_scores = [x + y for x, y in zip(one_dim_scores, one_dim_score_i)]
        two_dim_scores = [x + y for x, y in zip(two_dim_scores, two_dim_score_i)]
        pitch_scores = [x + y for x, y in zip(pitch_scores, pitch_score_i)]
        tempo_scores = [x + y for x, y in zip(tempo_scores, tempo_score_i)]
        rhythm_scores = [x + y for x, y in zip(rhythm_scores, rhythm_score_i)]
        a_d_scores = [x + y for x, y in zip(a_d_scores, a_d_score_i)]
        overall_scores = [x + y for x, y in zip(overall_scores, overall_score_i)]

        one_dim_k.append(one_dim_score_i[5])
        two_dim_k.append(two_dim_score_i[5])
        pitch_k.append(pitch_score_i[5])
        tempo_k.append(tempo_score_i[5])
        rhythm_k.append(rhythm_score_i[5])
        a_d_k.append(a_d_score_i[5])
        overall_k.append(overall_score_i[5])

        print(str(cnt) + " is finished!")
        if cnt == n_total:
            break

    one_dim_final = [x / n_total for x in one_dim_scores]
    two_dim_final = [x / n_total for x in two_dim_scores]
    pitch_final = [x / n_total for x in pitch_scores]
    tempo_final = [x / n_total for x in tempo_scores]
    rhythm_final = [x / n_total for x in rhythm_scores]
    a_d_final = [x / n_total for x in a_d_scores]
    overall_final = [x / n_total for x in overall_scores]

    one_dim_k_final = max(set(one_dim_k), key=one_dim_k.count)
    two_dim_k_final = max(set(two_dim_k), key=two_dim_k.count)
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
        print("Two dimensions results:")
        print("Random Forest (gini) Score: " + str(two_dim_final[0]))
        print("Random Forest (entropy) Score: " + str(two_dim_final[1]))
        print("Logistic Regression Score: " + str(two_dim_final[2]))
        print("KNN Score: " + str(two_dim_final[3]) + " with k = " + str(two_dim_k_final))
        print("Multi-layer Perceptron with Neural Networks score: " + str(two_dim_final[4]))
        print("XGB score: " + str(two_dim_final[6]))
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

    return one_dim_final[0], one_dim_final[1], one_dim_final[2], one_dim_final[3], one_dim_k_final, one_dim_final[4], one_dim_final[6]


def train_is_only_fake(songs_path, number_of_performances, create_midi_files_for_fake_performances, number_of_teachers, majority_or_avg, print, csv_path, folder_path):
    generated_data = auxiliary.generate_random_mistakes_data(songs_path, number_of_performances, create_midi_files_for_fake_performances)

    if create_midi_files_for_fake_performances:
        print("not ready yet")
    else:
        train_one_dim_fake, train_two_dim_fake, test = Automated_teacher.fake_teachers_algorithm(False,
                                                                                                 performances_data=generated_data,
                                                                                                 number_of_teachers=number_of_teachers,
                                                                                                 train_ratio=1,
                                                                                                 majority_or_avg=majority_or_avg,
                                                                                                 is_testing=print)

        train_one_dim_real, train_two_dim_real, test_final = getDataForSL(csv_path, folder_path, train_ratio=0)

    return train_one_dim_fake, train_two_dim_fake, test_final


def train_is_mixed(csv_path, folder_path, train_ratio, teachers):
    train_one_dim, train_two_dim, test = getDataForSL(csv_path, folder_path, train_ratio=train_ratio,
                                                      fake_teachers=teachers)

    return train_one_dim, train_two_dim, test


def choose_model(model, filename='finalized_next_step_model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def songs_to_csv(song_dict):
    for song in song_dict.values():
        song_pd = pd.DataFrame(song.performances,
                     columns=['Pitch', 'Tempo', 'Rhythm', 'Articulation', 'Dynamics',
                              "Teacher's Pitch", "Teacher's Tempo", "Teacher's Rhythm",
                              "Teacher's Articulation & Dynamics", "Teacher's Overall", 'label'])
        song_pd.to_csv(song.name + '.csv')


def final_tests(csv_path, folder_path):
    song_dict = processSurveyResults(csv_path, folder_path, [])
    song_1 = song_dict["HaKova Sheli"] # for test in the end
    song_2 = song_dict["Shir Eres"] # for test in the end

    labeled_data_performances = song_1.performances
    labeled_data_performances += song_2.performances

    labeled_data = pd.DataFrame(labeled_data_performances,
                                       columns=['Pitch', 'Tempo', 'Rhythm', 'Articulation', 'Dynamics',
                                                "Teacher's Pitch", "Teacher's Tempo", "Teacher's Rhythm",
                                                "Teacher's Articulation & Dynamics", "Teacher's Overall", 'label'])

    ### one_dim
    x_one_dim = pd.DataFrame(labeled_data[["Pitch", "Tempo", 'Rhythm', 'Articulation', 'Dynamics']])
    y_one_dim = labeled_data["label"]

    models = load_models("label_one_dim")
    one_dim_final_score = predict_all(models, x_one_dim, y_one_dim)
    one_dim_confidence_interval = proportion_confint(count=((1-one_dim_final_score)*6), nobs=6, alpha=0.1)

    ### Pitch
    x_pitch = pd.DataFrame(labeled_data["Pitch"])
    y_pitch = labeled_data["Teacher's Pitch"]

    models = load_models("Pitch")
    pitch_final_score = predict_all(models, x_pitch, y_pitch)
    pitch_confidence_interval = proportion_confint(count=((1 - pitch_final_score) * 6), nobs=6, alpha=0.1)

    ### Tempo
    x_tempo = pd.DataFrame(labeled_data[["Pitch", "Tempo"]])
    y_tempo = labeled_data["Teacher's Tempo"]

    models = load_models("Tempo")
    tempo_final_score = predict_all(models, x_tempo, y_tempo)
    tempo_confidence_interval = proportion_confint(count=((1 - tempo_final_score) * 6), nobs=6, alpha=0.1)

    ### Rhythm
    x_rhythm = pd.DataFrame(labeled_data["Rhythm"])
    y_rhythm = labeled_data["Teacher's Rhythm"]

    models = load_models("Rhythm")
    rhythm_final_score = predict_all(models, x_rhythm, y_rhythm)
    rhythm_confidence_interval = proportion_confint(count=((1 - rhythm_final_score) * 6), nobs=6, alpha=0.1)

    ### A&D
    x_a_d = pd.DataFrame(labeled_data[["Pitch", 'Articulation', 'Dynamics']])
    y_a_d = labeled_data["Teacher's Articulation & Dynamics"]

    models = load_models("Articulation & Dynamics")
    a_d_final_score = predict_all(models, x_a_d, y_a_d)
    a_d_confidence_interval = proportion_confint(count=((1 - a_d_final_score) * 6), nobs=6, alpha=0.1)

    ### Overall
    x_overall = pd.DataFrame(labeled_data[['Pitch', 'Tempo', 'Rhythm', 'Articulation', 'Dynamics']])
    y_overall = labeled_data["Teacher's Overall"]

    models = load_models("Overall")
    overall_final_score = predict_all(models, x_overall, y_overall)
    overall_confidence_interval = proportion_confint(count=((1 - overall_final_score) * 6), nobs=6, alpha=0.1)

    print(" ")
    print("###########")
    print("Final Results:")
    print("One Dim label score: " + str(one_dim_final_score) + ", error confidence interval: " + str(one_dim_confidence_interval))
    print("Pitch Score: " + str(pitch_final_score) + ", error confidence interval: " + str(pitch_confidence_interval))
    print("Tempo Score: " + str(tempo_final_score) + ", error confidence interval: " + str(tempo_confidence_interval))
    print("Rhythm Score: " + str(rhythm_final_score) + ", error confidence interval: " + str(rhythm_confidence_interval))
    print("A&D score: " + str(a_d_final_score) + ", error confidence interval: " + str(a_d_confidence_interval))
    print("Overall score: " + str(overall_final_score) + ", error confidence interval: " + str(overall_confidence_interval))
    print("###########")
    print(" ")


def load_models(name):
    models = []
    for i in range(1, 29):
        loaded_model = pickle.load(open('models/' + name + '/model_' + str(i) + '.sav', 'rb'))
        models.append(loaded_model)
    return models


def predict_all(models, x, y):
    cnt = 0
    for i in range(len(y)):
        y_hat_final = predict_from_models(models, pd.DataFrame(x.iloc[i]))
        if y_hat_final == y[i]:
            cnt += 1

    return cnt / len(y)


def predict_from_models(models, x):
    y_hat = []
    for model in models:
        y_hat.append(int(model.predict(x.to_numpy().reshape(1, -1))[0]))
    y_hat_final = max(set(y_hat), key=y_hat.count)
    return y_hat_final

if __name__ == "__main__":
    #plot_data_by_real_teachers("Music+evaluation_September+7%2C+2021_07.06.csv", "songs", number_of_fake_teachers=10)
    #train_test_real("Music+evaluation_September+7%2C+2021_07.06.csv", "songs", to_print=True)
    final_tests("Music+evaluation_September+7%2C+2021_07.06.csv", "songs")