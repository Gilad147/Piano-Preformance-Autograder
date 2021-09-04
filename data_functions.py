import pickle
import random
import matplotlib.pyplot as plt
import pandas as pd
import Performance_class
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
maxNumberOfTeachers = 6


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
    }).fillna(-1)
    results = []
    for i in range(0, n):
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
    features = []
    results = processSurveyResults(csv_path, number_of_teachers)
    for song in SurveyPerformanceList:
        performance, rhythm_feature, velocity_feature, duration_feature, pitch_feature, tempo_feature = \
            getFeatures(folder_path, song["name"], song["player_name"])
        performances.append(performance)
        scores = {"Rhythm": rhythm_feature, 'Dynamics': velocity_feature, 'Articulation': duration_feature,
                  'Pitch': pitch_feature, 'Tempo': tempo_feature}
        features.append(scores)

    for teacher in results:
        for i, grades in enumerate(teacher):
            if grades["rhythm"] != -1 and grades["a_d"] != -1 and grades["pitch"] != -1 and grades["tempo"] != -1 and \
                    grades["next_step"] != -1:
                all_performances_predications.append(
                    [features[i]["Rhythm"], features[i]['Dynamics'], features[i]['Articulation'],
                     features[i]['Pitch'],
                     features[i]['Tempo'], grades["rhythm"], grades["a_d"], grades["pitch"], grades["tempo"],
                     grades["next_step"]])

    return all_performances_predications, performances


def trainAndTest(csv_path, folder_path, train_ratio,
                 fake_teachers=0, fake_songs=0, number_of_teachers=maxNumberOfTeachers):
    label_mapping = {0: ["0", "-1"], 3: ["1", "-1"], 1: ["0", "0"], 2: ["0", "1"], 4: ["1", "0"],
                     5: ["1", "1"]}
    all_performances_predications, performances = getDataForSL(csv_path, number_of_teachers,
                                                               folder_path)
    labeled_data_train_one_dimension = []
    labeled_data_train_two_dimensions = []
    labeled_data_test = []
    numOfPerformances = len(all_performances_predications)
    train_number = round(numOfPerformances * train_ratio)
    train_tuple = tuple(random.sample(range(0, numOfPerformances), train_number))
    for i in range(numOfPerformances):
        if i in train_tuple:
            labeled_data_train_one_dimension.append(all_performances_predications[i])
            labeled_data_train_two_dimensions.append(labeled_data_train_one_dimension[-1].copy())
            dim1, dim2 = label_mapping[labeled_data_train_two_dimensions[-1][9]][0], \
                         label_mapping[labeled_data_train_two_dimensions[-1][9]][1]

            labeled_data_train_two_dimensions[-1][9] = dim1
            labeled_data_train_two_dimensions[-1].append(dim2)
        else:
            labeled_data_test.append(all_performances_predications[i])

    train_one_dimension = pd.DataFrame(labeled_data_train_one_dimension,
                                       columns=['Rhythm', 'Dynamics', 'Articulation', 'Pitch', 'Tempo',
                                                "Teacher's Pitch", "Teacher's Rhythm", "Teacher's Tempo",
                                                "Teacher's Articulation & Dynamics", 'label'])
    train_two_dimensions = pd.DataFrame(labeled_data_train_two_dimensions,
                                        columns=['Rhythm', 'Dynamics', 'Articulation', 'Pitch', 'Tempo',
                                                 "Teacher's Pitch", "Teacher's Rhythm", "Teacher's Tempo",
                                                 "Teacher's Articulation & Dynamics", 'label dim 1', 'label dim 2'])
    test = pd.DataFrame(labeled_data_test, columns=['Rhythm', 'Dynamics', 'Articulation', 'Pitch', 'Tempo',
                                                    "Teacher's Pitch", "Teacher's Rhythm", "Teacher's Tempo",
                                                    "Teacher's Articulation & Dynamics", 'label'])

    if fake_teachers > 0:
        generated_data = auxiliary.generate_random_mistakes_data('songs/performances', fake_songs, False)
        generated_data = performances + generated_data
        fake_train_dim_1, fake_train_dim_2 = Automated_teacher.fake_teachers_algorithm(False,
                                                                                       performances_data=generated_data,
                                                                                       number_of_teachers=fake_teachers,
                                                                                       train_ratio=1, is_testing=False)
        train_one_dimension = pd.concat(objs=[train_one_dimension, fake_train_dim_1])
        train_two_dimensions = pd.concat(objs=[train_two_dimensions, fake_train_dim_2])
        train_one_dimension = train_one_dimension.astype("float32")
        train_two_dimensions.astype("float32")
    one_dim_scores = auxiliary.test_algorithms_next_step_one_dimension(train_one_dimension, test, True, to_print=False)
    two_dim_scores = auxiliary.test_algorithms_next_step_two_dimensions(train_two_dimensions, test, True,
                                                                        to_print=False)
    pitch_scores = auxiliary.test_algorithms_scores(train_one_dimension, test, "Pitch", to_print=False)
    tempo_scores = auxiliary.test_algorithms_scores(train_one_dimension, test, "Tempo", to_print=False)
    rhythm_scores = auxiliary.test_algorithms_scores(train_one_dimension, test, "Rhythm", to_print=False)
    a_d_scores = auxiliary.test_algorithms_scores(train_one_dimension, test, "Articulation & Dynamics", to_print=False)
    return one_dim_scores, two_dim_scores, pitch_scores, tempo_scores, rhythm_scores, a_d_scores


def print_graph(scores, name, index, xlabel="Number of Fake Teachers and Fake Songs"):
    scores_df = pd.DataFrame(scores, columns=["Random Forest (gini)", "Random Forest (entropy)", "Logistic Regression",
                                              "KNN(max)", "Multi-layer Perceptron"],
                             index=index)

    plt.figure()
    scores_df.plot(xlabel=xlabel, ylabel="Model score", title=name + " predication scores",
                   ylim=(0, 1), figsize=(12, 12), grid=True)
    plt.show()


def plot_data_by_real_teachers(csv_path, folder_path, train_ratio, number_of_teachers=maxNumberOfTeachers):
    one_dim = []
    two_dim = []
    pitch = []
    tempo = []
    rhythm = []
    a_d = []
    x = []
    for i in range(2, number_of_teachers + 1):
        one_dim_scores, two_dim_scores, pitch_scores, tempo_scores, rhythm_scores, a_d_scores = \
            trainAndTest(csv_path,
                         folder_path,
                         train_ratio,
                         number_of_teachers=i)
        one_dim.append(one_dim_scores)
        two_dim.append(two_dim_scores)
        pitch.append(pitch_scores)
        tempo.append(tempo_scores)
        rhythm.append(rhythm_scores)
        a_d.append(a_d_scores)
        x.append(i)
        print("number of teachers: " + str(i) + " scoring done")

    x_label = "Number of Teachers"
    print_graph(one_dim, "One dimension Next step", x, xlabel="Number of Teachers")
    print_graph(two_dim, "Two dimensions Next step", x, xlabel="Number of Teachers")
    print_graph(pitch, "Pitch feature", x, xlabel="Number of Teachers")
    print_graph(tempo, "Tempo feature", x, xlabel="Number of Teachers")
    print_graph(rhythm, "Rhythm feature", x, xlabel="Number of Teachers")
    print_graph(a_d, "Articulation and Dynamics feature", x, xlabel="Number of Teachers")


def plot_fake_data(csv_path, folder_path, train_ratio, fake_songs_max, fake_teachers_max, fake_songs_min=0):
    one_dim = []
    two_dim = []
    pitch = []
    tempo = []
    rhythm = []
    a_d = []
    x = []
    one_dim_scores, two_dim_scores, pitch_scores, tempo_scores, rhythm_scores, a_d_scores = \
        trainAndTest(csv_path,
                     folder_path,
                     train_ratio,
                     fake_teachers=0,
                     fake_songs=0)
    one_dim.append(one_dim_scores)
    two_dim.append(two_dim_scores)
    pitch.append(pitch_scores)
    tempo.append(tempo_scores)
    rhythm.append(rhythm_scores)
    a_d.append(a_d_scores)
    x.append((0, 0))

    for i in range(1, fake_teachers_max):
        for j in range(fake_songs_min, fake_songs_max + 1):
            one_dim_scores, two_dim_scores, pitch_scores, tempo_scores, rhythm_scores, a_d_scores = trainAndTest(
                csv_path,
                folder_path,
                train_ratio,
                fake_teachers=i,
                fake_songs=j)
            one_dim.append(one_dim_scores)
            two_dim.append(two_dim_scores)
            pitch.append(pitch_scores)
            tempo.append(tempo_scores)
            rhythm.append(rhythm_scores)
            a_d.append(a_d_scores)
            x.append((i, j))
            print("number of teachers: " + str(i) + ", number of fake songs:" + str(j) + " scoring done")
    index = pd.MultiIndex.from_tuples(x, names=["Teachers", "Fake Songs"])
    print_graph(one_dim, "One dimension Next step", index)
    print_graph(two_dim, "Two dimensions Next step", index)
    print_graph(pitch, "Pitch feature", index)
    print_graph(tempo, "Tempo feature", index)
    print_graph(rhythm, "Rhythm feature", index)
    print_graph(a_d, "Articulation and Dynamics feature", index)


def choose_model(model, filename='finalized_next_step_model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


if __name__ == "__main__":
    plot_fake_data("Music+evaluation_August+30,+2021_03.17.csv", "songs", train_ratio=0.5, fake_songs_min=3,
                   fake_songs_max=4, fake_teachers_max=10)
