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


def get_performance_grades(performance_grades_df: pd.DataFrame, number_of_teachers):
    grades = []
    for i in range(0, number_of_teachers):
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


def processSurveyResults(csv_path, folder_path, n=maxNumberOfTeachers):
    results_df = pd.read_csv(csv_path, dtype={
        'string_col': 'float16',
        'int_col': 'float16'
    }).fillna(-1)
    song_dict = {}
    i = 19
    for performance in SurveyPerformanceList:
        performance_class = getPerformance(folder_path, performance["name"], performance["player_name"])
        if performance_class is None:
            continue
        performance_grades_df = results_df.iloc[:, i:i + 8]
        grades_df = get_performance_grades(performance_grades_df, n)
        performance_class.teachers_grades = grades_df.to_numpy()
        performance_class.give_labels(majority_or_avg=True)
        # create/update song object
        i += 8
    return song_dict


def getDataForSL(csv_path, number_of_teachers, folder_path):
    song_dict = processSurveyResults(csv_path, number_of_teachers, folder_path)
    songs_labels = []
    for songs in song_dict:
        songs_labels.append()

    return songs_labels


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

    xlabel = "Number of Teachers"
    print_graph(one_dim, "One dimension Next step", x, xlabel=xlabel)
    print_graph(two_dim, "Two dimensions Next step", x, xlabel=xlabel)
    print_graph(pitch, "Pitch feature", x, xlabel=xlabel)
    print_graph(tempo, "Tempo feature", x, xlabel=xlabel)
    print_graph(rhythm, "Rhythm feature", x, xlabel=xlabel)
    print_graph(a_d, "Articulation and Dynamics feature", x, xlabel=xlabel)


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
    processSurveyResults("Music+evaluation_September+6,+2021_03.10.csv", "songs")
