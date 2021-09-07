import pickle
import random
import matplotlib.pyplot as plt
import pandas as pd
import Performance_class
import Automated_teacher
import Song_Class
import auxiliary
import numpy as np

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
        song_name = performance["name"]
        performance_class = getPerformance(folder_path, song_name, performance["player_name"])
        if performance_class is None:
            continue
        performance_grades_df = results_df.iloc[:, i:i + 8]
        grades_df = get_performance_grades(performance_grades_df, n)
        performance_class.teachers_grades = grades_df.to_numpy()
        performance_class.give_labels(majority_or_avg=True)
        labels = performance_class.labels
        rhythm_feature, velocity_feature, duration_feature, pitch_feature, tempo_feature = performance_class.get_features()
        performance_attributes = {'Rhythm': rhythm_feature, 'Dynamics': velocity_feature,
                                  'Articulation': duration_feature, 'Pitch': pitch_feature, 'Tempo': tempo_feature,
                                  "Teacher's Pitch": labels[0], "Teacher's Rhythm": labels[1],
                                  "Teacher's Tempo": labels[2],
                                  "Teacher's Articulation & Dynamics": labels[3], 'label': labels[4]}
        performance_attributes_df = pd.Series(performance_attributes)
        if song_name not in song_dict:
            new_song = Song_Class.Song(performance["name"])
            song_dict[song_name] = new_song

        song_dict[song_name].performances = song_dict[song_name].performances.append(performance_attributes_df,
                                                                                     ignore_index=True)
        i += 8
    return song_dict


def getDataForSL(csv_path, folder_path, train_ratio, number_of_teachers=maxNumberOfTeachers, fake_teachers=0):
    song_dict = processSurveyResults(csv_path, folder_path, number_of_teachers)
    song_set = set(song_dict.values())
    train_number = round(10 * train_ratio)
    train_tuple = tuple(random.sample(range(0, 10), train_number))
    label_mapping = {0: ["0", "-1"], 3: ["1", "-1"], 1: ["0", "0"], 2: ["0", "1"], 4: ["1", "0"],
                     5: ["1", "1"]}

    train_one_dim = pd.DataFrame(columns=['Rhythm', 'Dynamics', 'Articulation', 'Pitch', 'Tempo',
                                          "Teacher's Pitch", "Teacher's Rhythm", "Teacher's Tempo",
                                          "Teacher's Articulation & Dynamics", 'label'])
    labeled_two_dim = []

    test = pd.DataFrame(columns=['Rhythm', 'Dynamics', 'Articulation', 'Pitch', 'Tempo',
                                 "Teacher's Pitch", "Teacher's Rhythm", "Teacher's Tempo",
                                 "Teacher's Articulation & Dynamics", 'label'])
    for i, song in enumerate(song_set):
        if i in train_tuple:
            train_one_dim = train_one_dim.append(song.performances)
            dim_one_grade = song.performances.to_numpy()
            dim_two_grades = np.zeros((dim_one_grade.shape[0], 11))
            dim_two_grades[:, :-1] = dim_one_grade
            for grade in dim_two_grades:
                dim1, dim2 = label_mapping[grade[9]][0], label_mapping[grade[9]][1]
                grade[9] = dim1
                grade[10] = dim2
                labeled_two_dim.append(grade)

        else:
            test = test.append(song.performances)
    if fake_teachers:
        generated_data = auxiliary.generate_random_mistakes_data('songs/new songs', 100, False)

        train_one_dim, train_two_dim = Automated_teacher.fake_teachers_algorithm(False,
                                                                                 performances_data=generated_data,
                                                                                 number_of_teachers=fake_teachers,
                                                                                 train_ratio=1, majority_or_avg=True)
    else:
        train_two_dim = pd.DataFrame(labeled_two_dim, columns=['Rhythm', 'Dynamics', 'Articulation', 'Pitch', 'Tempo',
                                                               "Teacher's Pitch", "Teacher's Rhythm", "Teacher's Tempo",
                                                               "Teacher's Articulation & Dynamics", 'label dim 1',
                                                               'label dim 2'])

    return train_one_dim.astype("float32"), train_two_dim.astype("float32"), test.astype("float32")


def print_graph(scores, name, index, xlabel="Number of Fake Teachers and Fake Songs"):
    scores_df = pd.DataFrame(scores, columns=["Random Forest (gini)", "Random Forest (entropy)", "Logistic Regression",
                                              "KNN(max)", "Multi-layer Perceptron"],
                             index=index)

    plt.figure()
    scores_df.plot(xlabel=xlabel, ylabel="Model score", title=name + " predication scores",
                   ylim=(0, 1), figsize=(12, 12), grid=True)
    plt.show()


def plot_data_by_real_teachers(csv_path, folder_path, fake_teachers=0):
    final_one_dim = []
    final_two_dim = []
    final_pitch = []
    final_tempo = []
    final_rhythm = []
    final_a_d = []

    x = [i / 10 for i in range(5, 10)]
    for j in range(5):
        one_dim = []
        two_dim = []
        pitch = []
        tempo = []
        rhythm = []
        a_d = []
        for i in range(5, 7):
            train_one_dim, train_two_dim, test = getDataForSL(csv_path,
                                                              folder_path,
                                                              train_ratio=i / 10, fake_teachers = fake_teachers)
            one_dim_scores, two_dim_scores, pitch_scores, tempo_scores, rhythm_scores, a_d_scores = \
                auxiliary.trainAndTest(train_one_dim, train_two_dim, test, True)
            one_dim.append(one_dim_scores)
            two_dim.append(two_dim_scores)
            pitch.append(pitch_scores)
            tempo.append(tempo_scores)
            rhythm.append(rhythm_scores)
            a_d.append(a_d_scores)
            print("scoring done for train ratio: " + str(x[i - 5]))
        final_one_dim.append(one_dim)
        final_two_dim.append(two_dim)
        final_pitch.append(pitch)
        final_tempo.append(tempo)
        final_rhythm.append(rhythm)
        final_a_d.append(a_d)
    final_one_dim = np.array(final_one_dim).mean(axis=0)
    final_two_dim = np.array(final_two_dim).mean(axis=0)
    final_pitch = np.array(final_pitch).mean(axis=0)
    final_tempo = np.array(final_tempo).mean(axis=0)
    final_rhythm = np.array(final_rhythm).mean(axis=0)
    final_a_d = np.array(final_a_d).mean(axis=0)
    xlabel = "Train Ratio"
    print_graph(final_one_dim, "One dimension Next step", x, xlabel=xlabel)
    print_graph(final_two_dim, "Two dimensions Next step", x, xlabel=xlabel)
    print_graph(final_pitch, "Pitch feature", x, xlabel=xlabel)
    print_graph(final_tempo, "Tempo feature", x, xlabel=xlabel)
    print_graph(final_rhythm, "Rhythm feature", x, xlabel=xlabel)
    print_graph(final_a_d, "Articulation and Dynamics feature", x, xlabel=xlabel)


def choose_model(model, filename='finalized_next_step_model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


if __name__ == "__main__":
    plot_data_by_real_teachers("Music+evaluation_September+6,+2021_03.10.csv", "songs")
