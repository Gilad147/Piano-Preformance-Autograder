import pretty_midi
import Performance_class
import random
import os
import pandas as pd
import auxiliary


def fake_teachers_algorithm(from_midi_files_or_not, number_of_teachers, train_ratio, folder=None,
                            performances_data=None, is_testing=True):
    all_performances_grades = []
    if from_midi_files_or_not:
        basepath = folder + ' - fake data/'
        with os.scandir(basepath) as songs:
            for song in songs:
                if song.is_dir():
                    song_path = song.name + '/'
                    with os.scandir(basepath + song_path) as entries:
                        for entry in entries:
                            if entry.is_dir():
                                song_performances_path = entry.name + '/'
                            elif entry.is_file() and entry.name != '.DS_Store':
                                song_perfect_name = entry.name
                        with os.scandir(basepath + song_path + song_performances_path) as performances:
                            for performance in performances:
                                if performance.name != '.DS_Store':
                                    rhythm_feature, velocity_feature, duration_feature, pitch_feature, tempo_feature = Performance_class.Performance(
                                        path=basepath + song_path + song_performances_path + performance.name,
                                        name=entry.name,
                                        player_name='fake name',
                                        original_path=basepath + song_path + song_perfect_name) \
                                        .get_features()
                                    if rhythm_feature != -1:
                                        all_performances_grades.append(
                                            [rhythm_feature, velocity_feature, duration_feature, pitch_feature,
                                             tempo_feature])
    else:
        for performance in performances_data:
            rhythm_feature, velocity_feature, duration_feature, pitch_feature, tempo_feature = performance.get_features()
            if rhythm_feature != -1:
                all_performances_grades.append([rhythm_feature, velocity_feature, duration_feature, pitch_feature,
                                                tempo_feature])

    teachers_scores, teachers_next_step = fake_teachers_feedback(all_performances_grades, number=number_of_teachers)

    labeled_data_train_one_dimension = []  # explain that in order to prevent data leakeage (group leakage), we split *performances* randomly into train-test
    labeled_data_train_two_dimensions = []
    labeled_data_test = []
    train_number = round(len(all_performances_grades) * train_ratio)
    train_tuple = tuple(random.sample(range(0, len(all_performances_grades)), train_number))

    label_mapping = {"0": ["0", "-1"], "3": ["1", "-1"], "1": ["0", "0"], "2": ["0", "1"], "4": ["1", "0"],
                     "5": ["1", "1"]}

    for j in range(len(all_performances_grades)):  # for every performance --> j
        for i in range(len(teachers_scores[0])):  # for every teacher --> i
            if j in train_tuple:
                labeled_data_train_one_dimension.append([all_performances_grades[j][0],  # Rhythm tech score
                                           all_performances_grades[j][1],  # Dynamics tech score
                                           all_performances_grades[j][2],  # Articulation tech score
                                           all_performances_grades[j][3],  # Pitch tech score
                                           all_performances_grades[j][4],  # Tempo tech score
                                           teachers_scores[j][i][0],  # i Teacher's Pitch score
                                           teachers_scores[j][i][1],  # i Teacher's Rhythm score
                                           teachers_scores[j][i][2],  # i Teacher's Tempo score
                                           teachers_scores[j][i][3],  # i Teacher's Articulation & Dynamics score
                                           teachers_next_step[j][i]])  # i Teacher's next step

                labeled_data_train_two_dimensions.append(labeled_data_train_one_dimension[-1].copy())
                dim1, dim2 = label_mapping[labeled_data_train_two_dimensions[-1][9]][0], \
                             label_mapping[labeled_data_train_two_dimensions[-1][9]][1]

                labeled_data_train_two_dimensions[-1][9] = dim1
                labeled_data_train_two_dimensions[-1].append(dim2)

            else:
                labeled_data_test.append([all_performances_grades[j][0],  # Rhythm tech score
                                           all_performances_grades[j][1],  # Dynamics tech score
                                           all_performances_grades[j][2],  # Articulation tech score
                                           all_performances_grades[j][3],  # Pitch tech score
                                           all_performances_grades[j][4],  # Tempo tech score
                                           teachers_scores[j][i][0],  # i Teacher's Pitch score
                                           teachers_scores[j][i][1],  # i Teacher's Rhythm score
                                           teachers_scores[j][i][2],  # i Teacher's Tempo score
                                           teachers_scores[j][i][3],  # i Teacher's Articulation & Dynamics score
                                           teachers_next_step[j][i]])  # i Teacher's next step

    train_one_dimension = pd.DataFrame(labeled_data_train_one_dimension, columns=['Rhythm', 'Dynamics', 'Articulation', 'Pitch', 'Tempo',
                                                      "Teacher's Pitch", "Teacher's Rhythm", "Teacher's Tempo",
                                                      "Teacher's Articulation & Dynamics", 'label'])
    train_two_dimensions = pd.DataFrame(labeled_data_train_two_dimensions, columns=['Rhythm', 'Dynamics', 'Articulation', 'Pitch', 'Tempo',
                                                      "Teacher's Pitch", "Teacher's Rhythm", "Teacher's Tempo",
                                                      "Teacher's Articulation & Dynamics", 'label dim 1', 'label dim 2'])
    test = pd.DataFrame(labeled_data_test, columns=['Rhythm', 'Dynamics', 'Articulation', 'Pitch', 'Tempo',
                                                    "Teacher's Pitch", "Teacher's Rhythm", "Teacher's Tempo",
                                                    "Teacher's Articulation & Dynamics", 'label'])

    # df_labeled = pd.DataFrame(labeled_data, columns= ['Rhythm', 'Dynamics', 'Articulation', 'Pitch', 'Tempo', 'label'])
    # test_algorithms(df_labeled)
    # print(df_labeled['label'].value_counts())
    # print(df_labeled)

    # train, test = train_test_split(df_labeled, test_size=0.3)
    if is_testing:
        auxiliary.test_algorithms_next_step_one_dimension(train_one_dimension, test, True, to_print=True)
        auxiliary.test_algorithms_next_step_two_dimensions(train_two_dimensions, test, True, to_print=True)
        auxiliary.test_algorithms_scores(train_one_dimension, test, "Pitch", to_print=True)
        auxiliary.test_algorithms_scores(train_one_dimension, test, "Tempo", to_print=True)
        auxiliary.test_algorithms_scores(train_one_dimension, test, "Rhythm", to_print=True)
        auxiliary.test_algorithms_scores(train_one_dimension, test, "Articulation & Dynamics", to_print=True)

    return train_one_dimension


def fake_teachers_feedback(performances, number):
    """
    return: [j][i] = [[perfomance_j_grade for teacher_i]]
    """
    next_step = [[] for x in range(len(performances))]
    scores = [[] for x in range(len(performances))]

    for i in range(number):
        pitch_unique_stumps = round(random.uniform(-0.15, 0.15), 2)
        rhythm_unique_stumps = round(random.uniform(-0.25, 0.25), 2)
        tempo_unique_stumps = round(random.uniform(-0.25, 0.25), 2)
        articulation_unique_stumps = round(random.uniform(-0.25, 0.25), 2)
        dynamics_unique_stumps = round(random.uniform(-0.3, 0.3), 2)

        pitch_unique_score = round(random.uniform(-0.2, 0.2), 2)
        rhythm_unique_score = round(random.uniform(-0.2, 0.2), 2)
        tempo_unique_score = round(random.uniform(-0.2, 0.2), 2)
        articulation_unique_score = round(random.uniform(-0.2, 0.2), 2)
        dynamics_unique_score = round(random.uniform(-0.2, 0.2), 2)

        teacher_i = Teacher(pitch_unique_stumps, rhythm_unique_stumps, tempo_unique_stumps,
                            articulation_unique_stumps, dynamics_unique_stumps,
                            pitch_unique_score, rhythm_unique_score, tempo_unique_score,
                            articulation_unique_score, dynamics_unique_score,
                            )

        for j in range(len(performances)):
            next_step[j].append(teacher_i.give_next_step_recco_stumps(rhythm_score=performances[j][0],
                                                    dynamics_score=performances[j][1],
                                                    articulation_score=performances[j][2],
                                                    pitch_score=performances[j][3],
                                                    tempo_score=performances[j][4]))

            performance_grades = [teacher_i.give_scores(performances[j][3], "Pitch"),
                                  teacher_i.give_scores(performances[j][0], "Rhythm"),
                                  teacher_i.give_scores(performances[j][4], "Tempo"),
                                  teacher_i.give_scores([performances[j][2], performances[j][1]], "Articulation & Dynamics")]
            scores[j].append(performance_grades)

    return scores, next_step


class Teacher:
    def __init__(self, pitch_unique_stumps, rhythm_unique_stumps, tempo_unique_stumps, articulation_unique_stumps,
                 dynamics_unique_stumps,
                 pitch_unique_score, rhythm_unique_score, tempo_unique_score, articulation_unique_score,
                 dynamics_unique_score):
        self.pitch_unique_stumps = pitch_unique_stumps
        self.rhythm_unique_stumps = rhythm_unique_stumps
        self.tempo_unique_stumps = tempo_unique_stumps
        self.articulation_unique_stumps = articulation_unique_stumps
        self.dynamics_unique_stumps = dynamics_unique_stumps

        self.pitch_unique_score = pitch_unique_score
        self.rhythm_unique_score = rhythm_unique_score
        self.tempo_unique_score = tempo_unique_score
        self.articulation_unique_score = articulation_unique_score
        self.dynamics_unique_score = dynamics_unique_score

    def give_next_step_recco_stumps(self, pitch_score, rhythm_score, tempo_score, articulation_score, dynamics_score):
        """
        mapping:
            same piece = 0-2:
                0 = slower
                1 = same pace
                2 = faster
            new piece = 3-5
                3 = easier
                4 = same level
                5 = harder
        """

        pitch_score += self.pitch_unique_stumps
        rhythm_score += self.rhythm_unique_stumps
        tempo_score += self.tempo_unique_stumps
        articulation_score += self.articulation_unique_stumps
        dynamics_score += self.dynamics_unique_stumps

        if pitch_score < 0.3:
            return "3"
        if 0.3 <= pitch_score < 0.5:
            return "0"
        if 0.5 <= pitch_score < 0.75:
            if 0 < tempo_score < 0.5:
                return "0"
            if -0.5 < tempo_score < 0:
                return "2"
            if rhythm_score < 0.5 or dynamics_score < 0.5 or articulation_score < 0.3:
                return "0"
            if 0.5 <= rhythm_score < 0.8 or \
                    0.5 <= abs(tempo_score) < 0.8 or dynamics_score < 0.8 or \
                    0.3 <= articulation_score < 0.5:
                return "1"
            else:
                return "5"
        else:
            cnt = 0
            if rhythm_score < 0.6:
                cnt += 1
            if abs(tempo_score) < 0.6:
                cnt += 1
            if dynamics_score < 0.6:
                cnt += 1
            if articulation_score < 0.3:
                cnt += 1
            if cnt >= 2:
                return "4"
            else:
                return "5"

    def give_scores(self, feature_score, feature_name):

        if feature_name == "Pitch":
            feature_score += self.pitch_unique_score
        elif feature_name == "Rhythm":
            feature_score += self.rhythm_unique_score
        elif feature_name == "Tempo":
            feature_score += self.tempo_unique_score
        elif feature_name == "Articulation & Dynamics":
            feature_score[0] += self.articulation_unique_score
            feature_score[1] += self.dynamics_unique_score
            feature_score = sum(feature_score)/2

        if feature_score < 0.2:
            return "0"
        elif 0.2 <= feature_score < 0.4:
            return "1"
        elif 0.4 <= feature_score < 0.6:
            return "2"
        elif 0.6 <= feature_score < 0.8:
            return "3"
        else:
            return "4"



