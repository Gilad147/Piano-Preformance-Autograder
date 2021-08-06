import pretty_midi
import Performance_class
import random
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def fake_teachers_algorithm(from_midi_files_or_not, number_of_teachers, train_ratio, folder=None, performances_data=None, ):
    all_performances_grades = []
    if from_midi_files_or_not:
        basepath = folder + '/'
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
                                    all_performances_grades.append(
                                        [rhythm_feature, velocity_feature, duration_feature, pitch_feature,
                                         tempo_feature])
    else:
        for performance in performances_data:
            rhythm_feature, velocity_feature, duration_feature, pitch_feature, tempo_feature = performance.get_features()
            all_performances_grades.append([rhythm_feature, velocity_feature, duration_feature, pitch_feature,
                                            tempo_feature])

    teachers_grades, teachers_unique = fake_teachers_feedback(all_performances_grades, number=number_of_teachers)

    labeled_data_train = [] #explain that in order to prevent data leakeage (group leakage), we split *teachers* randomly into train-test
    labeled_data_test = []
    train_number = round(number_of_teachers * train_ratio)
    train_tuple = tuple(random.sample(range(0, number_of_teachers), train_number))
    # labeled_data = []

    for i in range(len(all_performances_grades)):
        for j in range(len(teachers_grades)):
            if j in train_tuple:
                labeled_data_train.append([all_performances_grades[i][0],
                                           all_performances_grades[i][1],
                                           all_performances_grades[i][2],
                                           all_performances_grades[i][3],
                                           all_performances_grades[i][4],
                                           teachers_grades[j][i]])
            else:
                labeled_data_test.append([all_performances_grades[i][0],
                                          all_performances_grades[i][1],
                                          all_performances_grades[i][2],
                                          all_performances_grades[i][3],
                                          all_performances_grades[i][4],
                                          teachers_grades[j][i]])

    train = pd.DataFrame(labeled_data_train, columns=['Rhythm', 'Dynamics', 'Articulation', 'Pitch', 'Tempo', 'label'])
    test = pd.DataFrame(labeled_data_test, columns=['Rhythm', 'Dynamics', 'Articulation', 'Pitch', 'Tempo', 'label'])

    # df_labeled = pd.DataFrame(labeled_data, columns= ['Rhythm', 'Dynamics', 'Articulation', 'Pitch', 'Tempo', 'label'])
    #test_algorithms(df_labeled)
    #print(df_labeled['label'].value_counts())
    #print(df_labeled)

    #train, test = train_test_split(df_labeled, test_size=0.3)

    test_algorithms(train, test)

def test_algorithms(labeled_data_train, labeled_data_test):

    ### random forest
    x_train = labeled_data_train.drop(columns=['label'])
    y_train = labeled_data_train['label']

    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    x_test = labeled_data_test.drop(columns=['label'])
    y_test = labeled_data_test['label']

    print(model.score(x_test, y_test))


def fake_teachers_feedback(performances, number):
    """
    return: [i][j] = [[perfomance_j_grade for teacher_i]]
    """
    results = []
    teachers_unique_data = []
    for i in range(number):
        pitch_unique = round(random.uniform(-0.1, 0.1), 2)
        rhythm_unique = round(random.uniform(-0.2, 0.2), 2)
        tempo_unique = round(random.uniform(-0.2, 0.2), 2)
        articulation_unique = round(random.uniform(-0.2, 0.2), 2)
        dynamics_unique = round(random.uniform(-0.3, 0.3), 2)
        teachers_unique_data.append([pitch_unique, rhythm_unique, tempo_unique, articulation_unique, dynamics_unique])

        teacher_i = Teacher(pitch_unique, rhythm_unique, tempo_unique, articulation_unique, dynamics_unique)
        result_teacher_i = []
        for performance in performances:
            result_teacher_i.append(teacher_i.grade(rhythm_score=performance[0],
                                                    dynamics_score=performance[1],
                                                    articulation_score=performance[2],
                                                    pitch_score=performance[3],
                                                    tempo_score=performance[4]))
        results.append(result_teacher_i)

    return results, teachers_unique_data

class Teacher:
    def __init__(self, pitch_unique, rhythm_unique, tempo_unique, articulation_unique, dynamics_unique):
        self.pitch_unique = pitch_unique
        self.rhythm_unique = rhythm_unique
        self.tempo_unique = tempo_unique
        self.articulation_unique = articulation_unique
        self.dynamics_unique = dynamics_unique

    def grade(self, pitch_score, rhythm_score, tempo_score, articulation_score, dynamics_score):
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

        pitch_score += self.pitch_unique
        rhythm_score += self.rhythm_unique
        tempo_score += self.tempo_unique
        articulation_score += self.articulation_unique
        dynamics_score += self.dynamics_unique

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
