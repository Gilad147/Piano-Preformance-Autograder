import bisect
from datetime import datetime
import os
import numpy as np


# functions for numeric grades to written feedback
def determine_rhythm_feedback(scores, breakpoints=[0.4, 0.7],
                              grades=None):
    if grades is None:
        grades = ['you do not pay enough attention to rhythm',
                  'try to pay more attention to rhythmic transitions',
                  'great sense of rhythm, keep up']
    i = bisect.bisect(breakpoints, scores)
    return grades[i]


def determine_pitch_feedback(scores, breakpoints=[0.4, 0.7],
                             grades=None):
    if grades is None:
        grades = ['you have a lot of missing or wrong notes',
                  'try to be more accurate with note sequences',
                  'great melody, keep up']
    i = bisect.bisect(breakpoints, scores)
    return grades[i]


def determine_tempo_feedback(scores, breakpoints=[0.4, 0.7],
                             grades=None):
    if grades is None:
        grades = ['your feel of tempo is not consistent',
                  'try to play with a better feel the tempo',
                  'very steady playing, keep up']
    i = bisect.bisect(breakpoints, scores)
    return grades[i]


def determine_velocity_feedback(scores, breakpoints=[0.4, 0.7],
                                grades=None):
    if grades is None:
        grades = ['you are not sensitive enough to notes volume',
                  'try to be more sensitive to the dynamics',
                  'good expressive ability, keep up']
    i = bisect.bisect(breakpoints, scores)
    return grades[i]


def determine_overall_feedback(scores, breakpoints=[0.4, 0.8, 0.9],
                               grades=None):
    if grades is None:
        grades = ['there is still some work to do',
                  'you did a good job', 'you played very well',
                                        'it was excellent playing']
    i = bisect.bisect(breakpoints, scores)
    return grades[i]


def save_feedback_to_directory(path, grades, feedback, recommendation):
    text_file = open(path, "w")
    text_file.write("Numeric scores: "
                    "\nPitch-" + str(grades[3]) +
                    "\nTempo-" + str(grades[4]) +
                    "\nRhythm-" + str(grades[0]) +
                    "\nArticulation-" + str(grades[1]))
    text_file.write("\n\nVerbal Feedback: \n" + feedback[73:])
    text_file.write("Recommendation: " + recommendation)
    text_file.close()


def feedback_for_exit_application(grades, recommendation, feedback_path):
    overall_feedback = determine_overall_feedback(np.average(np.array(grades)))
    pitching_feedback = determine_pitch_feedback(grades[3])
    tempo_feedback = determine_tempo_feedback(grades[4])
    rhythm_feedback = determine_rhythm_feedback(grades[0])
    velocity_feedback = determine_velocity_feedback(grades[1])
    recommendation_dictionary = {'0': 'play slower', '1': 'play it again', '2': 'play faster',
                                 '3': 'play a easier piece', '4': 'play another piece',
                                 '5': 'play a harder piece'}
    feedback_message = overall_feedback + '\n' \
                       + 'Please pay attention to this technicals: ' + '\n' + '\n' \
                       + 'Pitch: ' + pitching_feedback + '\n' + '\n' \
                       + 'Tempo: ' + tempo_feedback + '\n' + '\n' \
                       + 'Rhythm: ' + rhythm_feedback + '\n' + '\n' \
                       + 'Articulation: ' + velocity_feedback + '\n' + '\n'
    save_feedback_to_directory(feedback_path, grades, feedback_message, recommendation_dictionary[recommendation])
    return feedback_message, recommendation_dictionary[recommendation]


def reformat_file_by_type(file_name):
    # getting rid of file suffix
    if 'ly' in file_name:
        return file_name[:-3]
    if 'png' in file_name:
        return file_name[:-4]
    if 'midi' in file_name:
        return file_name[:-5]


def next_piece_by_level(level, song_name):
    """Randomly draw a piece by the level given
        level: string digit stating the desired songbook level
        :returns:
                next_chart = the absolute path for the next chart
                next_midi  = the absolute path for the next midi
                name = the name of the chosen song without suffix
                next_level = the level of the chosen song

    # ATTENTION: treats each song as having .mid, .png, .ly files
    # Change is necessary to prevent bugging if other files are present
    """
    next_level = level
    if level <= 0:
        next_level = 0
    if level >= 3:
        next_level = 3
    directories_by_levels = {0: 'initial exercises', 1: 'initial exercises2', 2: 'initial exercises3',
                             3: 'hebrew Collection'}
    root_path = os.path.dirname(
        os.path.abspath('project directory/songs' + '/' + directories_by_levels[next_level]))
    next_songbook_path = os.path.join(root_path, directories_by_levels[next_level])
    files_in_songbook = os.listdir(next_songbook_path)
    chosen_piece = np.random.randint(0, len(files_in_songbook) / 3)
    name = reformat_file_by_type(files_in_songbook[chosen_piece])
    while name == song_name:
        chosen_piece = np.random.randint(0, len(files_in_songbook) / 3)
        name = reformat_file_by_type(files_in_songbook[chosen_piece])
    song_files_indexes = np.char.startswith(np.array(files_in_songbook), name)
    song_files = np.array(files_in_songbook)[song_files_indexes]
    next_chart = song_files[np.char.endswith(song_files, 'png')]
    midi_index = 1 - np.logical_or(np.char.endswith(song_files, 'png'), np.char.endswith(song_files, 'ly'))
    next_midi = song_files[midi_index == 1]
    next_chart = os.path.join(next_songbook_path, next_chart[0])
    next_midi = os.path.join(next_songbook_path, next_midi[0])
    return next_chart, next_midi, name, next_level


def find_new_tempo(tempo, recommendation):
    if int(recommendation) == 0:
        tempo -= 20
        if tempo < 60:
            tempo = 60
    if int(recommendation) == 2:
        tempo += 20
        if tempo > 160:
            tempo = 160
    tempo = round(tempo)
    return tempo


def next_action_by_recommendation(recommendation, chart_path, original_midi, song_name, song_level, tempo):
    # interprets predicted recommendation for student into the next trial settings
    tempo = find_new_tempo(tempo, recommendation)
    if int(recommendation) < 3:
        return chart_path, original_midi, song_name, song_level, tempo
    else:
        if int(recommendation) == 4:
            return next_piece_by_level(song_level, song_name), tempo
        else:
            if int(recommendation) == 3:
                return next_piece_by_level(song_level - 1, song_name), tempo
            else:
                return next_piece_by_level(song_level + 1, song_name), tempo


def directories(Data_Played, subject_id, song_name):
    """Saves data played into personal student directory
        Data_Played: the midi events table designed as desired
        :returns:
                midi_path_to_save = absolute path of personal student directory to save midi file played
    """
    root_path = os.path.dirname(os.path.abspath("Piano-Preformance-Auto")) + "/Students recordings"
    date_directory = datetime.date(datetime.now())
    complete_date_directory = os.path.join(root_path, str(date_directory))
    if not os.path.exists(complete_date_directory):
        os.mkdir(complete_date_directory)
    directory_of_subject = subject_id
    complete_subject_directory = os.path.join(complete_date_directory, directory_of_subject)
    if not os.path.exists(complete_subject_directory):
        os.mkdir(complete_subject_directory)
    now_time = str(datetime.time(datetime.now()))
    if len(now_time) == 15:
        now_time = now_time[:-7]
    else:
        now_time = now_time[:-6]
    completeName = os.path.join(complete_subject_directory, now_time + "-" + song_name + ".txt")
    midi_path_to_save = os.path.join(complete_subject_directory, now_time + "-" + song_name + ".midi")
    with open(completeName, 'w') as output:
        for row in Data_Played:
            output.write(str(row) + '\n')
    return midi_path_to_save
