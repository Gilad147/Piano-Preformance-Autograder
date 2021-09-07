import itertools
import math
import os
from datetime import datetime
import bisect
import pyaudio
import pygame
import pygame.midi
from pygame.locals import *
from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
import tkinter.font as tkFont
import numpy as np
from pygame.midi import midi_to_frequency
from Performance_class import Performance
from auxiliary import np2mid
import pretty_midi


def midi(chart_path, original_midi, subject_id, song_name, song_level):
    """Initialize midi trial with given arguments
        chart_path: an absolute path of the chart that is given to student
        original_midi: an absolute path of original midi track of chosen song
        subject_id: a 9 digits israeli ID
        song_name: the name of the song without file suffix
        song_level: the level of the songbook the song is taken from
    """
    Raw_Input = np.zeros((1, 4))
    Recording_Pushed_Last = False

    def input_design(Raw_Input):
        """Creates the desired design of the midi events table
            Raw_Input:
            :returns: the midi events table - each event = [start, end, pitch, velocity]
        """
        New_Input = np.zeros((1, 4))
        i = 1
        for row in range(len(Raw_Input)):
            if Raw_Input[row][1] == 144:
                # a key stroke consists of the 144 sign
                # a key being released consists of the 128 sign
                New_Input = np.append(New_Input.tolist(), [Raw_Input[row]], axis=0)
                Found_next = False
                indexer = row
                while not Found_next:
                    indexer += 1
                    if indexer == len(Raw_Input):
                        break
                    if Raw_Input[row][2] == Raw_Input[indexer][2]:
                        New_Input[i][1] = Raw_Input[indexer][0]
                        Found_next = True
                        i += 1
        return New_Input[1:].astype(float)

    # functions for numeric grades to written feedback
    def determine_rhythm_feedback(scores, breakpoints=[0.4, 0.8],
                                  grades=None):
        if grades is None:
            grades = ['you do not pay enough attention to rhythm',
                      'try to pay more attention to rhythmic transitions',
                      'great sense of rhythm, keep up']
        i = bisect.bisect(breakpoints, scores)
        return grades[i]

    def determine_pitch_feedback(scores, breakpoints=[0.4, 0.8],
                                 grades=None):
        if grades is None:
            grades = ['you have a lot of missing or wrong notes',
                      'try to be more accurate with note sequences',
                      'great melody, keep up']
        i = bisect.bisect(breakpoints, scores)
        return grades[i]

    def determine_tempo_feedback(scores, breakpoints=[0.4, 0.8],
                                 grades=None):
        if grades is None:
            grades = ['your feel of tempo is not consistent',
                      'try to play with a better feel the tempo',
                      'very steady playing, keep up']
        i = bisect.bisect(breakpoints, scores)
        return grades[i]

    def determine_velocity_feedback(scores, breakpoints=[0.4, 0.8],
                                    grades=None):
        if grades is None:
            grades = ['you are not sensitive enough to notes volume',
                      'try to be more sensitive to the dynamics',
                      'good expressive ability, keep up']
        i = bisect.bisect(breakpoints, scores)
        return grades[i]

    def determine_overall_feedback(scores, breakpoints=[0.4, 0.7, 0.9],
                                   grades=None):
        if grades is None:
            grades = ['there is still some work to do',
                      'you did a good job', 'you played very well'
                                            'it was excellent playing']
        i = bisect.bisect(breakpoints, scores)
        return grades[i]

    def save_feedback_to_directory(path, grades, feedback, recommendation):
        text_file = open(path, "w")
        text_file.write("Numeric scores: "
                        "\nPitch-" + str(grades[0]) +
                        "\nTempo-" + str(grades[1]) +
                        "\nRhythm-" + str(grades[2]) +
                        "\nArticulation-" + str(grades[3]))
        text_file.write("\n\nVerbal Feedback: \n" + feedback[73:])
        text_file.write("Recommendation: " + recommendation)
        text_file.close()

    def exit_application(grades, recommendation, feedback_path):
        """Handle end of trial and next step determination and set up
            grades: vector of size 4 with the predicted grades from ML processing
            recommendation: a string digit stating the predicted recommendation from ML processing
        """
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
        MsgBox = messagebox.askquestion('End of Trial', feedback_message + '\n' +
                                        'I advice you to '
                                        + recommendation_dictionary[recommendation] + '\n'
                                                                                      'do you want to keep training?',
                                        icon='warning')
        keyboard.close()
        pygame.midi.quit()
        pygame.quit()
        if MsgBox == 'no':
            return True

    def reformat_file_by_type(file_name):
        # getting rid of file suffix
        if 'ly' in file_name:
            return file_name[:-3]
        if 'png' in file_name:
            return file_name[:-4]
        if 'midi' in file_name:
            return file_name[:-5]

    def next_piece_by_level(level):
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

    def next_action_by_recommendation(recommendation):
        # interprets predicted recommendation for student into the next trial settings
        if int(recommendation) < 3:
            return chart_path, original_midi, song_name, song_level
        else:
            if int(recommendation) == 4:
                return next_piece_by_level(song_level)
            else:
                if int(recommendation) == 3:
                    return next_piece_by_level(song_level - 1)
                else:
                    return next_piece_by_level(song_level + 1)

    def directories(Data_Played):
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

    def stop(recording):
        # process stopping button instance
        if not recording:
            Msg_box_2 = messagebox.askquestion("Warning", "No data was recorded, do you want to exit?")
            if Msg_box_2 == 'no':
                return
            else:
                keyboard.close()
                pygame.midi.quit()
                pygame.quit()
                for widget in window.winfo_children():
                    widget.destroy()
                window.destroy()
                return
        Data_Played = input_design(Raw_Input[1:].astype('float32'))
        Data_Played[:, 0] = Data_Played[:, 0] / 1000
        Data_Played[:, 1] = Data_Played[:, 1] / 1000
        midi_path_to_save = directories(Data_Played)
        np2mid(Data_Played, midi_path_to_save, pretty_midi.PrettyMIDI(original_midi), True)
        performance = Performance(midi_path_to_save, song_name, subject_id, original_midi,
                                  prettyMidiFile_performance=None, prettyMidiFile_original=None)
        tech_grades = performance.get_features()
        print(tech_grades)

        # grades = performance.predict_grades(tech_grades)
        # recommendation = performance.predict_reccomendation(tech_grades)

        recommendation = '5'
        grades = tech_grades
        feedback_path = midi_path_to_save[:-5] + "-feedback.txt"
        stopping = exit_application(grades, recommendation, feedback_path)
        if stopping:
            for widget in window.winfo_children():
                widget.destroy()
            window.destroy()
        else:
            if int(recommendation) > 3:
                MsgBox = messagebox.askquestion('Practice continues',
                                                'Do you want to go on with my recommendation?\n\n '
                                                '*choosing "no" means playing the same piece again', icon='warning')
                if MsgBox == 'yes':
                    next_chart_path, next_original_midi, next_song_name, next_song_level = \
                        next_action_by_recommendation(recommendation)
            window.destroy()
            next_chart_path = chart_path
            next_original_midi = original_midi
            next_song_name = song_name
            next_song_level = song_level
            midi(next_chart_path, next_original_midi, subject_id, next_song_name, next_song_level)

    def record(data):
        data[0] = [1, 1, 1, 1]

    def clear(data):
        data[0] = [2, 2, 2, 2]

    # functions for midi trial attributes
    def place_stop_button():
        root_path = os.path.dirname(os.path.abspath("Piano-Preformance-Auto"))
        path = 'Images for GUI/stop button.png'
        complete_path = os.path.join(root_path, path)
        zoom = 0.4
        pixels_x, pixels_y = tuple([int(zoom * x) for x in Image.open(complete_path).size])
        photo2 = ImageTk.PhotoImage(Image.open(path).resize((pixels_x, pixels_y)))
        stop_button = Button(window, text='Stop', image=photo2, command=lambda: stop(Recording_Pushed_Last))
        stop_button.image = photo2
        stop_button.place(x=0, y=80)

    def place_record_button():
        root_path = os.path.dirname(os.path.abspath("Piano-Preformance-Auto"))
        path = 'Images for GUI/record button.png'
        complete_path = os.path.join(root_path, path)
        zoom = 0.4
        pixels_x, pixels_y = tuple([int(zoom * x) for x in Image.open(complete_path).size])
        photo2 = ImageTk.PhotoImage(Image.open(path).resize((pixels_x, pixels_y)))
        record_button = Button(window, text='Record', image=photo2, command=lambda: record(Raw_Input))
        record_button.image = photo2
        record_button.place(x=140, y=80)

    def place_clear_button():
        root_path = os.path.dirname(os.path.abspath("Piano-Preformance-Auto"))
        path = 'Images for GUI/clear button.png'
        complete_path = os.path.join(root_path, path)
        zoom = 0.2
        pixels_x, pixels_y = tuple([int(zoom * x) for x in Image.open(complete_path).size])
        photo2 = ImageTk.PhotoImage(Image.open(path).resize((pixels_x, pixels_y)))
        record_button = Button(window, text='Clear', image=photo2, command=lambda: clear(Raw_Input))
        record_button.image = photo2
        record_button.place(x=280, y=80)

    def place_note_chart(path):
        zoom = 1.8
        pixels_x, pixels_y = tuple([int(zoom * x) for x in Image.open(path).size])
        img = ImageTk.PhotoImage(Image.open(path).resize((pixels_x, pixels_y)))
        img_label = Label(window, image=img)
        img_label.image = img
        img_label.place(x=0, y=230)

    def place_recording_clearing_state(r_or_c):
        if r_or_c:
            path = 'Images for GUI/recording state.png'
        else:
            path = 'Images for GUI/clearing state.png'
        root_path = os.path.dirname(os.path.abspath("Piano-Preformance-Auto"))
        complete_path = os.path.join(root_path, path)
        zoom = 0.2
        pixels_x, pixels_y = tuple([int(zoom * x) for x in Image.open(complete_path).size])
        img = ImageTk.PhotoImage(Image.open(path).resize((pixels_x, pixels_y)))
        img_label = Label(window, image=img)
        img_label.image = img
        img_label.place(x=450, y=82)
        return img_label

    stream = pyaudio.PyAudio().open(
        rate=44100,
        channels=1,
        format=pyaudio.paInt16,
        output=True,
        frames_per_buffer=256
    )

    def get_sin_oscillator(freq=55, amp=1, sample_rate=44100):
        increment = (2 * math.pi * freq) / sample_rate
        return (math.sin(v) * amp for v in itertools.count(start=0, step=increment))

    def get_samples(notes_dict, num_samples=70):
        return [sum([int(next(osc) * 32767) \
                     for _, osc in notes_dict.items()]) \
                for _ in range(num_samples)]

    # GUI - Build Frame
    window = Tk()
    window.title("Trial")
    window.attributes("-fullscreen", True)
    window.resizable(width=FALSE, height=FALSE)
    # window['bg']='#33ABFF'
    fontStyle = tkFont.Font(family="Calibri", size=26)
    myLabel1 = Label(window, text="To start/reset recording press RECORD"
                                  "\nTo save recording and end trial press STOP", font=fontStyle)
    myLabel1.place(x=0, y=0)
    place_stop_button()
    place_record_button()
    place_clear_button()
    place_note_chart(chart_path)
    # Initialize MIDI
    pygame.init()
    pygame.midi.init()
    # print("There are " + str(pygame.midi.get_count()) + " MIDI devices")
    input_id = pygame.midi.get_default_input_id()
    if input_id == -1:
        print("Please connect a MIDI device and try again")
        exit(0)
    keyboard = pygame.midi.Input(input_id)
    pygame.display.set_caption("midi test")
    print("starting")
    going = True
    Started = False
    record_clear_state = Label(window, text="")
    reset_recording = False
    # reads midi events until stopped
    try:
        notes_dict = {}
        while going:
            try:
                window.update()
            except:
                break
            if np.all(Raw_Input[0] == [1, 1, 1, 1]):
                reset_recording = True
                Raw_Input = np.zeros((1, 4))
                Recording_Pushed_Last = True
                record_clear_state.destroy()
                record_clear_state = place_recording_clearing_state(True)
            if np.all(Raw_Input[0] == [2, 2, 2, 2]):
                reset_recording = True
                Raw_Input = np.zeros((1, 4))
                record_clear_state.destroy()
                record_clear_state = place_recording_clearing_state(False)
                Recording_Pushed_Last = False

            if notes_dict:
                samples = get_samples(notes_dict)
                samples = np.int16(samples).tobytes()
                stream.write(samples)

            if pygame.get_init():
                for e in pygame.event.get():
                    if e.type in [QUIT]:
                        going = False
                if pygame.midi.get_init():
                    if keyboard.poll():
                        midi_events = keyboard.read(10)
                        if midi_events[0][0][1] != 1:
                            if not Started:
                                time = midi_events[0][1]
                                Started = True
                            else:
                                if reset_recording:
                                    time = midi_events[0][1]
                                    reset_recording = False
                            edited_midi_event = [midi_events[0][1] - time] + midi_events[0][0][:-1]
                            Raw_Input = np.append(Raw_Input, [edited_midi_event], axis=0)
                            print(midi_events)
                        for event in midi_events:
                            (status, note, vel, _), _ = event
                            if status == 0x80 and note in notes_dict:
                                del notes_dict[note]
                            elif status == 0x90 and note not in notes_dict:
                                freq = midi_to_frequency(note)
                                notes_dict[note] = get_sin_oscillator(freq=freq, amp=vel / 500)
    except KeyboardInterrupt as err:
        print("Stopping...")


"""
reads num_events midi events from the buffer.
Input.read(num_events): return midi_event_list
Reads from the Input buffer and gives back midi events. [[[status,data1,data2,data3],timestamp],
 [[status,data1,data2,data3],timestamp],...]
"""
