import itertools
import math
import os
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
from datetime import datetime
import bisect


# Helper Functions
def midi(chart_path, original_midi, subject_id, song_name, song_level):
    print("song name: " + song_name, " song level: " + str(song_level))
    print(chart_path)
    print(original_midi)
    Raw_Input = np.zeros((1, 4))

    def input_design(Raw_Input):
        New_Input = np.zeros((1, 4))
        i = 1
        for row in range(len(Raw_Input)):
            if Raw_Input[row][1] == 144:
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

    def determine_grade_feedback(scores, breakpoints=[0.3, 0.5, 0.7, 0.8, 0.9],
                                 grades=['bad', 'ok', 'good', 'great', 'excellent', 'amazing']):
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

    def exit_application(grades, recommendation):
        overall_feedback = determine_overall_feedback(np.average(np.array(grades)))
        pitching_feedback = determine_grade_feedback(grades[3])
        tempo_feedback = determine_grade_feedback(grades[4])
        rhythm_feedback = determine_grade_feedback(grades[0])
        velocity_feedback = determine_grade_feedback(grades[1])
        recommendation_dictionary = {'0': 'play slower', '1': 'play it again', '2': 'play faster',
                                     '3': 'play this easier piece', '4': 'play another piece',
                                     '5': 'play this harder piece'}
        feedback_message = overall_feedback + '\n' \
                           + ' please pay attention to this technicals: ' + '\n' + '\n' \
                           + 'your pitching is ' + pitching_feedback + '\n' \
                           + 'tempo is ' + tempo_feedback + '\n' \
                           + 'rhythm is ' + rhythm_feedback + '\n' \
                           + 'and articulation ' + velocity_feedback + '\n'
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
        if 'ly' in file_name:
            return file_name[:-3]
        if 'png' in file_name:
            return file_name[:-4]
        if 'midi' in file_name:
            return file_name[:-5]

    def next_piece_by_level(level):
        next_level = level
        if level <= 0:
            next_level = 0
        if level >= 3:
            next_level = 3
        directories_by_levels = {0: 'initial exercises', 1: 'initial exercises2', 2: 'initial exercises3',
                                 3: 'hebrew Collection'}
        root_path = os.path.dirname(os.path.abspath('project directory/songs' + '/' + directories_by_levels[next_level]))
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

    def stop():
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
        next_chart_path, next_original_midi, next_song_name, next_song_level = \
            next_action_by_recommendation(recommendation)

        stopping = exit_application(grades, recommendation)
        if stopping:
            for widget in window.winfo_children():
                widget.destroy()
            window.destroy()
        else:
            window.destroy()
            midi(next_chart_path, next_original_midi, subject_id, next_song_name, next_song_level)

    def place_stop_button():
        photo2 = PhotoImage(file='/Users/orpeleg/Desktop/stop.png')
        photo2 = photo2.subsample(5, 5)
        stop_button = Button(window, text='Stop', image=photo2, command=stop)
        stop_button.image = photo2
        stop_button.place(x=0, y=0)

    def place_note_chart(path):
        img = ImageTk.PhotoImage(Image.open(path))
        img_label = Label(image=img)
        img_label.image = img
        img_label.place(x=0, y=60)

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
    window.geometry("850x300+10+10")
    window.resizable(width=FALSE, height=FALSE)
    # window['bg']='#33ABFF'
    fontStyle = tkFont.Font(family="Calibri", size=17)
    myLabel1 = Label(window, text="Press the button to stop recording", font=fontStyle)
    myLabel1.place(x=50, y=0)
    place_stop_button()
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
    try:
        notes_dict = {}
        while going:
            try:
                window.update()
            except:
                break

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
