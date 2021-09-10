import itertools
import math
import pyaudio
import pygame
import pygame.midi
from pygame.locals import *
from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
import tkinter.font as tkFont
from pygame.midi import midi_to_frequency
from Performance_class import Performance
from auxiliary import np2mid
import pretty_midi
from Files_Feedback import *
from Metronome import Metronome


def midi(chart_path, original_midi, subject_id, song_name, song_level, tempo, lily_path):
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

    def exit_application(grades, recommendation, feedback_path):
        """Handle end of trial and next step determination and set up
            grades: vector of size 4 with the predicted grades from ML processing
            recommendation: a string digit stating the predicted recommendation from ML processing
        """
        feedback_message, verbal_recommendation = feedback_for_exit_application(grades, recommendation, feedback_path)
        MsgBox = messagebox.askquestion('End of Trial', feedback_message + '\n' +
                                        'I advice you to '
                                        + verbal_recommendation + '\n'
                                                                  'do you want to keep training?',
                                        icon='warning')
        keyboard.close()
        pygame.midi.quit()
        pygame.quit()
        if MsgBox == 'no':
            return True

    def stop(recording):
        # process stopping button instance
        if not recording:
            Msg_box_2 = messagebox.askquestion("Warning", "No data was recorded, do you want to exit?")
            if Msg_box_2 == 'no':
                return
            else:
                if tempo != 60:
                    os.remove(original_midi)
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
        midi_path_to_save = directories(Data_Played, subject_id, song_name, tempo)
        np2mid(Data_Played, midi_path_to_save, pretty_midi.PrettyMIDI(original_midi), True)
        performance = Performance(midi_path_to_save, song_name, subject_id, original_midi,
                                  prettyMidiFile_performance=None, prettyMidiFile_original=None)
        tech_grades = performance.get_features()
        if tempo != 60:
            os.remove(original_midi)
        grades = performance.predict_grades(np.asarray(tech_grades).tolist())
        recommendation = performance.predict_reccomendation(np.asarray(tech_grades).tolist())
        feedback_path = midi_path_to_save[:-5] + "-feedback.txt"
        stopping = exit_application(grades, recommendation, feedback_path)
        if stopping:
            for widget in window.winfo_children():
                widget.destroy()
            window.destroy()
        else:
            MsgBox = messagebox.askquestion('Practice continues',
                                            'Do you want to go on with my recommendation?\n\n '
                                            '*choosing "no" means playing the same piece again', icon='warning')
            if MsgBox == 'yes':
                next_chart_path, next_original_midi, next_song_name, next_song_level, next_tempo = \
                    next_action_by_recommendation(recommendation, chart_path, original_midi,
                                                  song_name, song_level, tempo)
            else:
                next_chart_path = chart_path
                next_original_midi = original_midi
                next_song_name = song_name
                next_song_level = song_level
                next_tempo = tempo
            next_lily_path = next_chart_path[:-3] + 'ly'
            window.destroy()
            midi(next_chart_path, next_original_midi, subject_id, next_song_name,
                 next_song_level, next_tempo, next_lily_path)

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
        stop_button.place(x=0, y=120)

    def place_record_button():
        root_path = os.path.dirname(os.path.abspath("Piano-Preformance-Auto"))
        path = 'Images for GUI/record button.png'
        complete_path = os.path.join(root_path, path)
        zoom = 0.4
        pixels_x, pixels_y = tuple([int(zoom * x) for x in Image.open(complete_path).size])
        photo2 = ImageTk.PhotoImage(Image.open(path).resize((pixels_x, pixels_y)))
        record_button = Button(window, text='Record', image=photo2, command=lambda: record(Raw_Input))
        record_button.image = photo2
        record_button.place(x=140, y=120)
        return record_button

    def place_clear_button():
        root_path = os.path.dirname(os.path.abspath("Piano-Preformance-Auto"))
        path = 'Images for GUI/try again button.png'
        complete_path = os.path.join(root_path, path)
        zoom = 0.4
        pixels_x, pixels_y = tuple([int(zoom * x) for x in Image.open(complete_path).size])
        photo2 = ImageTk.PhotoImage(Image.open(path).resize((pixels_x, pixels_y)))
        clear_button = Button(window, text='Clear', image=photo2, command=lambda: clear(Raw_Input))
        clear_button.image = photo2
        clear_button.place(x=280, y=125)
        return clear_button

    def place_note_chart(path):
        zoom = 1.8
        pixels_x, pixels_y = tuple([int(zoom * x) for x in Image.open(path).size])
        img = ImageTk.PhotoImage(Image.open(path).resize((pixels_x, pixels_y)))
        img_label = Label(window, image=img)
        img_label.image = img
        img_label.place(x=0, y=270)

    def place_recording_clearing_state(r_or_c):
        if r_or_c:
            path = 'Images for GUI/recording state.png'
            zoom = 0.2
        else:
            path = 'Images for GUI/hang in there.png'
            zoom = 0.1
        root_path = os.path.dirname(os.path.abspath("Piano-Preformance-Auto"))
        complete_path = os.path.join(root_path, path)
        pixels_x, pixels_y = tuple([int(zoom * x) for x in Image.open(complete_path).size])
        img = ImageTk.PhotoImage(Image.open(path).resize((pixels_x, pixels_y)))
        img_label = Label(window, image=img)
        img_label.image = img
        img_label.place(x=450, y=140)
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
    time_signature = find_time_signature(lily_path)
    beats = ["4/4", "6/8", "2/4", "3/4"]
    metronome = Metronome(window, beats, tempo, time_signature)
    window.resizable(width=FALSE, height=FALSE)
    # window['bg']='#33ABFF'
    fontStyle = tkFont.Font(family="Calibri", size=26)
    myLabel1 = Label(window, text="To start recording press RECORD"
                                  "\nTo save recording and end trial press STOP"
                                  "\nTo delete recording and start over press Try Again", font=fontStyle)
    myLabel1.place(x=0, y=0)
    myLabel2 = Label(window, text="For your convenience you can use a metronome"
                                  "\nThe metronome will guide you to the requested tempo"
                                  "\nthe metronome will stop once you start playing")
    myLabel2.place(x=850, y=40)
    place_stop_button()
    record_or_reset_button = place_record_button()
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
                record_or_reset_button.destroy()
                record_or_reset_button = place_clear_button()
                reset_recording = True
                Raw_Input = np.zeros((1, 4))
                Recording_Pushed_Last = True
                record_clear_state.destroy()
                record_clear_state = place_recording_clearing_state(True)
            if np.all(Raw_Input[0] == [2, 2, 2, 2]):
                record_or_reset_button.destroy()
                record_or_reset_button = place_record_button()
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
                        metronome.stop_counter()
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
