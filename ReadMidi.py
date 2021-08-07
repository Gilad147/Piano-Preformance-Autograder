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
import numpy as np
from pygame.midi import midi_to_frequency
#from Performance_class import get_score


# Helper Functions
def midi(chart_path):
    Raw_Input = np.zeros((1, 4))

    def input_design(Raw_Input):
        New_Input = np.zeros((1,4))
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
        return New_Input[1:].astype(int)

    def exit_application():
        MsgBox = messagebox.askquestion('End of Trial', 'Thank you for participating \n your score is being calculated \n '
                                           'do you want to keep training?',
                                           icon='warning')
        keyboard.close()
        pygame.midi.quit()
        pygame.quit()
        window.destroy()
        if MsgBox == 'no':
            exit()

    def stop():
        print(input_design(Raw_Input[1:].astype('int')))
        #chart_path = get_score(Raw_Input[1:].astype('int'))
        exit_application()
        midi(chart_path)

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
    #window['bg']='#33ABFF'
    fontStyle = tkFont.Font(family="Calibri", size=17)
    myLabel1 = Label(window, text="Press the button to stop recording", font=fontStyle)
    myLabel1.place(x=50, y=0)
    place_stop_button()
    place_note_chart(chart_path)

    # Initialize MIDI
    pygame.init()
    pygame.midi.init()
    print("There are " + str(pygame.midi.get_count()) + " MIDI devices")
    input_id = pygame.midi.get_default_input_id()
    if input_id == -1:
        print("Please connect a MIDI device and try again")
        exit(0)
    keyboard = pygame.midi.Input(input_id)
    pygame.display.set_caption("midi test")
    print("starting")
    going = True
    FirstNote = False
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
                            if not FirstNote:
                                time = midi_events[0][1]
                                FirstNote = True
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