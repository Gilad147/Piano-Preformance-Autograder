import os
from functools import reduce
from tkinter import *
from tkinter import messagebox
import tkinter.font as tkFont
from ReadMidi import midi
from tkinter import ttk


def play_GUI():
    # Initialize GUI window
    window = Tk()
    window.title("First Stage")
    window.geometry("365x215+10+10")
    window.resizable(width=FALSE, height=FALSE)
    # window['bg']='#33ABFF'
    fontStyle = tkFont.Font(family="Calibri", size=17)

    def do_nothing():
        pass

    window.protocol('WM_DELETE_WINDOW', do_nothing)

    # first stage fields
    myLabel1 = Label(window, text="Welcome to your piano exercise!", font=fontStyle)
    myLabel_ask = Label(window, text="Please enter your id", font=fontStyle)
    myLabel2 = Label(window, text="ID:", font=fontStyle)
    myLabel1.place(x=0, y=0)
    myLabel_ask.place(x=0, y=30)
    myLabel2.place(x=60, y=72)
    subject_ID = Entry(window, width=15, font=fontStyle)
    subject_ID.place(x=95, y=70)
    save_user_id_arr = []

    def pathto_dict(path):
        dir = {}
        path = path.rstrip(os.sep)
        start = path.rfind(os.sep) + 1
        for path, dirs, files in os.walk(path):
            folders = path[start:].split(os.sep)
            subdir = dict.fromkeys(files)
            parent = reduce(dict.get, folders[:-1], dir)
            parent[folders[-1]] = subdir
        return dir

    def create_encoders(directory_tree):
        encoder_midi = {}
        encoder_chart = {}
        root_path = os.path.dirname(os.path.abspath("Piano-Preformance-Auto"))
        for folder in list(directory_tree['project directory']['songs'].keys()):
            for item in list(directory_tree['project directory']['songs'][folder]):
                item_name = item
                item = str(item)
                if "midi" in item:
                    encoder_midi[item[:-5]] = root_path + "/project directory/songs/" + folder + "/" + item_name
                else:
                    if "mid" in item:
                        encoder_midi[item[:-4]] = root_path + "/project directory/songs/" + folder + "/" + item_name
                if "png" in item:
                    encoder_chart[item[:-4]] = root_path + "/project directory/songs/" + folder + "/" + item_name
        return encoder_midi, encoder_chart

    # functions for second stage settings
    def second_stage_attributes():
        window.title("Song Selection")
        window.geometry("500x150")
        subject_ID.destroy()
        btn.destroy()
        myLabel_ask.destroy()
        myLabel2.destroy()

    def second_stage_labels():
        directory_tree = pathto_dict("project directory")
        first_combo_items = sorted(list(directory_tree['project directory']['songs'].keys()))
        myLabel1.configure(text="Please choose a song using the dropdowns below")
        n = StringVar()
        m = StringVar()
        typeChosen = ttk.Combobox(window, width=27, textvariable=n)
        typeChosen['values'] = first_combo_items
        typeChosen.current(1)
        songChosen = ttk.Combobox(window, width=27, textvariable=m)
        songChosen['values'] = ()
        myLabel1.place(x=0, y=0)
        typeChosen.place(x=30, y=30)
        songChosen.place(x=30, y=70)
        typeChosen.current(1)

        def arrange_items(items):
            arranged = []
            for item in items:
                item_name = item
                item = str(item)
                if "midi" in item:
                    arranged.append([item[:-5]])
                else:
                    if "mid" in item:
                        arranged.append([item[:-4]])
            return arranged

        def check_combo():
            second_combo_items = sorted(list(directory_tree['project directory']['songs'][typeChosen.get()].keys()))
            songChosen['values'] = arrange_items(second_combo_items)
            songChosen.set("")

        ok1 = ttk.Button(window, text="ok", command=check_combo)
        ok1.place(x=320, y=30)

        def find_level_by_songbook(songbook):
            levels_by_dictionary = {'initial exercises': 0, 'initial exercises2': 1, 'initial exercises3': 2,
                                    'hebrew Collection': 4}
            return levels_by_dictionary[songbook]

        def confirm():
            chosen_song = songChosen.get()
            song_level = find_level_by_songbook(typeChosen.get())
            directory_encoder_midi, directory_encoder_chart = create_encoders(directory_tree)
            original_midi = directory_encoder_midi[chosen_song]
            chart_path = directory_encoder_chart[chosen_song]
            if songChosen.get() != "":
                messagebox.showinfo('Attention', 'The trial will now begin')
            window.destroy()
            midi(chart_path, original_midi, save_user_id_arr[0], chosen_song, song_level)

        ok2 = ttk.Button(window, text="Confirm", command=confirm)
        ok2.place(x=320, y=70)

    # second stage main function
    def second_stage():
        second_stage_attributes()
        second_stage_labels()

    def is_id(sub_id):
        for let in sub_id:
            if ord(let) < 48 or ord(let) > 57:
                return False
        return True

    # transition from 1st to 2nd stage
    def clicked():
        sub_id = str(subject_ID.get())
        if len(sub_id) == 9 and is_id(sub_id):
            user_id = subject_ID.get()
            save_user_id_arr.append(user_id)
            messagebox.showinfo('Thank you for your cooperation', 'You are transferred to the trial window \nGood luck')
            second_stage()
        else:
            messagebox.showinfo('ID Error', 'Enter your correct 9 Digit ID')

    btn = Button(window, text="Enter", command=clicked)
    btn.place(x=163, y=120)
    window.mainloop()


if __name__ == '__main__':
    play_GUI()
