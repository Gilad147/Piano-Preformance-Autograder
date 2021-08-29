from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
import tkinter.font as tkFont
from ReadMidi import midi


def play_GUI(chart_path):
    # Initialize GUI window
    window = Tk()
    window.title("First Stage")
    window.geometry("365x215+10+10")
    window.resizable(width=FALSE, height=FALSE)
    #window['bg']='#33ABFF'
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

    # functions for second stage settings
    def second_stage_attributes():
        window.title("Second Stage")
        window.geometry("850x300+10+10")
        subject_ID.destroy()
        btn.destroy()

    def second_stage_labels():
        myLabel_ask.destroy()
        myLabel1.configure(text="Attention! You have 1 trial!")
        myLabel2.configure(text="when you confirm that you understand, the notes appear and your trial begins")
        myLabel3 = Label(window, text="Pay attention! when the notes appear the recording begins", font=fontStyle)
        myLabel4 = Label(window, text="When you finish, press 's' key on the keyboard to stop recording", font=fontStyle)
        myLabel5 = Label(window, text="Please confirm that you have read and understood all of the instructions", font=fontStyle)
        myLabel1.place(x=0, y=0)
        myLabel2.place(x=0, y=30)
        myLabel3.place(x=0, y=60)
        myLabel4.place(x=0, y=90)
        myLabel5.place(x=0, y=120)

        def confirm():
            window.destroy()
        confirm_button = Button(window, text='I confirm', command=confirm)
        confirm_button.place(x=570, y=123)

    # second stage main function
    def second_stage():
        second_stage_attributes()
        second_stage_labels()

    def is_id(sub_id):
        for let in sub_id:
            if ord(let) < 48 or ord(let) >57:
                return False
        return True

    # transition from 1st to 2nd stage
    def clicked():
        sub_id = str(subject_ID.get())
        if len(sub_id) == 9 and is_id(sub_id):
            messagebox.showinfo('Thank you for your cooperation', 'You are transferred to the trial window \nGood luck')
            second_stage()
        else:
            messagebox.showinfo('ID Error', 'Enter your correct 9 Digit ID')

    btn = Button(window, text="Enter", command=clicked)
    btn.place(x=163, y=120)
    window.mainloop()
    midi(chart_path)


if __name__ == '__main__':
    chart_path = '/Users/orpeleg/Desktop/91b+lifneishanimrabot'
    play_GUI(chart_path)






#def stop_record_buttons():
 #   #def stop():
  #   #   messagebox.showinfo('End of Trial', 'Thank you for participating')
   #   #  window.destroy()

#    def change_buttons():
 #       record_button.destroy()
  #      photo2 = PhotoImage(file='/Users/orpeleg/Desktop/stop.png')
   #     photo2 = photo2.subsample(3, 3)
    #    stop_button = Button(window, text='Stop', image=photo2, command=stop)
     #   stop_button.image = photo2
      #  stop_button.place(x=50, y=150)
       # #place_note_chart('/Users/orpeleg/Desktop/91b+lifneishanimrabot')

    #def record():
     #   change_buttons()
    #photo = PhotoImage(file='/Users/orpeleg/Desktop/record.png')
    #photo = photo.subsample(3, 3)
    #record_button = Button(window, text='Record', image=photo, command=record)
    #record_button.image = photo
    #record_button.place(x=50, y=150)