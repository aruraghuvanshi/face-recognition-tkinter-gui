from tkinter import Canvas
import tkinter as tk
import pickle
import cv2
import os
import glob
import numpy as np
import time
import face_recognition as fr
from PIL import ImageTk


window = tk.Tk()
window.title("Face_Recogniser")
# window.geometry('1280x720')
# window.configure(background='lightgreen')

canvas = Canvas(width=400, height=400, bg='lightgreen')
canvas.pack(expand='yes', fill='both')
image = ImageTk.PhotoImage(file="bk2.jpg")
canvas.create_image(0, 0, image=image, anchor='nw')

window.attributes('-fullscreen', True)
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)


def videoRecognizer():

    TRAIN_NAMES = ['Aru', 'Supriya', 'Sheet', 'Gauri', 'Mum', 'Dad', 'DubuDubu', 'Papa', 'Mummy',
               'Jassi', 'Macho', 'Rajgopal', 'Tasneem', 'Hanisha', 'Kullu']


    with open('face_encodings.txt', 'rb') as fo:
        face_train = pickle.load(fo)

    face_locations = []
    face_names = []
    process = True

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if process:
            face_locations = fr.face_locations(frame)
            face_encodings = fr.face_encodings(frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = fr.compare_faces(face_train, face_encoding, tolerance=0.6)
                names = 'Unrecognized'
                face_distances = fr.face_distance(face_train, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    names = TRAIN_NAMES[best_match_index]
                face_names.append(names)

        process = not process
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 1
            right *= 1
            bottom *= 1
            left *= 1
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, name, (left + 6, bottom - 12), font, 0.7, (0, 255, 0), 1)

        cv2.imshow('Frame', frame)
        k = cv2.waitKey(30)
        if k == 27:
            break
    cap.release()


allfiles = sorted(glob.glob('test\*.jpg'))
target = allfiles[-1]

lbl = tk.Label(window, text=f"Current Target: {target}", width=27, height=1,
                   fg="white", bg="slateblue", font=('arial', 11, 'bold'))
lbl.place(x=825, y=400)



def imageRecognizer():

    allfiles = sorted(glob.glob('test\*.jpg'))

    IMAGEPATH = f"{allfiles[-1]}"

    im = cv2.imread(target)
    if im.shape[0] > 2000:
        scale = 0.25
    elif 1000 > im.shape[0] >= 2000:
        scale = 0.5
    elif 500 > im.shape[0] >= 1000:
        scale = 0.75
    else:
        scale = 1.0

    TRAIN_NAMES = ['Aru', 'Supriya', 'Sheet', 'Gauri', 'Mum', 'Dad', 'DubuDubu', 'Papa', 'Mummy',
               'Jassi', 'Macho', 'Rajgopal', 'Tasneem', 'Hanisha', 'Kullu']


    with open('face_encodings.txt', 'rb') as fo:
        face_train = pickle.load(fo)

    face_locations = []
    face_names = []

    process = True
    frame = cv2.imread(IMAGEPATH)
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    rgb_small_frame = small_frame[:, :, ::-1]
    frame = cv2.cvtColor(rgb_small_frame, cv2.COLOR_BGR2RGB)

    if process:
        face_locations = fr.face_locations(frame)
        face_encodings = fr.face_encodings(frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = fr.compare_faces(face_train, face_encoding, tolerance=0.6)
            names = 'Unrecognized'
            face_distances = fr.face_distance(face_train, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                names = TRAIN_NAMES[best_match_index]
            face_names.append(names)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 1
        right *= 1
        bottom *= 1
        left *= 1
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 12), font, 0.7, (0, 255, 0), 1)

    cv2.imshow('Frame', frame)
    cv2.waitKey(0)


# title = tk.Label(window, text="FACE-R", bg="royalblue", fg="midnightblue", width=43,
#                    height=1, font=('verdana', 30, 'bold'))
# title.place(x=200, y=2)

message = tk.Label(window, text="REAL TIME FACE RECOGNITION", bg="royalblue", fg="white", width=50,
                   height=3, font=('arial', 30, 'bold'))
authormsg = tk.Label(window, text="Developed by Aru Raghuvanshi", bg="royalblue", fg="white", width=25,
                   height=1, font=('arial', 15, 'bold'))
message.place(x=200, y=50)
authormsg.place(x=810, y=160)

vbutton = tk.Button(window, command=videoRecognizer, text='Webcam Recognizer', fg='white', bg='darkslateblue',
                    width=20, height=3, activebackground='dodgerblue',
                    font=('arial', 15, 'bold'))
vbutton.place(x=425, y=300)

ibutton = tk.Button(window, command=imageRecognizer, text='Image Recognizer', fg='white', bg='darkslateblue',
                    width=20, height=3, activebackground='dodgerblue',
                    font=('arial', 15, 'bold'))
ibutton.place(x=825, y=300)

qbutton= tk.Button(window, text='Exit', command=window.destroy , fg="white" ,bg="royalblue",
                   width=20, height=2, activebackground="Red", font=('arial', 15, ' bold '))
qbutton.place(x=625, y=600)


clabel = tk.Label(window, text="Copyright© Aru Raghuvanshi 27-09-2020 All Rights Reserved",
                  bg="purple", fg="white", width=220,
                  height=1, font=('arial', 8, 'bold'))
clabel.place(x=0, y=845)


window.mainloop()