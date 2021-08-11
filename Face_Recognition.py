import os
import numpy as np
import face_recognition as fr
import cv2 as cv
from datetime import datetime

path = 'faces'         # database
people = []
class_Names = []
name_List = os.listdir(path)
print(name_List)

for name in name_List:
    known_img = cv.imread(f'{path}/{name}')
    people.append(known_img)
    class_Names.append(os.path.splitext(name)[0])
print(class_Names)


def find_encodings(images):
    encoded = []

    for img in images:
        encoding = fr.face_encodings(img)[0]
        encoded.append(encoding)
    return encoded


def write_attendance(name):
    with open('Appearance.csv', 'r+') as f:         # a CSV file to save the name & time  (one time)
        data_list = f.readlines()
        nameList = []
        for line in data_list:
            entry = line.split(', ')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dt_str = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dt_str}')


encode_known = find_encodings(people)
print('.... Face encoded ....')

webcam_cap = cv.VideoCapture(0)

while True:
    isTrue, frame = webcam_cap.read()
    small_frame = cv.resize(frame, (0, 0), None, 0.25, 0.25)

    facesCurFrame = fr.face_locations(small_frame)
    encodesCurFrame = fr.face_encodings(small_frame, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = fr.compare_faces(encode_known, encodeFace)
        face_dis = fr.face_distance(encode_known, encodeFace)    # output is a list in range(0, 1) lower number higher accuracy
        # print(faceDis)
        matchIndex = np.argmin(face_dis)
        name = 'UNKNOWN'
        if matches[matchIndex]:
            name = class_Names[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv.rectangle(frame, (x1, y2-35), (x2, y2), (255, 0, 0), cv.FILLED)
            cv.putText(frame, name, (x1+6, y2-6), cv.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            write_attendance(name)
        else:
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4          # adjust the rectangle
            cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv.rectangle(frame, (x1, y2 - 35), (x2, y2), (255, 0, 0), cv.FILLED)
            cv.putText(frame, name, (x1 + 6, y2 - 6), cv.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (255, 255, 255), 2)

    cv.imshow('WebCam', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):        # Press Q to exit
        break
