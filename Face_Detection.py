import face_recognition as fr
import os
import cv2 as cv
import numpy as np
from time import sleep


def get_encoded_faces():
    encoded = {}

    for dir_path, d_names, f_names in os.walk("./faces"):
        for f in f_names:
            if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]  # Load faces in database
                encoded[f.split(".")[0]] = encoding

    return encoded


def unknown_image_encoded(img):
    face = fr.load_image_file("faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding


def classify_face(im):
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv.imread(im, 1)

    face_locations = fr.face_locations(img)
    unknown_face_encodings = fr.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = fr.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = fr.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv.rectangle(img, (left - 20, top - 20), (right + 20, bottom + 20), (255, 0, 0), 2)

            # Draw a label with a name below the face
            cv.rectangle(img, (left - 20, bottom - 15), (right + 20, bottom + 20), (255, 0, 0), cv.FILLED)
            font = cv.FONT_HERSHEY_DUPLEX
            cv.putText(img, name, (left - 20, bottom + 15), font, 1.0, (255, 255, 255), 2)

    # Show the final process

    while True:
        cv.imshow('Detected face(s)', img)
        if cv.waitKey(1) & 0xFF == ord('q'):     # Press Q to exit
            return face_names


# Input picture
print(classify_face("test2.jpg"))