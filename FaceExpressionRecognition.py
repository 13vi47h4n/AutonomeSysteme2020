import face_recognition
import cv2
import numpy as np
import sys
import time
import itertools

# global variables
fps_constant = 20

# TODO: init expression face recognition and model 


# init camera
if (len(sys.argv) > 1):
    video_input = sys.argv[1]
else:
    print("Kamerainput wählen (Entweder Zahl oder URL)")
    video_input = input()
try:
    video_input = int(video_input)
except:
    pass
video_capture = cv2.VideoCapture(video_input)

# init some variables
frame_number = 0
face_locations = []
face_expressions = []
cropped = 0
start_time_current = time.time()
start_time_old = time.time()
while True:
    # set timers for FPS calculation
    if frame_number % fps_constant == 0:
        start_time_old = start_time_current
        start_time_current = time.time()

    
    ret, frame = video_capture.read()
    # face recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_locations = face_recognition.face_locations(rgb_frame)

    # face expression recognition
    face_expressions = []
    for (top, right, bottom, left) in face_locations:
            # TODO: Magic Face Expression Recognition
            # face_expressions.append(<string>)
            pass
    frame_number += 1

    # graphical output face expression recognition
    for (top, right, bottom, left), face_expression in itertools.zip_longest(face_locations, face_expressions, fillvalue=''):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0,0,255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, face_expression, (left + 6, bottom -6), font, 1, (255, 255, 255), 1)

    # graphical output stats
    fps = fps_constant / (start_time_current - start_time_old)
    stats = "Output FPS: {} | Frame: {}".format(int(fps), frame_number)
    cv2.rectangle(frame, (0,0), (300, 25), (255,0,0), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, stats, (6, 19), font, 0.5, (255, 255, 255), 1)

    # display resulting image
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.imshow('Video', frame)

    # break when 'q' is being pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()