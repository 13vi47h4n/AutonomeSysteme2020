import face_recognition
import cv2
import numpy as np
import sys
import time
import itertools
from resnet import ResNetModel

# global variables
fps_constant = 10
process_Nth_frame = 1
scale_factor = 2
resize_factor = 1

# initialize face expression recognition
face_exp_rec = ResNetModel(size=int(224/resize_factor))

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
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# init some variables
frame_number = 0
face_locations = []
face_expressions = []
cropped = 0
start_time_current = time.time()
start_time_old = time.time()
while True:
    time_at_start = time.time()
    print("Frame: {}".format(frame_number))
    # set timers for FPS calculation
    if frame_number % fps_constant == 0:
        start_time_old = start_time_current
        start_time_current = time.time()

    ret, frame = video_capture.read()
    # face recognition
    if frame_number % process_Nth_frame == 0:
        small_framme = cv2.resize(frame, (0, 0), fx=1/scale_factor, fy=1/scale_factor)
        rgb_frame = cv2.cvtColor(small_framme, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        time_after_face_rec = time.time()
        print("Time Face Recognition: {:.2f}".format(time_after_face_rec - time_at_start))

        # face expression recognition
        face_expressions = []
        for (top, right, bottom, left) in face_locations:
            # Magic Face Expression Recognition
            face_image = frame[top*scale_factor:bottom*scale_factor, left*scale_factor:right*scale_factor]
            face_image = cv2.cvtColor(small_framme, cv2.COLOR_BGR2RGB)
            face_exp = face_exp_rec.face_expression(face_image)
            face_expressions.append(face_exp[0])
        
        time_after_expr_rec = time.time()
        print("Time Face Expression Recognition: {:.2f}".format(time_after_expr_rec - time_after_face_rec))

    frame_number += 1

    # graphical output face expression recognition
    for (top, right, bottom, left), face_expression in itertools.zip_longest(face_locations, face_expressions, fillvalue=''):
        top *= scale_factor
        right *= scale_factor
        bottom *= scale_factor
        left *= scale_factor
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
    print("Output formatting: {:.2f}".format(time.time() - time_after_expr_rec))

    # display resulting image
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.imshow('Video', frame)

    # break when 'q' is being pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
