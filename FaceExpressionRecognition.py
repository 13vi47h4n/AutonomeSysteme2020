import cv2
import face_recognition
import numpy as np
import sys
import time
import itertools
from resnet import ResNetModel
from TextExport import TextExport
from fast_face_detection import FastFaceRecognition

# global variables
fps_constant = 10
process_Nth_frame = 4
scale_factor = 3
resize_factor = 1

# initialize face expression recognition
face_exp_rec = ResNetModel(size=int(224/resize_factor), mode="gpu")
fast_face_detection = FastFaceRecognition()

# initialize logger
if (len(sys.argv) > 2):
    export = TextExport(sys.argv[2])
else:
    export = TextExport("output.yml")

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
        start_time_current = time.time() + 1

    ret, frame = video_capture.read()
    if not ret:
        print("End of input")
        break

    # face recognition
    if frame_number % process_Nth_frame == 0:
        small_frame = cv2.resize(frame, (0, 0), fx=1/scale_factor, fy=1/scale_factor)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        time_before = time.time()
        face_locations1 = fast_face_detection.face_locations(rgb_frame)
        time_after_fast = time.time()
        face_locations2 = face_recognition.face_locations(rgb_frame)
        time_after_face_rec = time.time()
        print("Time comparison: Fast[{}], Before[{}]".format(time_after_fast - time_before, time_after_face_rec - time_after_fast))
        print("Result fast: {}".format(face_locations1))
        print("Result before: {}".format(face_locations2))

        # face expression recognition
        face_expressions = []
        for (top, left, bottom, right) in face_locations:
            # Magic Face Expression Recognition
            face_image = frame[int(top)*scale_factor:int(bottom) * scale_factor, int(left)*scale_factor:int(right)*scale_factor]
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_exp = face_exp_rec.face_expression(face_image)
            face_expressions.append(face_exp)

        time_after_expr_rec = time.time()
    else:
        cv2.waitKey(33)

    frame_number += 1

    # graphical output face expression recognition
    for (right, top, left, bottom), face_expression in itertools.zip_longest(face_locations, face_expressions, fillvalue=''):
        top *= scale_factor
        right *= scale_factor
        bottom *= scale_factor
        left *= scale_factor
        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), 2)
        cv2.rectangle(frame, (int(left), int(bottom)),
                      (int(right), int(bottom) + 25), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, face_expression, (int(left) + 6, int(bottom) + 18),
                    font, 0.8, (255, 255, 255), 1)

    # graphical output stats
    fps = fps_constant / (start_time_current - start_time_old)
    stats = "Output FPS: {} | Frame: {}".format(int(fps), frame_number)
    cv2.rectangle(frame, (0, 0), (300, 25), (255, 0, 0), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, stats, (6, 19), font, 0.5, (255, 255, 255), 1)

    # display resulting image
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.imshow('Video', frame)

    # log when 'l' is being pressed
    if cv2.waitKey(1) & 0xFF == ord('l'):
        for (top, right, bottom, left), face_expression in itertools.zip_longest(face_locations, face_expressions, fillvalue=''):
            export.append(frame_number, (int(top), int(left)),
                          (int(right), int(bottom)), face_expression)

    # break when 'q' is being pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
