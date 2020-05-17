import cv2
import argparse
import numpy as np
import sys
import time
import itertools
from resnet import ResNetModel
from TextExport import TextExport
from Ultralight.vision.ssd.config.fd_config import define_img_size


parser = argparse.ArgumentParser(
    description='detect_video')

parser.add_argument('--net_type', default="RFB", type=str,
                    help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
parser.add_argument('--input_size', default=480, type=int,
                    help='define network input size,default optional value 128/160/320/480/640/1280')
parser.add_argument('--threshold', default=0.7, type=float,
                    help='score threshold')
parser.add_argument('--candidate_size', default=1000, type=int,
                    help='nms candidate size')
parser.add_argument('--path', default="imgs", type=str,
                    help='imgs dir')
parser.add_argument('--test_device', default="cuda:0", type=str,
                    help='cuda:0 or cpu')
parser.add_argument('--video_path', default="/home/linzai/Videos/video/16_1.MP4", type=str,
                    help='path of video')
args = parser.parse_args()

net_type = 'RFB'

test_device = args.test_device

candidate_size = 1000#args.candidate_size
threshold = 0.7#args.threshold

label_path = "./models/voc-model-labels.txt"

net_type = 'RFB'

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

input_img_size = args.input_size
define_img_size(input_img_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

from Ultralight.vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from Ultralight.vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from Ultralight.vision.utils.misc import Timer

if net_type == 'slim':
    model_path = "models/pretrained/version-slim-320.pth"
    # model_path = "models/pretrained/version-slim-640.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=test_device)
elif net_type == 'RFB':
    model_path = "models/pretrained/version-RFB-320.pth"
    # model_path = "models/pretrained/version-RFB-640.pth"
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)
else:
    print("The net type is wrong!")
    sys.exit(1)
net.load(model_path)

# global variables
fps_constant = 10
process_Nth_frame = 4
scale_factor = 1
resize_factor = 1

# initialize face expression recognition
face_exp_rec = ResNetModel(size=int(224/resize_factor), mode="gpu")

# initialize logger
if (len(sys.argv) > 2):
    export = TextExport(sys.argv[2])
else:
    export = TextExport("output.yml")

# init camera
# if (len(sys.argv) > 1):
#     video_input = sys.argv[1]
# else:
#     print("Kamerainput wählen (Entweder Zahl oder URL)")
#     video_input = input()
# try:
#     video_input = int(video_input)
# except:
#     pass

video_capture = cv2.VideoCapture(0)
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
    if not ret:
        print("End of input")
        break

    # face recognition
    if frame_number % process_Nth_frame == 0:
        small_framme = cv2.resize(
            frame, (0, 0), fx=1/scale_factor, fy=1/scale_factor)
        rgb_frame = cv2.cvtColor(small_framme, cv2.COLOR_BGR2RGB)
        face_locations, labels, probs = predictor.predict(rgb_frame, candidate_size / 2, threshold)
        time_after_face_rec = time.time()
        print("Time Face Recognition: {:.2f}".format(
            time_after_face_rec - time_at_start))

        # face expression recognition
        face_expressions = []
        print(face_locations)
        for (top, left, bottom, right) in face_locations:
            # Magic Face Expression Recognition
            face_image = frame[int(top)*scale_factor:int(bottom) *
                               scale_factor, int(left)*scale_factor:int(right)*scale_factor]
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_exp = face_exp_rec.face_expression(face_image)
            face_expressions.append(face_exp)

        time_after_expr_rec = time.time()
        if len(face_expressions) > 0:
            print("Time Face Expression Recognition: {:.2f}".format(
                time_after_expr_rec - time_after_face_rec))
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
    fps = 1#fps_constant / (start_time_current - start_time_old)
    stats = "Output FPS: {} | Frame: {}".format(int(fps), frame_number)
    cv2.rectangle(frame, (0, 0), (300, 25), (255, 0, 0), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, stats, (6, 19), font, 0.5, (255, 255, 255), 1)
    print("Output formatting: {:.2f}".format(
        time.time() - time_after_expr_rec))

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
