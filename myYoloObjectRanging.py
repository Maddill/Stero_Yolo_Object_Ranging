import sys
import cv2
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt

# Function for stereo vision and depth estimation
import triangulation as tri
import calibration

# Mediapipe for face detection
import mediapipe as mp
import time
from track import *

mp_draw = mp.solutions.drawing_utils


# Open both cameras
cap_right = cv2.VideoCapture(2)
cap_left = cv2.VideoCapture(1)

parser = argparse.ArgumentParser()
parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
parser.add_argument('--source1', type=str, default='1', help='source')  # file/folder, 0 for webcam
parser.add_argument('--source2', type=str, default='2', help='source')  # file/folder, 0 for webcam
parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
# class 0 is person, 1 is bycicle, 2 is car... 79 is oven
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--evaluate', action='store_true', help='augmented inference')
parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
parser.add_argument('--visualize', action='store_true', help='visualize features')
parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
opt = parser.parse_args()
opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

# Stereo vision setup parameters
frame_rate = 60  # Camera frame rate (maximum at 120 fps)
B = 10.5  # Distance between the cameras [cm]
f = 1  # Camera lense's focal length [mm]
alpha = 70.42  # Camera field of view in the horisontal plane [degrees]


# Main program loop with face detector and depth estimation using stereo vision
with detect as object_detection:

    while(cap_right.isOpened() and cap_left.isOpened()):

        succes_right, frame_right = cap_right.read()
        succes_left, frame_left = cap_left.read()

    ################## CALIBRATION #########################################################

        frame_right, frame_left = calibration.undistortRectify(
            frame_right, frame_left)

    ########################################################################################

        # If cannot catch any frame, break
        if not succes_right or not succes_left:
            break

        else:

            start = time.time()

            # Convert the BGR image to RGB
            frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
            frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

            # Process the image and object
            results_right = object_detection.process(frame_right)
            results_left = object_detection.process(frame_left)

            # Convert the RGB image to BGR
            frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
            frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)

            ################## CALCULATING DEPTH #########################################################

            center_right = 0
            center_left = 0

            if results_right.detections:
                for id, detection in enumerate(results_right.detections):
                    mp_draw.draw_detection(frame_right, detection)

                    bBox = detection.location_data.relative_bounding_box

                    h, w, c = frame_right.shape

                    boundBox = int(bBox.xmin * w), int(bBox.ymin *h), int(bBox.width * w), int(bBox.height * h)

                    center_point_right = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)

                    cv2.putText(frame_right, f'{int(detection.score[0]*100)}%',
                                (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            if results_left.detections:
                for id, detection in enumerate(results_left.detections):
                    mp_draw.draw_detection(frame_left, detection)

                    bBox = detection.location_data.relative_bounding_box

                    h, w, c = frame_left.shape

                    boundBox = int(bBox.xmin * w), int(bBox.ymin *h), int(bBox.width * w), int(bBox.height * h)

                    center_point_left = (
                        boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)

                    cv2.putText(frame_left, f'{int(detection.score[0]*100)}%', 
                                (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            # If no ball can be caught in one camera show text "TRACKING LOST"
            if not results_right.detections or not results_left.detections:
                cv2.putText(frame_right, "TRACKING LOST", (75, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame_left, "TRACKING LOST", (75, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            else:
                # Function to calculate depth of object. Outputs vector of all depths in case of several balls.
                # All formulas used to find depth is in video presentaion
                depth = tri.find_depth(
                    center_point_right, center_point_left, frame_right, frame_left, B, f, alpha)

                cv2.putText(frame_right, "Distance: " + str(round(depth, 1)),
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(frame_left, "Distance: " + str(round(depth, 1)),
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
                print("Depth: ", str(round(depth, 1)))

            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime
            #print("FPS: ", fps)

            cv2.putText(frame_right, f'FPS: {int(fps)}', (20, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(frame_left, f'FPS: {int(fps)}', (20, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            # Show the frames
            cv2.imshow("frame right", frame_right)
            cv2.imshow("frame left", frame_left)

            # Hit "q" to close the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


# Release and destroy all windows before termination
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()
