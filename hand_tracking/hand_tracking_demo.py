"""
Simple hand-tracking demo to control the Tilburg Hand (all except the side movements of the fingers)
with your own hand, through a webcam.

At the beginning, it helps to record a range of finger movements to re-normalize the range of motion of
the fingers. To do so, press 'c' and open and close your hand while moving it around the camera image.
Press 'c' again to stop recording; the Tilburg Hand will immediately start to move.
"""

import sys
import os
import numpy as np
import time
import pynput  # for keyboard presses

import cv2
import mediapipe as mp

from tilburg_hand import TilburgHandMotorInterface, Finger, Unit


is_calibrating = False
calibrated = False
time_calibration_started = 0
range_min = np.asarray([0.0]*18)
range_max = np.asarray([np.pi/2]*18)
calibration = []


def on_press(key):
    global is_calibrating
    global time_calibration_started
    global range_min
    global range_max
    global calibration
    global calibrated

    if hasattr(key, 'char') and key.char == 'c':
        if is_calibrating:
            if time.time()-time_calibration_started > 1.0:
                is_calibrating = False
                print("Ending calibration with [", len(calibration), "] samples...")

                if len(calibration) > 20:
                    calib = np.asarray(calibration)

                    range_min = np.min(calib, axis=0)
                    range_max = np.max(calib, axis=0)
                    calibrated = True

                    print("Calibration successful: ")
                    # print('\t', np.round(range_min, 2))
                    # print('\t', np.round(range_max, 2))
                    # print()

        else:
            # Begin calibration
            print("Beginning calibration...")
            is_calibrating = True
            time_calibration_started = time.time()


def get_handv2_joints_from_landmarks(landmarks):
    # https://google.github.io/mediapipe/images/mobile/hand_landmarks.png

    joints = [0]*18

    def landmark_to_numpy(landmark):
        return np.asarray([landmark.x, landmark.y, landmark.z])

    def get_angle_rad(vec1, vec2):
        return np.arccos(np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2))  # * 180.0 / np.pi

    def main_fingers_joints(landmarks, indices, finger_id):
        keypoints = [landmark_to_numpy(landmarks.landmark[i]) for i in indices]  # tip, dip, pip, mcp
        # landmarks_mcps = [landmark_to_numpy(landmarks.landmark[i]) for i in [5, 9, 13]]  # tip, dip, pip, mcp

        wrist = landmark_to_numpy(landmarks.landmark[0])

        vec1 = keypoints[1]-keypoints[2]
        vec2 = keypoints[0]-keypoints[1]
        dip_angle = get_angle_rad(vec1, vec2)

        vec1 = keypoints[2]-keypoints[3]
        vec2 = keypoints[1]-keypoints[2]
        pip_angle = get_angle_rad(vec1, vec2)

        pip_angle = (pip_angle + dip_angle)/2.0
        dip_angle = pip_angle

        vec1 = keypoints[3]-wrist
        vec2 = keypoints[2]-keypoints[3]
        mcp_angle = get_angle_rad(vec1, vec2)

        # vec1 = keypoints[3]-wrist
        # vec2 = keypoints[2]-keypoints[3]
        # abd_angle = get_angle_rad(vec1, vec2)
        # TODO:
        abd_angle = 0

        return [dip_angle, pip_angle, mcp_angle, abd_angle]

    # Index;  landmarks: 8, 7, 6, 5
    index_joints = main_fingers_joints(landmarks, [8, 7, 6, 5], 0)
    joints[Finger.INDEX_DIP] = index_joints[0]
    joints[Finger.INDEX_PIP] = index_joints[1]
    joints[Finger.INDEX_MCP] = index_joints[2]
    joints[Finger.INDEX_ABD] = index_joints[3]

    # Middle; landmarks: 12, 11, 10, 9
    middle_joints = main_fingers_joints(landmarks, [12, 11, 10, 9], 1)
    joints[Finger.MIDDLE_DIP] = middle_joints[0]
    joints[Finger.MIDDLE_PIP] = middle_joints[1]
    joints[Finger.MIDDLE_MCP] = middle_joints[2]
    joints[Finger.MIDDLE_ABD] = middle_joints[3]

    # Pinky;   landmarks: 20, 19, 18, 17
    ring_joints = main_fingers_joints(landmarks, [20, 19, 18, 17], 2)
    joints[Finger.RING_DIP] = ring_joints[0]
    joints[Finger.RING_PIP] = ring_joints[1]
    joints[Finger.RING_MCP] = ring_joints[2]
    joints[Finger.RING_ABD] = ring_joints[3]

    # Thumb;  landmarks: 4, 3, 2, 1, 0
    thumb_keypoints = [landmark_to_numpy(landmarks.landmark[i]) for i in [4, 3, 2, 1]]
    wrist = landmark_to_numpy(landmarks.landmark[0])

    vec1 = thumb_keypoints[1]-thumb_keypoints[2]
    vec2 = thumb_keypoints[0]-thumb_keypoints[1]
    joints[Finger.THUMB_IP] = get_angle_rad(vec1, vec2)

    vec1 = thumb_keypoints[2]-thumb_keypoints[3]
    vec2 = thumb_keypoints[1]-thumb_keypoints[2]
    joints[Finger.THUMB_MCP] = get_angle_rad(vec1, vec2)

    # vec1 = thumb_keypoints[2]-thumb_keypoints[3]
    # vec2 = thumb_keypoints[1]-thumb_keypoints[2]
    # TODO: thumb mcp abd / rot
    joints[Finger.THUMB_ABD] = 0.0

    vec1 = thumb_keypoints[3]-wrist
    vec2 = thumb_keypoints[2]-thumb_keypoints[3]
    joints[Finger.THUMB_CMC] = get_angle_rad(vec1, vec2)

    return np.asarray(joints)


print("Usage:\n\t", sys.argv[0], " [path-to-folder-with-config]")
print("\t* Config folder should have a config.json file and a calibration.json file, as used/produced by motor_gui.")
print("")
print("The demo needs to be calibrated each time it is opened. To calibrate the joints, show your hand to the camera \
       and press 'c'. Move the hand while opening and closing your fingers, and possibly moving/rotating your hand. \
       The procedure should take just a few seconds. Press 'c' again to finish the acquisition of the calibration \
       trajectory. The tilburg hand will not move during calibration, but it will start mimicking your hand as soon as \
       calibration is finished")

config_path = ''
if len(sys.argv) == 2:
    config_path = sys.argv[1]

config_file = os.path.join(config_path, 'config.json')
calibration_file = os.path.join(config_path, 'calibration.json')

if not os.path.exists(config_file):
    config_file = None
if not os.path.exists(calibration_file):
    calibration_file = None

motors = TilburgHandMotorInterface(config_file=config_file, calibration_file=calibration_file, verbose=False)
motor_ids = list(range(motors.n_motors))

ret = motors.connect()
if ret <= 0:
    print("PROBLEM CONNECTING TO THE MOTORS' BOARD")
    exit()


listener = pynput.keyboard.Listener(
        on_release=on_press)
listener.start()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
      static_image_mode=False,
      model_complexity=1,
      max_num_hands=1,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as hands:
    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                joints = get_handv2_joints_from_landmarks(hand_landmarks)
                joints = np.clip(joints/(np.pi/2.0), 0, 1)

                # control the motors
                if is_calibrating:
                    calibration.append(joints)

                if calibrated and not is_calibrating:
                    for i in range(len(joints)):
                        if (range_max[i]-range_min[i]) >= 0.1:
                            joints[i] = (joints[i]-range_min[i])/(range_max[i]-range_min[i])
                    joints = np.clip(joints, 0, 1)

                # Fixed joints must remain at the default zero position
                for motor_id in [Finger.THUMB_CMC, Finger.INDEX_ABD, Finger.MIDDLE_ABD, Finger.RING_ABD]:
                    zero = motors.motor_calib_zero_pos_ticks[motor_id]
                    min_ = motors.motor_calib_min_range_ticks[motor_id]
                    max_ = motors.motor_calib_max_range_ticks[motor_id]
                    joints[motor_id] = (zero - min_) / (max_ - min_)

                if not is_calibrating:
                    print(joints)
                    motors.set_pos_vector(joints, unit=Unit.NORMALIZED, margin_pct=0.02)
                # end control-the-motors

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            scale_prop = 1.4
            dim = (int(image.shape[1]*scale_prop), int(image.shape[0]*scale_prop))
            resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

            cv2.imshow('MediaPipe Hands', cv2.flip(resized, 1))

            if cv2.waitKey(5) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        motors.disconnect()
        cap.release()
