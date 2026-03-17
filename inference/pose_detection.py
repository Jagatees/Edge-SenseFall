import tensorflow as tf
import numpy as np
import cv2
import time


class PoseEstimator:
    def __init__(self, model_path='models/movenet_model.tflite'):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.prev_center = None
        self.smoothed_center_y = None
        self.prev_time = time.time()

    def estimate_pose(self, frame):
        """
        Run the MoveNet pose estimation on a frame
        """
        img = cv2.resize(frame, (192, 192))
        img = np.expand_dims(img.astype(np.uint8), axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()

        keypoints = self.interpreter.get_tensor(self.output_details[0]['index'])
        return keypoints

    def detect_fall_pose(self, keypoints, frame_width, frame_height):
        """
        Detect fall using:
        1. Body orientation (lying vs standing)
        2. Downward velocity
        3. Multi-keypoint body center tracking
        """

        # THRESHOLDS FOR ADJUSTING
        CONF_THRESHOLD = 0.2
        FALL_RATIO_THRESHOLD = 0.9
        FALL_VELOCITY_THRESHOLD = 25
        SMOOTHING = 0.7

        # Keypoints
        keypoints = keypoints[0][0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]

        # Ensure keypoints are visible
        if (
                left_shoulder[2] < CONF_THRESHOLD or
                right_shoulder[2] < CONF_THRESHOLD or
                left_hip[2] < CONF_THRESHOLD or
                right_hip[2] < CONF_THRESHOLD
        ):
            return False

        # =============================
        # Body Ratio (High for falls)
        # =============================
        shoulder_width = abs(left_shoulder[1] - right_shoulder[1])
        body_height = abs(
            ((left_shoulder[0] + right_shoulder[0]) / 2) -
            ((left_hip[0] + right_hip[0]) / 2)
        )

        if body_height == 0:
            return False

        body_ratio = shoulder_width / body_height

        # =============================
        # Body Center
        # =============================
        points = [left_shoulder, right_shoulder, left_hip, right_hip]

        center_x = 0
        center_y = 0
        count = 0

        for kp in points:
            y, x, conf = kp
            if conf > CONF_THRESHOLD:
                center_x += x * frame_width
                center_y += y * frame_height
                count += 1

        if count == 0:
            return False

        center_x /= count
        center_y /= count

        # =============================
        # Smooth movement
        # =============================
        if self.smoothed_center_y is None:
            self.smoothed_center_y = center_y
        else:
            self.smoothed_center_y = (
                    SMOOTHING * self.smoothed_center_y +
                    (1 - SMOOTHING) * center_y
            )

        # =============================
        # Velocity Calculation using body center
        # Positive when body moves down, negative when moving up
        # =============================
        velocity = 0
        current_time = time.time()

        if self.prev_center is not None and self.prev_time is not None:
            dt = current_time - self.prev_time
            if dt > 0:
                velocity = (self.smoothed_center_y - self.prev_center[1]) / dt

        self.prev_center = (center_x, self.smoothed_center_y)
        self.prev_time = current_time

        # =============================
        # Debug Prints
        # =============================
        # print("Body ratio:", body_ratio)
        # print("Velocity:", velocity)

        # =============================
        # Fall Decision
        # =============================
        if body_ratio > FALL_RATIO_THRESHOLD and velocity > FALL_VELOCITY_THRESHOLD:
            print("[POSE] POSSIBLE FALL DETECTED")
            return True

        return False

