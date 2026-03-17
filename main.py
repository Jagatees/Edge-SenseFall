import time
from sensors.camera import Camera
from sensors.mmWave import MMWave
from sensors.microphone import Microphone

from inference.pose_detection import PoseEstimator

# radar = MMWave()  # mmWave radar initialization
camera = Camera()  # Initialize camera module
# mic = Microphone()  # Initialize Microphone

pose_model = PoseEstimator()  # Load pose estimation model (MoveNet)

print("SYSTEM STARTED")

try:

    # mic.start()

    while True:

        # Detect human presence using mmWave radar
        # presence, distance, velocity = radar.detect_human()

        # If radar does not detect a person, skip processing
        # if not presence:
        # continue

        # Capture frame from webcam
        frame = camera.get_frame()

        # If frame capture failed, skip this loop iteration
        if frame is None:
            continue

        # Check if a person is visible using motion detection
        person_visible = camera.detect_person(frame)

        # Detect fall pattern from radar movement data
        # radar_fall = radar.detect_fall_pattern(distance, velocity)

        # If no person detected in camera view, skip pose estimation
        if not person_visible:
            continue

        # Get frame dimensions for coordinate scaling
        frame_height, frame_width, _ = frame.shape

        # Run pose estimation model on frame to detect body keypoints
        keypoints = pose_model.estimate_pose(frame)

        # Determine whether the detected pose indicates a fall
        camera_fall = pose_model.detect_fall_pose(keypoints, frame_width, frame_height)

        # Small delay to reduce CPU usage and stabilize processing
        time.sleep(0.2)

except KeyboardInterrupt:
    # Stop mic recording
    print("Stopping mic...")
# mic.stop()

