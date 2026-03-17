import cv2


class Camera:
    """
    Handles webcam access and simple human presence detection
    using background subtraction and contour detection.
    """

    def __init__(self):
        # Initialize webcam (device index 0 = default webcam)
        self.cap = cv2.VideoCapture(0)

        # Create a background subtractor model to detect motion
        # history: number of frames used to build background model
        # varThreshold: sensitivity to movement changes
        # detectShadows=False reduces false detections
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=25,
            detectShadows=False
        )

    def get_frame(self):
        """
        Capture and return frame from webcam
        """

        # Capture a frame from the webcam
        ret, frame = self.cap.read()

        # If frame capture failed, return None
        if not ret:
            return None

        # Return the captured frame
        return frame

    def detect_person(self, frame):
        """
        Detect person based on contours
        """
        CONTOUR_THRESHOLD = 5000

        # Resize the frame to reduce computation load
        # Smaller resolution improves performance on Raspberry Pi
        frame_small = cv2.resize(frame, (320, 240))

        # Apply background subtraction to detect moving objects
        fg_mask = self.bg_subtractor.apply(frame_small)

        # Smooth the mask to remove noise
        fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)

        # Convert mask into a binary image
        # Pixels above threshold become white (movement)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Find contours (moving object outlines)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through detected contours
        for c in contours:
            # If contour area is large enough, assume it is a person
            if cv2.contourArea(c) > CONTOUR_THRESHOLD:
                print("[CAMERA] HUMAN DETECTED")
                return True

        # If no large movement detected
        return False
