import numpy as np
import tensorflow as tf
import sounddevice as sd
from scipy.signal import butter, lfilter
import queue
import threading
import time
import csv

class Microphone:
    """
    Microphone class using webcam microphone + YAMNet TFLite
    Returns True if emergency sound detected
    """

    def __init__(
        self,
        yamnet_model_path="models/yamnet.tflite",
        class_map_path="models/yamnet_class_map.csv",
        sample_rate=16000,
        chunk_duration=0.96,
        detection_threshold=0.3,
        event_hold_time=2.0,  # seconds to keep detection true
    ):
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.detection_threshold = detection_threshold
        self.event_hold_time = event_hold_time

        self.audio_queue = queue.Queue()
        self.running = False

        # Detection state
        self.emergency_detected = False
        self.last_detection_time = 0

        # Load YAMNet
        self.interpreter = tf.lite.Interpreter(model_path=yamnet_model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Load class labels
        self.class_names = self._load_class_map(class_map_path)

        self.target_keywords = [
            "shout",
            "yell",
            "scream",
            "cry",
            "bang",
            "thump",
            "thud",
            "crash",
            "slap",
            "impact",
        ]

        # ---------------------------------------------------
        # Initialize bandpass filter ONCE (optimization)
        # This avoids recomputing filter every audio chunk
        # ---------------------------------------------------
        self._init_bandpass_filter()

    def _load_class_map(self, path):
        class_names = []
        with open(path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                class_names.append(row[2])
        return class_names

    def _init_bandpass_filter(self):
        """
        Precompute the bandpass filter for fall sounds.
        Keeps frequencies typically produced by impacts.
        """
        lowcut = 100
        highcut = 2000
        order = 4

        nyquist = 0.5 * self.sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist

        self.b, self.a = butter(order, [low, high], btype="band")

    def bandpass_filter(self, audio):
        """
        Apply precomputed bandpass filter.
        """
        return lfilter(self.b, self.a, audio)

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        audio = indata[:, 0]
        self.audio_queue.put(audio.copy())

    def start(self):
        self.running = True

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback,
            blocksize=self.chunk_samples,
        )
        self.stream.start()

        self.thread = threading.Thread(target=self._process_audio)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, "stream"):
            self.stream.stop()
            self.stream.close()

    def _process_audio(self):
        while self.running:
            try:
                audio_chunk = self.audio_queue.get(timeout=1)
                detected = self.detect_fall_sound(audio_chunk)

                if detected:
                    self.emergency_detected = True
                    self.last_detection_time = time.time()

            except queue.Empty:
                pass

            # Reset flag after hold time
            if self.emergency_detected:
                if time.time() - self.last_detection_time > self.event_hold_time:
                    self.emergency_detected = False

    def detect_fall_sound(self, audio):
        """
        Returns True if an emergency / fall-related sound is detected.

        Processing pipeline:
        1. Ensure audio length matches YAMNet input requirement
        2. Apply band-pass filter to remove background noise
        3. Check if the audio is loud enough (ignore quiet noise)
        4. Detect sudden impact spikes (typical of falls)
        5. Run YAMNet classification
        6. Look for emergency-related sound labels
        """

        # ---------------------------------------------------
        # YAMNet requires a fixed input size
        # ---------------------------------------------------
        REQUIRED_SAMPLES = 15600  # ~0.975s audio at 16kHz

        audio = audio.astype(np.float32)

        # Pad or trim audio to match model input size
        if len(audio) < REQUIRED_SAMPLES:
            audio = np.pad(audio, (0, REQUIRED_SAMPLES - len(audio)))
        elif len(audio) > REQUIRED_SAMPLES:
            audio = audio[:REQUIRED_SAMPLES]

        # ---------------------------------------------------
        # Step 1: Band-pass filter
        # Removes low rumble (fans, AC) and high-frequency noise
        # Keeps frequencies where impacts usually occur
        # ---------------------------------------------------
        audio = self.bandpass_filter(audio).astype(np.float32)

        # ---------------------------------------------------
        # Step 2: Loudness gate (ignore quiet background noise)
        # ---------------------------------------------------
        rms = np.sqrt(np.mean(audio ** 2))
        LOUDNESS_THRESHOLD = 0.02

        if rms < LOUDNESS_THRESHOLD:
            return False

        # ---------------------------------------------------
        # Step 3: Detect sudden impact peaks
        # Falls usually produce a spike in amplitude
        # ---------------------------------------------------
        peak = np.max(np.abs(audio))
        IMPACT_THRESHOLD = 0.35

        if peak < IMPACT_THRESHOLD:
            return False

        # ---------------------------------------------------
        # Step 4: Run YAMNet inference
        # ---------------------------------------------------
        input_tensor = np.expand_dims(audio, axis=0)

        self.interpreter.set_tensor(
            self.input_details[0]["index"], input_tensor
        )
        self.interpreter.invoke()

        scores = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )[0]

        # Get top predicted classes
        top_indices = np.argsort(scores)[::-1][:5]

        # ---------------------------------------------------
        # Step 5: Check if predicted sound matches emergency types
        # ---------------------------------------------------
        for idx in top_indices:
            score = scores[idx]
            label = self.class_names[idx]

            if score < self.detection_threshold:
                continue

            for keyword in self.target_keywords:
                if keyword in label.lower():
                    print(f"[AUDIO] {label} detected ({score:.2f})")
                    return True

        return False

    def is_emergency_detected(self):
        return self.emergency_detected
