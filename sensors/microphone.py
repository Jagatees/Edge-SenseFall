import os
import csv
import time
import queue
import threading

import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter
from tflite_runtime.interpreter import Interpreter


class Microphone:
    """
    Microphone sensor using YAMNet TFLite.
    Returns confidence score for weighted fusion instead of only True/False.
    """

    def __init__(
        self,
        yamnet_model_path="models/yamnet.tflite",
        class_map_path="models/yamnet_class_map.csv",
        sample_rate=16000,
        chunk_duration=0.975,
        detection_threshold=0.30,
        event_hold_time=2.0,
        min_alert_peak=0.08,
        loudness_reference=0.02,
    ):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)

        self.detection_threshold = detection_threshold
        self.event_hold_time = event_hold_time
        self.min_alert_peak = min_alert_peak
        self.loudness_reference = loudness_reference

        self.audio_queue = queue.Queue(maxsize=5)
        self.running = False
        self.thread = None
        self.stream = None

        # ---------------------------------------------------
        # Detection / fusion state
        # latest_confidence = current chunk confidence
        # held_confidence   = confidence held briefly for fusion
        # ---------------------------------------------------
        self.latest_confidence = 0.0
        self.held_confidence = 0.0
        self.latest_label = "none"
        self.last_detection_time = 0.0

        self.last_result = {
            "sensor": "microphone",
            "confidence": 0.0,
            "instant_confidence": 0.0,
            "label": "none",
            "detected": False,
            "timestamp": time.time(),
            "rms": 0.0,
            "peak": 0.0,
            "top_predictions": [],
        }

        # ---------------------------------------------------
        # Resolve model paths safely
        # ---------------------------------------------------
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(base_dir)

        self.yamnet_model_path = (
            yamnet_model_path
            if os.path.isabs(yamnet_model_path)
            else os.path.join(project_root, yamnet_model_path)
        )

        self.class_map_path = (
            class_map_path
            if os.path.isabs(class_map_path)
            else os.path.join(project_root, class_map_path)
        )

        if not os.path.exists(self.yamnet_model_path):
            raise FileNotFoundError(f"YAMNet model not found: {self.yamnet_model_path}")

        if not os.path.exists(self.class_map_path):
            raise FileNotFoundError(f"Class map not found: {self.class_map_path}")

        # ---------------------------------------------------
        # Load TFLite interpreter
        # ---------------------------------------------------
        self.interpreter = Interpreter(model_path=self.yamnet_model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # ---------------------------------------------------
        # Load YAMNet labels
        # ---------------------------------------------------
        self.class_names = self._load_class_map(self.class_map_path)

        # ---------------------------------------------------
        # Keywords for fall-risk / alert-like sounds
        # ---------------------------------------------------
        self.fall_keywords = [
            "bang",
            "thump",
            "thud",
            "crash",
            "slap",
            "smack",
            "breaking",
            "glass",
            "explosion",
            "fireworks",
            "yell",
            "shout",
            "scream",
            "cry",
        ]

        # ---------------------------------------------------
        # Keywords for normal household / background sounds
        # ---------------------------------------------------
        self.normal_keywords = [
            "speech",
            "television",
            "tv",
            "clapping",
            "hands",
            "tap",
            "door",
            "music",
            "fan",
            "inside",
            "room",
            "conversation",
            "male speech",
            "female speech",
        ]

        # ---------------------------------------------------
        # Initialize band-pass filter once
        # ---------------------------------------------------
        self._init_bandpass_filter()

    def _load_class_map(self, path):
        """
        Load YAMNet class labels from CSV.
        """
        class_names = []
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                class_names.append(row[2])
        return class_names

    def _init_bandpass_filter(self):
        """
        Precompute band-pass filter.
        Removes low rumble and very high-frequency noise.
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
        Apply precomputed band-pass filter.
        """
        return lfilter(self.b, self.a, audio)

    def audio_callback(self, indata, frames, time_info, status):
        """
        Push microphone audio into queue for background processing.
        """
        if status:
            print(f"[AUDIO STATUS] {status}")

        audio = indata[:, 0]

        try:
            self.audio_queue.put_nowait(audio.copy())
        except queue.Full:
            pass

    def start(self, device=None):
        """
        Start microphone stream and background processing thread.
        """
        if self.running:
            print("[INFO] Microphone already running.")
            return

        self.running = True

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self.audio_callback,
                blocksize=self.chunk_samples,
                device=device,
            )
            self.stream.start()
        except Exception as e:
            self.running = False
            raise RuntimeError(f"Failed to start audio stream: {e}")

        self.thread = threading.Thread(target=self._process_audio, daemon=True)
        self.thread.start()

        print("[INFO] Microphone started.")

    def stop(self):
        """
        Stop microphone stream and processing thread.
        """
        self.running = False

        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"[WARN] Error stopping stream: {e}")
            finally:
                self.stream = None

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1)

        print("[INFO] Microphone stopped.")

    def _process_audio(self):
        """
        Continuously process audio chunks and update the latest fusion result.
        """
        while self.running:
            try:
                audio_chunk = self.audio_queue.get(timeout=1)

                result = self.detect_sound(audio_chunk)

                instant_confidence = result["confidence"]
                label = result["label"]
                rms = result["rms"]
                peak = result["peak"]
                top_predictions = result["top_predictions"]

                self.latest_confidence = instant_confidence
                self.latest_label = label

                # ---------------------------------------------------
                # Hold confidence briefly so fusion won't miss
                # short sound events like impact or shout
                # ---------------------------------------------------
                if instant_confidence >= self.detection_threshold:
                    self.held_confidence = instant_confidence
                    self.last_detection_time = time.time()
                else:
                    if time.time() - self.last_detection_time > self.event_hold_time:
                        self.held_confidence = 0.0

                detected = self.held_confidence >= self.detection_threshold

                self.last_result = {
                    "sensor": "microphone",
                    "confidence": float(self.held_confidence),
                    "instant_confidence": float(self.latest_confidence),
                    "label": label,
                    "detected": detected,
                    "timestamp": time.time(),
                    "rms": float(rms),
                    "peak": float(peak),
                    "top_predictions": top_predictions,
                }

                # ---------------------------------------------------
                # Neat debug output
                # ---------------------------------------------------
                timestamp = time.strftime("%H:%M:%S")
                print(f"\n[MIC] {timestamp}")
                print(
                    f"Conf: {self.held_confidence:.3f} | "
                    f"Instant: {self.latest_confidence:.3f} | "
                    f"RMS: {rms:.4f} | Peak: {peak:.4f}"
                )
                print(f"Status: {label}")
                print("Top predictions:")

                if top_predictions:
                    for pred_label, pred_score in top_predictions:
                        print(f"  - {pred_label}: {pred_score * 100:.2f}%")
                else:
                    print("  - none")

                if detected:
                    print(f"[ALERT] FALL RISK SOUND DETECTED -> {label}")
                else:
                    if label.startswith("normal:"):
                        print(f"[INFO] Likely household/background sound -> {label}")
                    elif label == "too_quiet":
                        print("[INFO] Too quiet / no meaningful sound")
                    else:
                        print("[INFO] No fall-risk sound detected")

            except queue.Empty:
                if time.time() - self.last_detection_time > self.event_hold_time:
                    self.held_confidence = 0.0
                    self.last_result["confidence"] = 0.0
                    self.last_result["detected"] = False

            except Exception as e:
                print(f"[ERROR] Audio processing failed: {e}")

    def detect_sound(self, audio):
        """
        Run sound detection and return a structured result with confidence score.

        Returns:
            {
                "confidence": float,
                "label": str,
                "rms": float,
                "peak": float,
                "top_predictions": list[(label, score)]
            }
        """

        # ---------------------------------------------------
        # Step 0: Prepare audio for YAMNet
        # YAMNet expects ~0.975s (15600 samples @ 16kHz)
        # ---------------------------------------------------
        REQUIRED_SAMPLES = 15600
        audio = np.asarray(audio, dtype=np.float32)

        if len(audio) < REQUIRED_SAMPLES:
            audio = np.pad(audio, (0, REQUIRED_SAMPLES - len(audio)), mode="constant")
        elif len(audio) > REQUIRED_SAMPLES:
            audio = audio[:REQUIRED_SAMPLES]

        # ---------------------------------------------------
        # Step 1: Band-pass filter
        # Removes low rumble and high-frequency noise
        # ---------------------------------------------------
        audio = self.bandpass_filter(audio).astype(np.float32)

        # ---------------------------------------------------
        # Step 2: Extract signal features
        # RMS = loudness
        # Peak = sudden spike / possible impact
        # ---------------------------------------------------
        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))

        # ---------------------------------------------------
        # Step 3: Ignore very quiet audio
        # ---------------------------------------------------
        if rms < 0.003:
            return {
                "confidence": 0.0,
                "label": "too_quiet",
                "rms": rms,
                "peak": peak,
                "top_predictions": [],
            }

        # ---------------------------------------------------
        # Step 4: Prepare input tensor for model
        # ---------------------------------------------------
        input_shape = self.input_details[0]["shape"]

        if len(input_shape) == 1:
            input_tensor = audio.astype(np.float32)
        else:
            input_tensor = np.expand_dims(audio, axis=0).astype(np.float32)

        # ---------------------------------------------------
        # Step 5: Run TFLite inference
        # ---------------------------------------------------
        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])

        if output_data.ndim == 1:
            scores = output_data
        else:
            scores = output_data[0]

        # ---------------------------------------------------
        # Step 6: Get top 5 predicted classes
        # ---------------------------------------------------
        top_indices = np.argsort(scores)[::-1][:5]
        top_predictions = []

        fall_score = 0.0
        normal_score = 0.0
        best_fall_label = "none"
        best_normal_label = "none"

        for idx in top_indices:
            score = float(scores[idx])
            label = self.class_names[idx]
            label_lower = label.lower()

            top_predictions.append((label, score))

            # Fall-risk labels
            if any(keyword in label_lower for keyword in self.fall_keywords):
                if score > fall_score:
                    fall_score = score
                    best_fall_label = label

            # Normal household labels
            if any(keyword in label_lower for keyword in self.normal_keywords):
                if score > normal_score:
                    normal_score = score
                    best_normal_label = label

        # ---------------------------------------------------
        # Step 7: Convert audio features into 0..1 scores
        # These help fusion by considering both:
        # - semantic class score
        # - loudness
        # - impact strength
        # ---------------------------------------------------
        loudness_score = min(rms / self.loudness_reference, 1.0) if self.loudness_reference > 0 else 0.0
        impact_score = min(peak / self.min_alert_peak, 1.0) if self.min_alert_peak > 0 else 0.0

        # ---------------------------------------------------
        # Step 8: Build final confidence score
        #
        # fall_confidence:
        # mostly based on fall-risk label score,
        # with some contribution from loudness and peak
        #
        # normal_confidence:
        # used mainly for labeling, not alerting
        # ---------------------------------------------------
        fall_confidence = (
            0.70 * fall_score +
            0.15 * loudness_score +
            0.15 * impact_score
        )
        fall_confidence = float(np.clip(fall_confidence, 0.0, 1.0))

        normal_confidence = (
            0.85 * normal_score +
            0.15 * loudness_score
        )
        normal_confidence = float(np.clip(normal_confidence, 0.0, 1.0))

        # ---------------------------------------------------
        # Step 9: Decide label
        # ---------------------------------------------------
        if fall_confidence >= self.detection_threshold:
            status_label = f"fall_risk: {best_fall_label}"
            confidence = fall_confidence
        elif normal_confidence >= 0.10:
            status_label = f"normal: {best_normal_label}"
            confidence = 0.0
        else:
            status_label = "uncertain"
            confidence = 0.0

        return {
            "confidence": confidence,
            "label": status_label,
            "rms": rms,
            "peak": peak,
            "top_predictions": top_predictions,
        }

    def get_confidence(self):
        """
        Return current instant confidence score.
        """
        return float(self.latest_confidence)

    def get_held_confidence(self):
        """
        Return held confidence score for fusion.
        """
        return float(self.held_confidence)

    def is_emergency_detected(self):
        """
        Convenience boolean. Fusion should preferably use confidence instead.
        """
        return self.held_confidence >= self.detection_threshold

    def get_detection_result(self):
        """
        Standard output format for weighted fusion.
        """
        return {
            "sensor": self.last_result["sensor"],
            "confidence": float(self.last_result["confidence"]),
            "instant_confidence": float(self.last_result["instant_confidence"]),
            "label": self.last_result["label"],
            "detected": bool(self.last_result["detected"]),
            "timestamp": float(self.last_result["timestamp"]),
            "rms": float(self.last_result["rms"]),
            "peak": float(self.last_result["peak"]),
            "top_predictions": self.last_result["top_predictions"],
        }


if __name__ == "__main__":
    mic = None

    try:
        mic = Microphone()

        print("[INFO] Starting microphone test...")
        print("[INFO] Speak, clap, shout, or make impact sounds to test.")
        print("[INFO] Press Ctrl+C to stop.")

        mic.start()

        while True:
            result = mic.get_detection_result()

            print(
                f"\n[FUSION-READY] "
                f"sensor={result['sensor']} | "
                f"confidence={result['confidence']:.3f} | "
                f"instant={result['instant_confidence']:.3f} | "
                f"detected={result['detected']} | "
                f"label={result['label']}"
            )

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
    finally:
        if mic is not None:
            mic.stop()
