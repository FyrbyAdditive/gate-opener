'''
gate-opener, an app for automatically opening gates with inference
Copyright (C) 2025 Timothy Ellis

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License   
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see
<https://www.gnu.org/licenses/>.
'''
import cv2
import time
import threading
import logging
from .yolo_utils import YOLOProcessor
from .gate_control_interface import custom_open_gate, custom_close_gate # Import custom functions

logger = logging.getLogger(__name__)

class CameraProcessor(threading.Thread):
    def __init__(self, app_instance, config, set_latest_frame_callback, update_detection_stats_callback):
        super().__init__(daemon=True) # Daemon thread exits when main program exits
        self.app_instance = app_instance
        self.config = config
        self.yolo_processor = YOLOProcessor(config) # Initialize YOLO processor
        self.set_latest_frame_callback = set_latest_frame_callback
        self.update_detection_stats_callback = update_detection_stats_callback

        self.cap = None
        self.running = False
        self.frame_width = self.config.getint('Webcam', 'FrameWidth')
        self.frame_height = self.config.getint('Webcam', 'FrameHeight')
        
        self.render_persistent_activation_zone = True # Default to showing the zone
        self.gate_is_open = False
        self.last_detection_in_zone_time = 0
        self.previous_gate_action = "IDLE" # Track previous state to call custom functions only on change
        self.gate_open_duration_config = self.config.getint('Gate', 'OpenDuration', fallback=5)

        self.latest_raw_frame_for_capture = None
        self.raw_frame_lock = threading.Lock() # Lock for accessing the raw frame

        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        # Inference FPS control
        self.max_inference_fps = self.config.getint('YOLO', 'MaxInferenceFPS', fallback=0)
        self.inference_frame_delay = 1.0 / self.max_inference_fps if self.max_inference_fps > 0 else 0

        self.db_trigger_classes = [] # Cache for class names from DB for gate triggering

    def _init_camera(self):
        source_str = self.config.get('Webcam', 'Source')
        try:
            source = int(source_str) # Check if it's a number (camera index)
        except ValueError:
            source = source_str # It's a path or URL

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            logger.error(f"Error: Could not open video source: {source}")
            print(f"Error: Could not open video source: {source}")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        logger.info(f"Camera initialized with source: {source} at {self.frame_width}x{self.frame_height}")
        return True

    def _load_db_trigger_classes(self):
        """Loads class names from the database where is_trigger is true."""
        if not self.app_instance:
            logger.warning("CameraProcessor: App instance not available, cannot load DB classes.")
            return
        with self.app_instance.app_context():
            from .db_utils import get_all_classes as db_get_all_classes # Local import to use within context
            all_classes_from_db = db_get_all_classes() # This now returns list of dicts with 'is_trigger'
            
            self.db_trigger_classes = [cls['name'] for cls in all_classes_from_db if cls['is_trigger']]
            
            # Also update the YOLOProcessor's list for direction tracking
            if self.yolo_processor:
                self.yolo_processor.update_target_classes_for_direction_tracking(self.db_trigger_classes)
            logger.info(f"Loaded {len(self.db_trigger_classes)} TRIGGER classes from DB for activation zone: {self.db_trigger_classes}")

    def run(self):
        self.running = True
        if not self._init_camera():
            self.running = False
            # Update stats to reflect error
            self.update_detection_stats_callback({
                "fps": 0, 
                "device": self.yolo_processor.get_device() if self.yolo_processor else "N/A",
                "status": "Camera Error"
            })
            return

        self._load_db_trigger_classes() # Load classes from DB at thread start

        last_inference_time = time.time() # For FPS limiter
        logger.info(f"Camera processing thread started. Gate open duration after zone clear: {self.gate_open_duration_config}s")

        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to grab frame from camera. Re-initializing...")
                    time.sleep(1) # Wait a bit before retrying
                    if self.cap: # Ensure cap exists before releasing
                        self.cap.release()
                    if not self._init_camera():
                        logger.error("Failed to re-initialize camera. Stopping thread.")
                        self.running = False # Stop the thread if camera can't be re-initialized
                        break
                    continue # Try to read next frame after re-init

                # Resize frame if necessary (though cap.set should handle it)
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                
                # Store a copy of this raw, resized frame for capture for training
                # This is done before any overlays are added for the live feed.
                with self.raw_frame_lock:
                    self.latest_raw_frame_for_capture = frame.copy()
                
                # --- Inference FPS Limiter ---
                if self.inference_frame_delay > 0:
                    current_time_for_fps_limit = time.time()
                    elapsed_since_last_inference = current_time_for_fps_limit - last_inference_time
                    if elapsed_since_last_inference < self.inference_frame_delay:
                        sleep_duration = self.inference_frame_delay - elapsed_since_last_inference
                        time.sleep(sleep_duration)
                    last_inference_time = time.time() # Update after potential sleep

                # Process frame with YOLO
                processed_frame, detections = self.yolo_processor.detect_and_track(
                    frame.copy(), # Send a copy
                    draw_configured_zone=self.render_persistent_activation_zone
                )

                # Calculate FPS
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                if elapsed_time >= 1.0: # Update FPS every second
                    self.fps = self.frame_count / elapsed_time
                    self.frame_count = 0
                    self.start_time = time.time()

                # Encode frame to JPEG
                encode_ret, buffer = cv2.imencode('.jpg', processed_frame)
                if encode_ret:
                    self.set_latest_frame_callback(buffer.tobytes())

                # Gate logic based on activation zone
                current_time = time.time()
                target_object_in_zone_this_frame = False
                
                if detections: # Ensure detections is not None
                    for det in detections:
                        detected_class_name = det.get('class_name')
                        is_in_zone = det.get('is_in_zone', False)
                        
                        if detected_class_name in self.db_trigger_classes and is_in_zone:
                            target_object_in_zone_this_frame = True
                            # logger.debug(f"Target object {detected_class_name} (ID: {det.get('track_id')}) is IN ZONE.")
                            break # One target object in zone is enough for this frame
                
                gate_action = "IDLE" # Default state
                if target_object_in_zone_this_frame:
                    self.gate_is_open = True
                    self.last_detection_in_zone_time = current_time
                    gate_action = "OPEN"
                    if self.previous_gate_action != "OPEN":
                        custom_open_gate() # Call custom function
                    # logger.debug(f"Gate OPEN. Object in zone. Last detection in zone time updated: {self.last_detection_in_zone_time}")
                elif self.gate_is_open: # No target object in zone now, but gate was open
                    if current_time - self.last_detection_in_zone_time >= self.gate_open_duration_config:
                        self.gate_is_open = False
                        gate_action = "IDLE" # Gate closes
                        if self.previous_gate_action != "IDLE":
                            custom_close_gate() # Call custom function
                        # logger.debug(f"Gate CLOSED. Zone clear for {self.gate_open_duration_config}s.")
                    else:
                        gate_action = "OPEN" # Gate remains open during linger period
                        # logger.debug(f"Gate OPEN (linger). Zone clear, but within {self.gate_open_duration_config}s. Time since last in zone: {current_time - self.last_detection_in_zone_time:.2f}s")
                
                # If gate_action is IDLE and was not previously IDLE (e.g. initial state or forced close)
                # and it wasn't handled by the elif self.gate_is_open block
                if gate_action == "IDLE" and self.previous_gate_action != "IDLE" and not self.gate_is_open:
                    custom_close_gate()

                self.previous_gate_action = gate_action # Update previous action
                self.update_detection_stats_callback({
                    "fps": self.fps,
                    "device": self.yolo_processor.get_device(),
                    "last_detection_time": current_time,
                    "gate_status": gate_action, 
                    "detections": detections,
                    "object_in_zone": target_object_in_zone_this_frame # Add this flag
                })
            except Exception as e:
                logger.error(f"Unhandled exception in CameraProcessor loop: {e}", exc_info=True)
                # Optionally, set self.running = False to stop the thread on critical errors,
                # or add a delay and continue if it might be a transient issue.
                time.sleep(1) # Wait a bit before trying the next frame

        if self.cap:
            self.cap.release()
        logger.info("Camera processing thread stopped.")
        self.update_detection_stats_callback({
            "status": "Stopped"
        })

    def stop(self):
        self.running = False
        self.join(timeout=2) # Wait for thread to finish
        logger.info("CameraProcessor stop requested.")

    def capture_current_frame_for_training(self):
        # Return a copy of the latest raw frame processed by the main loop
        with self.raw_frame_lock:
            if self.latest_raw_frame_for_capture is not None:
                return self.latest_raw_frame_for_capture.copy()
        return None # Return None if no frame has been captured yet

    def get_yolo_processor(self): # To access training methods etc.
        return self.yolo_processor
