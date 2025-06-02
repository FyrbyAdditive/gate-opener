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
import torch
from ultralytics import YOLO
import numpy as np
import logging
import os # Ensure os module is imported

logger = logging.getLogger(__name__)

class YOLOProcessor:
    def __init__(self, config):
        self.config = config
        self.model_path = self.config.get('YOLO', 'ModelPath')
        # Clean the model_path to remove any inline comments
        if ';' in self.model_path:
            self.model_path = self.model_path.split(';', 1)[0].strip()
        self.conf_threshold = self.config.getfloat('YOLO', 'ConfidenceThreshold')
        # Initialize with config, but can be updated dynamically
        self.target_classes_names_for_direction_tracking = self.config.get_list('YOLO', 'TargetClasses') 
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"YOLOv8 using device: {self.device}")
        print(f"YOLOv8 using device: {self.device}")

        try:
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"Attempting to load model from (cleaned path): '{self.model_path}' which resolves to: '{os.path.abspath(self.model_path)}'")
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            # Perform a dummy inference to ensure model is loaded and warm up
            dummy_img = np.zeros((self.config.getint('Webcam', 'FrameHeight'), self.config.getint('Webcam', 'FrameWidth'), 3), dtype=np.uint8)
            self.model(dummy_img, verbose=False) 
            logger.info(f"YOLOv8 model '{self.model_path}' loaded successfully on {self.device}.")
        except Exception as e:
            logger.error(f"Error loading YOLOv8 model: {e}")
            print(f"Error loading YOLOv8 model: {e}")
            self.model = None
        
        # Load activation zone points
        self._parse_activation_zone_points(self.config.get('Gate', 'ActivationZonePoints', fallback=""))

    def get_device(self):
        return self.device

    def update_target_classes_for_direction_tracking(self, class_names_list):
        """Allows updating the list of class names for which direction is tracked."""
        self.target_classes_names_for_direction_tracking = class_names_list
        logger.info(f"YOLOProcessor target classes for activation zone updated to: {class_names_list}")

    def _parse_activation_zone_points(self, zone_points_str):
        """Parses the activation zone points string and updates the internal list."""
        self.activation_zone_points_norm_str = zone_points_str
        self.activation_zone_points_norm = []
        if self.activation_zone_points_norm_str:
            try:
                points_list = self.activation_zone_points_norm_str.split(';')
                for p_str in points_list:
                    coords = p_str.split(',')
                    if len(coords) == 2:
                        self.activation_zone_points_norm.append(
                            (float(coords[0].strip()), float(coords[1].strip()))
                        )
                if len(self.activation_zone_points_norm) < 3:
                    logger.warning("Parsed ActivationZonePoints has less than 3 vertices. Zone detection will be disabled.")
                    self.activation_zone_points_norm = [] # Disable if not enough points
            except ValueError as e:
                logger.error(f"Error parsing ActivationZonePoints string '{zone_points_str}': {e}. Zone detection disabled.")
                self.activation_zone_points_norm = []

    def update_activation_zone(self, zone_points_str):
        self._parse_activation_zone_points(zone_points_str)
        logger.info(f"Activation zone updated. New raw string: '{zone_points_str}'. Parsed points: {self.activation_zone_points_norm}")

    def detect_and_track(self, frame, draw_configured_zone=True):
        if not self.model:
            return frame, []

        frame_height, frame_width = frame.shape[:2]
        results = self.model.track(frame, persist=True, verbose=False, conf=self.conf_threshold, half= (self.device=='cuda') )

        detections_data = [] # Store data for external use

        if results and results[0].boxes is not None and results[0].names:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
            confs = results[0].boxes.conf.cpu().numpy()  # Confidences
            clss = results[0].boxes.cls.cpu().numpy()    # Class IDs
            track_ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else None # Tracking IDs
            class_names_map = results[0].names # dict: {class_id: class_name}

            # Prepare activation zone polygon for current frame dimensions
            activation_zone_abs = []
            if self.activation_zone_points_norm:
                for p_norm in self.activation_zone_points_norm:
                    activation_zone_abs.append(
                        (int(p_norm[0] * frame_width), int(p_norm[1] * frame_height))
                    )

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                conf = confs[i]
                cls_id = int(clss[i])
                class_name = class_names_map.get(cls_id, "Unknown")
                
                track_id = track_ids[i] if track_ids is not None else None

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} ({conf:.2f})"
                if track_id is not None:
                    label += f" ID:{track_id}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                detection_info = {
                    "class_name": class_name,
                    "confidence": conf,
                    "box": (x1, y1, x2, y2),
                    "track_id": track_id,
                    "is_in_zone": False # Default to false
                }

                # Activation zone check for target classes
                if class_name in self.target_classes_names_for_direction_tracking and self.activation_zone_points_norm:
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    point_to_test = (cx, cy)
                    
                    # Check if the center of the object is inside the polygon
                    # cv2.pointPolygonTest returns >0 if inside, <0 if outside, 0 if on the edge
                    if activation_zone_abs: # Ensure zone is defined
                        is_inside = cv2.pointPolygonTest(np.array(activation_zone_abs, dtype=np.int32), point_to_test, False) >= 0
                        detection_info["is_in_zone"] = is_inside
                        if is_inside:
                            cv2.putText(frame, "IN ZONE", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                detections_data.append(detection_info)
        
        # Draw activation zone polygon
        logger.debug(f"Attempting to draw activation zone. Points: {self.activation_zone_points_norm}, Frame WxH: {frame_width}x{frame_height}")
        if draw_configured_zone:
            if self.activation_zone_points_norm and len(self.activation_zone_points_norm) >= 3:
                pts = np.array([[int(p[0]*frame_width), int(p[1]*frame_height)] for p in self.activation_zone_points_norm], np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(frame,[pts],isClosed=True,color=(255,0,255),thickness=2) # Magenta color for zone
                logger.debug("Configured activation zone drawn on stream.")
            else:
                logger.debug("Skipped drawing configured activation zone (no points or <3 points).")

        return frame, detections_data

    def train_model(self, data_yaml_path, epochs=50, imgsz=640):
        if not self.model: # Should not happen if initialized correctly
            logger.error("Model not loaded, cannot start training.")
            return False, "Model not loaded."
        
        try:
            # The base model (e.g., yolov8s.pt) will be used for transfer learning
            # The 'model' instance here is already loaded with self.model_path
            # For training, you typically specify the base model like YOLO('yolov8s.yaml').load('yolov8s.pt')
            # or just YOLO('yolov8s.pt') and then call train.
            # Let's assume we are fine-tuning the loaded self.model_path
            
            training_model = YOLO(self.model_path) # Re-init for training to ensure clean state if needed
            
            logger.info(f"Starting YOLOv8 training with data: {data_yaml_path}, epochs: {epochs}, imgsz: {imgsz}")
            results = training_model.train(
                data=data_yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                project="data/runs/detect", # Saves to data/runs/detect/trainX
                name="gate_controller_custom_training",
                exist_ok=True # Overwrite if previous run with same name exists
            )
            logger.info(f"Training completed. Results saved in: {results.save_dir}")
            # The best model is typically at results.save_dir / 'weights' / 'best.pt'
            best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
            if os.path.exists(best_model_path):
                logger.info(f"Best model saved at: {best_model_path}")
                logger.info(f"Attempting to reload active YOLO model with: {best_model_path}")
                try:
                    new_model = YOLO(best_model_path)
                    new_model.to(self.device)
                    # Perform a dummy inference to ensure model is loaded and warm up
                    dummy_img = np.zeros((self.config.getint('Webcam', 'FrameHeight'), self.config.getint('Webcam', 'FrameWidth'), 3), dtype=np.uint8)
                    new_model(dummy_img, verbose=False)
                    
                    self.model = new_model # Atomically update the model
                    self.model_path = best_model_path # Update the internal model_path attribute
                    logger.info(f"Successfully reloaded active model with {best_model_path} on {self.device}")

                    # Update config file with the new model path
                    try:
                        self.config.set_value('YOLO', 'ModelPath', best_model_path)
                        self.config.save()
                        logger.info(f"Updated config file: YOLO ModelPath set to {best_model_path}")
                    except Exception as e_config:
                        logger.error(f"Failed to update config file with new model path: {e_config}", exc_info=True)
                        # Continue, as model reload itself was successful
                except Exception as e_reload:
                    logger.error(f"Failed to reload the newly trained model: {e_reload}", exc_info=True)
                    return True, f"Training successful (best model: {best_model_path}), but failed to auto-reload model."
                return True, f"Training successful. Best model: {best_model_path}. Model reloaded and config updated."
            else:
                return False, "Training finished, but best model not found."

        except Exception as e:
            logger.error(f"Error during YOLOv8 training: {e}")
            return False, f"Training failed: {str(e)}"

# --- Helper for training data preparation (can be expanded) ---
def prepare_yolo_dataset(db_conn, image_records, class_map, dataset_name, base_datasets_path, raw_images_base_path):
    """
    Prepares dataset in YOLO format.
    - image_records: list of image rows from DB
    - class_map: dict of {class_id: class_name}
    - dataset_name: e.g., 'my_gate_dataset'
    - base_datasets_path: path to 'data/datasets/'
    - raw_images_base_path: path to 'data/images_raw/'
    Returns path to data.yaml
    """
    import yaml
    import shutil

    dataset_root = os.path.join(base_datasets_path, dataset_name)
    images_train_path = os.path.join(dataset_root, "images", "train")
    labels_train_path = os.path.join(dataset_root, "labels", "train")
    # Add /val paths if you implement train/val split
    
    os.makedirs(images_train_path, exist_ok=True)
    os.makedirs(labels_train_path, exist_ok=True)

    # Get all classes from DB for data.yaml
    cursor = db_conn.cursor()
    cursor.execute("SELECT id, name FROM classes ORDER BY id") # YOLO expects class IDs to be 0-indexed in files
    all_db_classes = cursor.fetchall()
    
    # Create a mapping from DB class_id to 0-indexed YOLO class_id
    yolo_class_id_map = {db_cls['id']: i for i, db_cls in enumerate(all_db_classes)}
    yolo_class_names = [db_cls['name'] for db_cls in all_db_classes]


    for img_record in image_records:
        # Copy image
        original_image_path = img_record['filepath'] # This should be the full path
        if not os.path.isabs(original_image_path): # If stored as relative to raw_images_base_path
             original_image_path = os.path.join(raw_images_base_path, os.path.basename(img_record['filename']))

        if not os.path.exists(original_image_path):
            logger.warning(f"Image file not found: {original_image_path}, skipping.")
            continue
        
        img_filename_base = os.path.splitext(os.path.basename(img_record['filename']))[0]
        shutil.copy(original_image_path, os.path.join(images_train_path, os.path.basename(img_record['filename'])))

        # Create label file
        label_file_path = os.path.join(labels_train_path, f"{img_filename_base}.txt")
        
        cursor.execute("SELECT class_id, x_center, y_center, width, height FROM annotations WHERE image_id = ?", (img_record['id'],))
        annotations = cursor.fetchall()
        
        with open(label_file_path, 'w') as f_label:
            for ann in annotations:
                if ann['class_id'] not in yolo_class_id_map:
                    logger.warning(f"Annotation class_id {ann['class_id']} not in yolo_class_id_map. Skipping annotation for image {img_record['filename']}")
                    continue
                yolo_cls_idx = yolo_class_id_map[ann['class_id']]
                f_label.write(f"{yolo_cls_idx} {ann['x_center']} {ann['y_center']} {ann['width']} {ann['height']}\n")

    # Create data.yaml
    data_yaml_content = {
        'path': os.path.abspath(dataset_root), # YOLO needs absolute path or relative to where it's run
        'train': os.path.join("images", "train"),
        'val': os.path.join("images", "train"), # Using train for val for simplicity, create a split for real training
        'names': {i: name for i, name in enumerate(yolo_class_names)}
    }
    data_yaml_path = os.path.join(dataset_root, "data.yaml")
    with open(data_yaml_path, 'w') as f_yaml:
        yaml.dump(data_yaml_content, f_yaml, sort_keys=False)
    
    logger.info(f"YOLO dataset prepared at: {dataset_root}")
    logger.info(f"data.yaml created at: {data_yaml_path}")
    return data_yaml_path
