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
from flask import (
    Blueprint, render_template, Response, request, redirect, url_for, jsonify, current_app, flash
)
from flask import send_from_directory # Import send_from_directory
import os
import time
import json # Import the json module
from datetime import datetime # Import datetime

from .auth import auth_required
from . import get_latest_frame, get_detection_stats, camera_processor_instance
from .db_utils import (
    get_db, add_class, get_all_classes, delete_class, update_class_trigger_status,
    add_image_record, get_all_images, get_image_by_id, delete_image_record,
    add_annotation, get_annotations_for_image, delete_annotations_for_image,
    get_class_by_name, get_class_by_id
)
from .yolo_utils import prepare_yolo_dataset # For training data prep
from .gate_control_interface import custom_open_gate, custom_close_gate # For manual control

import logging # Add logging import
main_bp = Blueprint('main', __name__)
logger = logging.getLogger(__name__) # Get a logger instance


# --- Helper ---
def get_config():
    return current_app.config_manager

# --- Main Routes ---
@main_bp.route('/')
@auth_required
def index():
    return redirect(url_for('main.live_feed'))

@main_bp.route('/live_feed')
@auth_required
def live_feed():
    return render_template('live_feed.html')

def generate_video_feed():
    no_frame_count = 0
    max_no_frame_before_break = 100 # e.g., 10 seconds if sleep is 0.1s
    # Target a maximum streaming FPS to the client, e.g., 15 FPS.
    # This helps prevent overwhelming browsers that struggle with rapid multipart/x-mixed-replace.
    target_stream_fps = 15 
    delay_between_frames = 1.0 / target_stream_fps

    while True:
        try:
            frame_bytes = get_latest_frame()
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                no_frame_count = 0 # Reset counter on successful frame
                time.sleep(delay_between_frames) # Control streaming FPS
            else:
                no_frame_count += 1
                if no_frame_count > max_no_frame_before_break:
                    logger.warning(f"generate_video_feed: No new frame for {max_no_frame_before_break * 0.1:.1f} seconds. Stopping stream.")
                    break # Stop the generator
                # Wait if no frame is available or if camera is initializing
                time.sleep(0.1) 
        except GeneratorExit:
            logger.info("generate_video_feed: Client disconnected.")
            break # Client closed connection
        except Exception as e:
            logger.error(f"Error in generate_video_feed: {e}", exc_info=True)
            # Optionally, yield a placeholder error image or just break/continue
            time.sleep(1) # Wait a bit before trying again or break

@main_bp.route('/video_feed')
@auth_required
def video_feed():
    return Response(generate_video_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@main_bp.route('/detection_stats')
@auth_required
def detection_stats_json():
    stats = get_detection_stats()
    
    def convert_numpy_types(data):
        """Recursively converts numpy types in a dictionary or list to Python natives."""
        if isinstance(data, dict):
            return {k: convert_numpy_types(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [convert_numpy_types(i) for i in data]
        # Check for numpy scalar types (e.g., np.float32, np.int64)
        # and also general Python types that might have an item() method but aren't what we want (like dict_items)
        elif hasattr(data, 'item') and not isinstance(data, (dict, str, bytes)): 
            return data.item()
        return data

    stats = convert_numpy_types(stats) # Apply to the whole stats dict first

    # Format timestamp if it exists
    if stats.get('last_detection_time'):
        stats['last_detection_time_str'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stats['last_detection_time']))

    return jsonify(stats)


@main_bp.route('/manual_gate_control', methods=['POST'])
@auth_required
def manual_gate_control():
    action = request.form.get('action')
    
    if action == 'open':
        custom_open_gate()
        if camera_processor_instance:
            camera_processor_instance.gate_is_open = True
            camera_processor_instance.previous_gate_action = "OPEN" # Reflect manual override
            camera_processor_instance.last_detection_in_zone_time = time.time() # Reset timer as if object just appeared
        flash("Manual gate open command sent.", "success")
        logger.info("Manual gate open command executed.")
    elif action == 'close':
        custom_close_gate()
        if camera_processor_instance:
            camera_processor_instance.gate_is_open = False
            camera_processor_instance.previous_gate_action = "IDLE" # Reflect manual override
        flash("Manual gate close command sent.", "success")
        logger.info("Manual gate close command executed.")
    else:
        flash(f"Unknown manual gate action: '{action}'", "warning")
        logger.warning(f"Unknown manual gate action: '{action}'")

    return redirect(url_for('main.live_feed'))


# --- Training Page Routes ---
@main_bp.route('/training')
@auth_required
def training_page():
    classes = get_all_classes()
    images_from_db = get_all_images()
    
    processed_images = []
    for img_row in images_from_db:
        img_dict = dict(img_row) # Convert sqlite3.Row to a mutable dict
        if isinstance(img_dict['timestamp'], str):
            try:
                img_dict['timestamp'] = datetime.strptime(img_dict['timestamp'], '%Y-%m-%d %H:%M:%S.%f') # Adjust format if needed
            except ValueError: # Fallback for format without microseconds
                img_dict['timestamp'] = datetime.strptime(img_dict['timestamp'], '%Y-%m-%d %H:%M:%S')
        processed_images.append(img_dict)
        
    return render_template('training.html', classes=classes, images=processed_images,
                           raw_images_path_display=get_config().get('Training', 'RawImagesPath'))


@main_bp.route('/training/capture_image', methods=['POST'])
@auth_required
def capture_image():
    if camera_processor_instance:
        frame = camera_processor_instance.capture_current_frame_for_training()
        if frame is not None:
            raw_images_dir = get_config().get('Training', 'RawImagesPath')
            os.makedirs(raw_images_dir, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            filepath = os.path.join(raw_images_dir, filename)
            
            try:
                cv2.imwrite(filepath, frame)
                height, width = frame.shape[:2]
                image_id = add_image_record(filename, filepath, width, height)
                if image_id:
                    flash(f"Image '{filename}' captured and saved (ID: {image_id}).", "success")
                else:
                    flash(f"Image '{filename}' saved, but DB record failed.", "warning")
            except Exception as e:
                flash(f"Error saving image: {e}", "danger")
                print(f"Error saving image: {e}")
        else:
            flash("Failed to capture image from camera.", "danger")
    else:
        flash("Camera processor not available.", "danger")
    return redirect(url_for('main.training_page'))

@main_bp.route('/training/start_training', methods=['POST'])
@auth_required
def start_training():
    if not camera_processor_instance or not camera_processor_instance.get_yolo_processor():
        flash("YOLO processor not available for training.", "danger")
        return redirect(url_for('main.training_page'))

    yolo_trainer = camera_processor_instance.get_yolo_processor()
    
    # 1. Prepare data
    db_conn = get_db()
    all_db_images = get_all_images() # Get all images that might have annotations
    
    # Get class map {id: name}
    classes_rows = get_all_classes()
    class_map = {row['id']: row['name'] for row in classes_rows}

    dataset_name = f"gate_dataset_{time.strftime('%Y%m%d%H%M')}"
    base_datasets_path = get_config().get('Training', 'DatasetsPath')
    raw_images_base_path = get_config().get('Training', 'RawImagesPath') # For resolving relative paths if stored that way

    try:
        data_yaml_path = prepare_yolo_dataset(
            db_conn, all_db_images, class_map, dataset_name, 
            base_datasets_path, raw_images_base_path
        )
    except Exception as e:
        flash(f"Error preparing dataset: {e}", "danger")
        logger.error(f"Dataset preparation error: {e}", exc_info=True)
        return redirect(url_for('main.training_page'))

    # 2. Trigger training
    epochs = int(request.form.get('epochs', 50)) # Get from form or default
    imgsz = int(request.form.get('imgsz', 640))   # Get from form or default

    flash(f"Starting training with dataset: {data_yaml_path}. This may take a while...", "info")
    
    # Run training in a separate thread to avoid blocking the web server
    # For simplicity here, it's synchronous. Consider Celery or threading for long tasks.
    success, message = yolo_trainer.train_model(data_yaml_path, epochs=epochs, imgsz=imgsz)

    if success:
        flash(f"Training finished successfully! {message}", "success")
    else:
        flash(f"Training failed or completed with issues: {message}", "danger")
        
    return redirect(url_for('main.training_page'))


@main_bp.route('/training/add_class', methods=['POST'])
@auth_required
def training_add_class():
    class_name = request.form.get('class_name')
    if class_name:
        if add_class(class_name.strip()):
            flash(f"Class '{class_name}' added.", "success")
        else:
            flash(f"Class '{class_name}' already exists or error adding.", "warning")
    else:
        flash("Class name cannot be empty.", "danger")
    return redirect(url_for('main.training_page'))

@main_bp.route('/training/delete_class/<int:class_id>', methods=['POST'])
@auth_required
def training_delete_class(class_id):
    cls = get_class_by_id(class_id)
    if cls:
        delete_class(class_id)
        flash(f"Class '{cls['name']}' and its annotations deleted.", "success")
    else:
        flash("Class not found.", "danger")
    return redirect(url_for('main.training_page'))

@main_bp.route('/training/update_trigger_status', methods=['POST'])
@auth_required
def training_update_trigger_status():
    data = request.get_json()
    class_id = data.get('class_id')
    is_trigger = data.get('is_trigger')

    if class_id is None or is_trigger is None:
        return jsonify({"success": False, "message": "Missing class_id or is_trigger status."}), 400

    try:
        class_id = int(class_id)
        # is_trigger will be boolean true/false from JSON
    except ValueError:
        return jsonify({"success": False, "message": "Invalid class_id format."}), 400

    if update_class_trigger_status(class_id, bool(is_trigger)):
        # Reload trigger classes in CameraProcessor
        if camera_processor_instance:
            camera_processor_instance._load_db_trigger_classes() # Reload the list in the camera processor
            logger.info(f"Updated trigger status for class ID {class_id} to {is_trigger}. Reloaded trigger classes in camera processor.")
        else:
            logger.warning("camera_processor_instance not available to reload trigger classes.")
        
        return jsonify({"success": True, "message": "Trigger status updated successfully."})
    else:
        return jsonify({"success": False, "message": "Failed to update trigger status in database."}), 500

@main_bp.route('/training/delete_image/<int:image_id>', methods=['POST'])
@auth_required
def training_delete_image(image_id):
    filepath_to_delete = delete_image_record(image_id) # Also deletes annotations via DB cascade or explicit call
    if filepath_to_delete:
        try:
            if os.path.exists(filepath_to_delete):
                os.remove(filepath_to_delete)
            flash(f"Image and its annotations deleted successfully.", "success")
        except OSError as e:
            flash(f"DB record deleted, but error removing image file '{filepath_to_delete}': {e}", "warning")
    else:
        flash("Image not found or error deleting.", "danger")
    return redirect(url_for('main.training_page'))


# --- Edit Image Page Routes ---
@main_bp.route('/training/edit_image/<int:image_id>', methods=['GET', 'POST'])
@auth_required
def edit_image_page(image_id):
    image_record = get_image_by_id(image_id)
    if not image_record:
        flash("Image not found.", "danger")
        return redirect(url_for('main.training_page'))

    if request.method == 'POST':
        annotations_data = request.form.get('annotations') # JSON string from JS
        # Clear existing annotations for this image first
        delete_annotations_for_image(image_id)
        
        try:
            annotations = json.loads(annotations_data)
            for ann in annotations:
                js_class_name = ann.get('className')
                if not js_class_name:
                    logger.warning(f"Annotation for image {image_id} received without a className. Skipping.")
                    flash(f"Annotation received without a class name. Skipping.", "warning")
                    continue
                db_class = get_class_by_name(js_class_name.strip()) # Ensure no leading/trailing spaces
                if not db_class:
                    logger.warning(f"Class '{js_class_name}' not found in database for image {image_id}. Annotation skipped.")
                    flash(f"Error: Class '{js_class_name}' not found in database. Annotation skipped.", "warning")
                    continue

                # Annotations from JS are x, y, w, h (top-left corner, pixel values)
                # Convert to YOLO format: x_center, y_center, width, height (all relative to image size)
                img_w, img_h = image_record['width'], image_record['height']
                
                x, y, w, h = float(ann['x']), float(ann['y']), float(ann['w']), float(ann['h'])

                x_center_rel = (x + w / 2) / img_w
                y_center_rel = (y + h / 2) / img_h
                width_rel = w / img_w
                height_rel = h / img_h
                
                add_annotation(image_id, db_class['id'], x_center_rel, y_center_rel, width_rel, height_rel)
                logger.info(f"Saved annotation for image {image_id} with class '{js_class_name}' (ID: {db_class['id']})")
            flash("Annotations saved successfully.", "success")
        except json.JSONDecodeError:
            flash("Error decoding annotation data.", "danger")
        except Exception as e:
            logger.error(f"Error saving annotations for image {image_id}: {e}", exc_info=True)
            flash(f"Error saving annotations: {str(e)}", "danger")
        return redirect(url_for('main.edit_image_page', image_id=image_id))

    existing_annotations = get_annotations_for_image(image_id)
    # Convert existing annotations from YOLO format (relative) back to absolute pixel values for JS
    # (x_center_rel * img_w) - (width_rel * img_w / 2) = x_abs
    # (y_center_rel * img_h) - (height_rel * img_h / 2) = y_abs
    # (width_rel * img_w) = w_abs
    # (height_rel * img_h) = h_abs
    js_annotations = []
    if image_record['width'] and image_record['height']:
        img_w, img_h = image_record['width'], image_record['height']
        for ann in existing_annotations:
            x_abs = (ann['x_center'] * img_w) - (ann['width'] * img_w / 2)
            y_abs = (ann['y_center'] * img_h) - (ann['height'] * img_h / 2)
            w_abs = ann['width'] * img_w
            h_abs = ann['height'] * img_h
            js_annotations.append({
                "x": round(x_abs), "y": round(y_abs), 
                "w": round(w_abs), "h": round(h_abs),
                "className": ann['class_name'] # Send class name for display/selection
            })

    classes = get_all_classes()
    # Construct the image URL carefully. If filepath is absolute, use it.
    # If relative, it's relative to the app's root or a known static folder.
    # For images in `data/images_raw`, we need a route to serve them or make them accessible.
    # Simplest for now: assume they are accessible via a direct path if the `data` folder is served,
    # or create a dedicated route. Let's create a route to serve these raw images.
    
    # The image path in DB might be absolute or relative to `raw_images_dir`
    image_path_from_db = image_record['filepath']
    if not os.path.isabs(image_path_from_db):
        # This assumes filepath in DB is just the filename, and it's in RawImagesPath
        image_filename = os.path.basename(image_path_from_db)
        image_url = url_for('main.serve_raw_image', filename=image_filename)
    else: # If it's an absolute path, this direct URL won't work unless data is web-accessible
          # For security, it's better to serve via a route.
        image_filename = os.path.basename(image_path_from_db)
        image_url = url_for('main.serve_raw_image', filename=image_filename)


    return render_template('edit_image.html', image=image_record, image_url=image_url,
                           classes=classes, annotations_json=json.dumps(js_annotations))

@main_bp.route('/data/images_raw/<path:filename>')
@auth_required
def serve_raw_image(filename):
    raw_images_dir = get_config().get('Training', 'RawImagesPath')
    # Ensure raw_images_dir is an absolute path for send_from_directory
    return send_from_directory(os.path.abspath(raw_images_dir), filename)


# --- Setup Page Routes ---
@main_bp.route('/setup')
@auth_required
def setup_page():
    config_manager = get_config() # Get your custom ConfigManager instance
    return render_template('setup.html', config_manager=config_manager)

@main_bp.route('/setup/get_zone_points', methods=['GET'])
@auth_required
def get_zone_points():
    config = get_config()
    zone_points_str = config.get('Gate', 'ActivationZonePoints', fallback="")
    return jsonify({"zone_points_str": zone_points_str})

@main_bp.route('/setup/save_zone', methods=['POST'])
@auth_required
def save_zone_points():
    data = request.get_json()
    new_zone_points_str = data.get('points_str', '')

    # Basic validation: can it be parsed into at least 3 points?
    temp_points = []
    if new_zone_points_str:
        try:
            points_list = new_zone_points_str.split(';')
            for p_str in points_list:
                coords = p_str.split(',')
                if len(coords) == 2:
                    temp_points.append((float(coords[0].strip()), float(coords[1].strip())))
        except ValueError:
            return jsonify({"success": False, "message": "Invalid points format."}), 400

    if len(temp_points) < 3:
        # Allow saving an empty string to clear the zone, but not an invalid polygon
        if new_zone_points_str != "": 
            return jsonify({"success": False, "message": "A valid zone must have at least 3 points."}), 400

    config = get_config()
    config.set_value('Gate', 'ActivationZonePoints', new_zone_points_str)
    config.save()

    if camera_processor_instance and camera_processor_instance.get_yolo_processor():
        camera_processor_instance.get_yolo_processor().update_activation_zone(new_zone_points_str)
        flash("Activation zone saved and updated.", "success")
        return jsonify({"success": True, "message": "Activation zone saved."})
    else:
        flash("Activation zone saved to config, but live update failed (camera processor not ready).", "warning")
        return jsonify({"success": False, "message": "Zone saved to config, but live update failed."}), 500

# Note: Clearing the zone is handled by saving an empty string via save_zone_points.
# A dedicated /setup/clear_zone route could be added if more specific logic is needed,
# but for now, saving an empty string achieves the same.

@main_bp.route('/set_stream_zone_visibility', methods=['POST'])
@auth_required
def set_stream_zone_visibility():
    data = request.get_json()
    visible = data.get('visible', True) # Default to true if not specified
    if camera_processor_instance:
        camera_processor_instance.render_persistent_activation_zone = bool(visible)
        logger.info(f"Stream persistent zone visibility set to: {camera_processor_instance.render_persistent_activation_zone}")
        return jsonify({"success": True, "visibility_set_to": camera_processor_instance.render_persistent_activation_zone})
    else:
        logger.warning("Attempted to set stream zone visibility, but camera_processor_instance is not available.")
        return jsonify({"success": False, "message": "Camera processor not available."}), 500


# --- Login Route (if basic auth is not enough or for future use) ---
@main_bp.route('/login', methods=['GET', 'POST'])
def login():
    # This is a placeholder if you want to implement form-based login later
    # For now, basic auth is handled by @auth_required
    if request.method == 'POST':
        # username = request.form['username']
        # password = request.form['password']
        # if check_auth(username, password):
        #     user = User(username)
        #     login_user(user)
        #     return redirect(url_for('main.index'))
        # flash('Invalid credentials')
        pass
    return render_template('login.html') # A simple login form
