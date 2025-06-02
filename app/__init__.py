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
from flask import Flask
from flask_login import LoginManager
import os
import threading
from .config_manager import ConfigManager
# Import camera_feed components carefully to avoid circular dependencies if they need 'app'
# from .camera_feed import CameraProcessor # Example

# Global variable for camera thread and processed frame
# This is a simplified approach. For more robust systems, consider queues.
camera_thread = None
camera_processor_instance = None # Will hold the instance of CameraProcessor

# Lock for thread-safe access to the latest frame
frame_lock = threading.Lock()
latest_processed_frame = None # Stores the latest frame as JPEG bytes
detection_stats = {"fps": 0, "device": "CPU", "last_detection_time": None, "gate_status": "N/A"}

db = None # Placeholder for database object if using an ORM like SQLAlchemy

def create_app(config_manager_instance):
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.urandom(24)
    app.config['CONFIG_MANAGER'] = config_manager_instance
    
    # Store config in app context for easier access in blueprints/routes
    app.config_manager = config_manager_instance

    # Setup LoginManager if username/password are set
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'main.login' # Assuming 'main' is the blueprint name

    # User loader for Flask-Login
    @login_manager.user_loader
    def load_user(user_id):
        # In a real app, you'd fetch user from a database
        # For this basic auth, we'll handle it differently or simplify
        from .auth import User
        if user_id == config_manager_instance.get('WebServer', 'Username'):
             return User(user_id)
        return None

    # Initialize and start the camera processing thread
    global camera_processor_instance
    from .camera_feed import CameraProcessor # Import here to avoid circularity
    
    # Pass necessary global vars or use app.extensions for them
    camera_processor_instance = CameraProcessor(
        app_instance=app, # Pass the app instance
        config=config_manager_instance,
        set_latest_frame_callback=lambda frame: set_global_frame(frame),
        update_detection_stats_callback=lambda stats: update_global_stats(stats)
    )
    camera_processor_instance.start()


    from .routes import main_bp
    app.register_blueprint(main_bp)

    # Initialize database utility
    from .db_utils import init_app_db
    init_app_db(app) # This will set up the connection and create tables if not exist

    return app

def set_global_frame(frame_bytes):
    global latest_processed_frame
    with frame_lock:
        latest_processed_frame = frame_bytes

def update_global_stats(stats):
    global detection_stats
    with frame_lock: # Reuse frame_lock or create a separate one for stats
        detection_stats.update(stats)

def get_latest_frame():
    with frame_lock:
        if camera_processor_instance and not camera_processor_instance.is_alive():
            # This import is fine here as it's for a specific check
            from .camera_feed import logger as camera_logger # Use a distinct logger or the main app logger
            camera_logger.warning("get_latest_frame: CameraProcessor thread is no longer alive!")
        return latest_processed_frame

def get_detection_stats():
    with frame_lock:
        return detection_stats.copy()
