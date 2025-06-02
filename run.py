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
import os
import logging
from app import create_app, db
from app.config_manager import ConfigManager
from app.db_utils import init_app_db

# Basic logging configuration
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    config = ConfigManager()
    app = create_app(config)
    
    # Initialize database
    with app.app_context():
        init_app_db(app)

    # Ensure data directories exist
    os.makedirs(config.get('Training', 'RawImagesPath'), exist_ok=True)
    os.makedirs(config.get('Training', 'DatasetsPath'), exist_ok=True)
    os.makedirs("data/runs", exist_ok=True)

    # Start the camera feed processing thread
    # The thread is started within create_app to have access to app context if needed
    # or can be started here if it's fully independent.
    # For simplicity, let's assume camera_thread.start() is handled after app creation
    # or implicitly if it's designed to start on first access.
    # If camera_thread is an instance of a class with a start method:
    # from app.camera_feed import camera_processor # Assuming camera_processor is the instance
    # camera_processor.start()

    use_https = config.get_boolean('WebServer', 'UseHTTPS')
    cert_path = config.get('WebServer', 'CertPath')
    key_path = config.get('WebServer', 'KeyPath')
    ssl_context = None

    if use_https:
        if os.path.exists(cert_path) and os.path.exists(key_path):
            ssl_context = (cert_path, key_path)
            print(f"Attempting to start HTTPS server with cert: {cert_path}, key: {key_path}")
        else:
            print(f"Warning: HTTPS enabled but certificate or key file not found. Check paths: {cert_path}, {key_path}")
            print("Falling back to HTTP.")
            use_https = False

    print(f"Starting server on {config.get('WebServer', 'Host')}:{config.getint('WebServer', 'Port')}")
    
    # For development, Flask's built-in server is fine.
    # For production, use Gunicorn or Waitress.
    app.run(host=config.get('WebServer', 'Host'),
            port=config.getint('WebServer', 'Port'),
            debug=config.get_boolean('WebServer', 'Debug'),
            ssl_context=ssl_context if use_https else None,
            threaded=True,
            use_reloader=False) # Important for handling background tasks and multiple requests
