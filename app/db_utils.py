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
import sqlite3
import os
from flask import g, current_app
from datetime import datetime

def get_db_path():
    # Uses the path from config_manager attached to the app context
    if hasattr(current_app, 'config_manager'):
        return current_app.config_manager.get('Database', 'Path', fallback='data/gate_controller.db')
    # Fallback if current_app or config_manager is not available (e.g., script usage)
    from .config_manager import ConfigManager
    config = ConfigManager()
    return config.get('Database', 'Path', fallback='data/gate_controller.db')

def get_db():
    db_path = get_db_path()
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        
    db = getattr(g, '_database', None)
    if db is None:
        # Register converter for DATETIME columns named 'timestamp'
        # Adjust format string if microseconds are not always present or format differs
        def adapt_datetime_iso(val):
            return val.isoformat(" ")
        sqlite3.register_adapter(datetime, adapt_datetime_iso)
        sqlite3.register_converter("timestamp", lambda val: datetime.strptime(val.decode('utf-8'), '%Y-%m-%d %H:%M:%S.%f') if val else None)
        
        g._database = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        g._database.row_factory = sqlite3.Row # Access columns by name
    return g._database

def close_db(e=None):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db_schema(db_conn):
    cursor = db_conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS classes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            is_trigger BOOLEAN DEFAULT TRUE -- New column, defaults to TRUE for existing classes
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            filepath TEXT NOT NULL,
            width INTEGER,
            height INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            class_id INTEGER NOT NULL,
            x_center REAL NOT NULL, -- YOLO format: relative to image width
            y_center REAL NOT NULL, -- YOLO format: relative to image height
            width REAL NOT NULL,    -- YOLO format: relative to image width
            height REAL NOT NULL,   -- YOLO format: relative to image height
            FOREIGN KEY (image_id) REFERENCES images (id) ON DELETE CASCADE,
            FOREIGN KEY (class_id) REFERENCES classes (id)
        )
    ''')
    db_conn.commit()

def init_app_db(app):
    """Initializes the database for the app and ensures schema exists."""
    with app.app_context(): # Ensures g is available
        db_conn = get_db()
        init_db_schema(db_conn)
        # Populate default classes from config if table is empty
        default_classes_str = app.config_manager.get('Training', 'DefaultClasses', fallback='')
        if default_classes_str:
            default_classes = [c.strip() for c in default_classes_str.split(',')]
            cursor = db_conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM classes")
            if cursor.fetchone()[0] == 0:
                for class_name in default_classes:
                    # When inserting default classes, they should also be triggers by default
                    try:
                        # Assuming new classes added this way should be triggers by default
                        cursor.execute("INSERT INTO classes (name, is_trigger) VALUES (?, TRUE)", (class_name,))
                    except sqlite3.IntegrityError:
                        pass # Class already exists
                db_conn.commit()
    app.teardown_appcontext(close_db)


# --- Class Management ---
def add_class(name):
    db = get_db()
    try:
        # New classes added via UI are triggers by default
        cursor = db.execute("INSERT INTO classes (name, is_trigger) VALUES (?, TRUE)", (name,))
        db.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None # Class already exists

def get_class_by_name(name):
    db = get_db()
    cursor = db.execute("SELECT * FROM classes WHERE name = ?", (name,))
    return cursor.fetchone()

def get_class_by_id(class_id):
    db = get_db()
    cursor = db.execute("SELECT * FROM classes WHERE id = ?", (class_id,))
    return cursor.fetchone()

def get_all_classes():
    db = get_db()
    cursor = db.execute("SELECT id, name, is_trigger FROM classes ORDER BY name")
    rows = cursor.fetchall()
    # Convert sqlite3.Row objects to dictionaries and ensure is_trigger is Python boolean
    classes_list = []
    for row in rows:
        classes_list.append({
            'id': row['id'], 'name': row['name'], 'is_trigger': bool(row['is_trigger'])
        })
    return classes_list

def delete_class(class_id):
    db = get_db()
    # Also consider what to do with annotations using this class_id
    # For now, we'll just delete the class. You might want to prevent this
    # if annotations exist or cascade delete them (requires schema change or manual delete).
    db.execute("DELETE FROM annotations WHERE class_id = ?", (class_id,)) # Example: remove related annotations
    db.execute("DELETE FROM classes WHERE id = ?", (class_id,))
    db.commit()

def update_class_trigger_status(class_id, is_trigger):
    db = get_db()
    try:
        cursor = db.execute("UPDATE classes SET is_trigger = ? WHERE id = ?", (is_trigger, class_id))
        db.commit()
        return cursor.rowcount > 0 # Returns True if a row was updated
    except sqlite3.Error as e:
        current_app.logger.error(f"Database error updating trigger status for class_id {class_id}: {e}")
        return False


# --- Image Management ---
def add_image_record(filename, filepath, width, height):
    db = get_db()
    try:
        cursor = db.execute(
            "INSERT INTO images (filename, filepath, width, height, timestamp) VALUES (?, ?, ?, ?, ?)",
            (filename, filepath, width, height, datetime.now())
        )
        db.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError: # Should not happen if filenames are unique
        print(f"Error adding image {filename}, it might already exist.")
        return None


def get_image_by_id(image_id):
    db = get_db()
    cursor = db.execute("SELECT * FROM images WHERE id = ?", (image_id,))
    return cursor.fetchone()

def get_all_images():
    db = get_db()
    cursor = db.execute("SELECT id, filename, filepath, timestamp FROM images ORDER BY timestamp DESC")
    return cursor.fetchall()

def delete_image_record(image_id):
    db = get_db()
    image_record = get_image_by_id(image_id)
    if image_record:
        # First delete associated annotations
        db.execute("DELETE FROM annotations WHERE image_id = ?", (image_id,))
        # Then delete the image record
        db.execute("DELETE FROM images WHERE id = ?", (image_id,))
        db.commit()
        # Physical file deletion should be handled in the route
        return image_record['filepath']
    return None

# --- Annotation Management ---
def add_annotation(image_id, class_id, x_center, y_center, width, height):
    db = get_db()
    cursor = db.execute(
        "INSERT INTO annotations (image_id, class_id, x_center, y_center, width, height) VALUES (?, ?, ?, ?, ?, ?)",
        (image_id, class_id, x_center, y_center, width, height)
    )
    db.commit()
    return cursor.lastrowid

def get_annotations_for_image(image_id):
    db = get_db()
    cursor = db.execute(
        "SELECT a.id, a.class_id, c.name as class_name, a.x_center, a.y_center, a.width, a.height "
        "FROM annotations a JOIN classes c ON a.class_id = c.id "
        "WHERE a.image_id = ?", (image_id,)
    )
    return cursor.fetchall()

def delete_annotations_for_image(image_id):
    db = get_db()
    db.execute("DELETE FROM annotations WHERE image_id = ?", (image_id,))
    db.commit()

def update_annotation(annotation_id, class_id, x_center, y_center, width, height):
    # This function might be needed if you allow editing existing annotations directly
    # For simplicity, the current edit page might clear and add new annotations
    db = get_db()
    db.execute(
        "UPDATE annotations SET class_id = ?, x_center = ?, y_center = ?, width = ?, height = ? WHERE id = ?",
        (class_id, x_center, y_center, width, height, annotation_id)
    )
    db.commit()
