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
import configparser
import os

class ConfigManager:
    def __init__(self, config_file='config.ini'):
        self.config_file = config_file
        # Configure ConfigParser to handle inline comments starting with ';'
        # This requires a space before the ';' for it to be treated as a comment.
        # Values like "foo;bar" will be preserved. Values like "foo ; bar comment" will have " ; bar comment" stripped.
        self.config = configparser.ConfigParser(inline_comment_prefixes=(';',))
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file {self.config_file} not found.")
        self.config.read(self.config_file)

    def _get_cleaned_value(self, section, key, fallback=None):
        """
        Helper to get a value. Inline comments should be handled by ConfigParser's
        inline_comment_prefixes setting during initialization.
        """
        try:
            # self.config.get() should return the value with inline comments already processed
            # if inline_comment_prefixes was set in the ConfigParser constructor.
            value = self.config.get(section, key)
            return value
        except (configparser.NoSectionError, configparser.NoOptionError):
            if fallback is not None:
                return fallback
            raise

    def get(self, section, key, fallback=None):
        try:
            # Value should be clean due to inline_comment_prefixes
            return self.config.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback

    def getint(self, section, key, fallback=None):
        value_str = self._get_cleaned_value(section, key, fallback=str(fallback) if fallback is not None else None)
        if value_str is None: return fallback # Should be handled by _get_cleaned_value if fallback was for missing key
        try:
            return int(value_str)
        except (ValueError, TypeError): # TypeError if value_str is None and not caught earlier
            if fallback is not None: return fallback
            raise ValueError(f"Invalid literal for int() with base 10: '{value_str}' in section '{section}', key '{key}'")

    def getfloat(self, section, key, fallback=None):
        value_str = self._get_cleaned_value(section, key, fallback=str(fallback) if fallback is not None else None)
        if value_str is None: return fallback
        try:
            return float(value_str)
        except (ValueError, TypeError):
            if fallback is not None: return fallback
            raise ValueError(f"Could not convert string to float: '{value_str}' in section '{section}', key '{key}'")

    def get_boolean(self, section, key, fallback=None):
        value_str = self._get_cleaned_value(section, key, fallback=str(fallback) if fallback is not None else None)
        if value_str is None: return fallback
        
        # configparser.ConfigParser.BOOLEAN_STATES maps 'true'/'false', 'on'/'off', 'yes'/'no', '1'/'0'
        # We can leverage that or implement a simpler one if needed.
        # For robustness, let's use a similar check to configparser's internal logic.
        if value_str.lower() in self.config.BOOLEAN_STATES:
            return self.config.BOOLEAN_STATES[value_str.lower()]
        if fallback is not None:
            return fallback
        raise ValueError(f'Not a boolean: {value_str} in section \'{section}\', key \'{key}\'')

    def get_list(self, section, key, delimiter=',', fallback=None):
        value = self._get_cleaned_value(section, key, fallback=None) # Get cleaned string first
        if value is None:
            return fallback if fallback is not None else []
        return [item.strip() for item in value.split(delimiter)]

    def set_value(self, section, key, value):
        """Sets a value in the specified section and key."""
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, key, str(value)) # Ensure value is stored as string

    def save(self):
        """Saves the current configuration back to the .ini file."""
        with open(self.config_file, 'w') as configfile:
            self.config.write(configfile)
# Example usage (typically instantiated once and passed around or accessed via app.config)
# config = ConfigManager()
# camera_source = config.get('Webcam', 'Source')
