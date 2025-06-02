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
from flask import request, Response, current_app
from functools import wraps
from flask_login import UserMixin

class User(UserMixin):
    def __init__(self, id):
        self.id = id

def check_auth(username, password):
    """This function is called to check if a username /
    password combination is valid."""
    config_username = current_app.config_manager.get('WebServer', 'Username')
    config_password = current_app.config_manager.get('WebServer', 'Password')
    return username == config_username and password == config_password

def authenticate():
    """Sends a 401 response that enables basic auth"""
    return Response(
    'Could not verify your access level for that URL.\n'
    'You have to login with proper credentials', 401,
    {'WWW-Authenticate': 'Basic realm="Login Required"'})

def auth_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Only apply auth if username is set in config
        config_username = current_app.config_manager.get('WebServer', 'Username')
        if not config_username: # No username configured, auth disabled
            return f(*args, **kwargs)

        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

