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
import logging

logger = logging.getLogger(__name__)

def custom_open_gate():
    """
    Placeholder function to be filled with custom code for opening the gate.
    This function is called when the system determines the gate should open.
    
    Example: Trigger a GPIO pin, send an API request, etc.
    """
    logger.info("SIMULATING: custom_open_gate() called. Implement your gate opening logic here.")
    # TODO: Add your custom gate opening code here
    pass

def custom_close_gate():
    """
    Placeholder function to be filled with custom code for closing the gate.
    This function is called when the system determines the gate should close.

    Example: Trigger a GPIO pin, send an API request, etc.
    """
    logger.info("SIMULATING: custom_close_gate() called. Implement your gate closing logic here.")
    # TODO: Add your custom gate closing code here
    pass
