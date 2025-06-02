# Gate Opener
## About
A python web application for opening and closing a gate with AI detection of... whatever you want to trigger it. Robot lawnmowers, ride on lawnmowers people, animals, cars and so on. It is also possible to control the gate manually.

You can choose to use the existing categories provided by the model, but also provided is a really simple training interface that allows you to get good results by training it on real things in your yard, so how you trigger it is entirely up to you!

There are two empty functions waiting for you to fill with your own code to trigger relays or whatever to open the gate.

I should say I have not used this yet as the gate I want to use it on is not in great shape, and I have not connected it to any kind of gate opening hardware yet.

I have currently only tested it on an AGX Orin, so I don't really know how it will perform on your platform.

## Installation & Running
```
git clone https://github.com/FyrbyAdditive/gate-opener
cd gate-opener
pip install -r requirements.txt
python run.py
```
## Usage
### Behavior Notes
Please note that if a recognised object is in the activation area, the manual gate controls will be disabled and the gate will open. This is mostly because I didn't really decide how this should otherwise work yet.

### Coding
The functions you need to fill in for however your gate control works are in gate_control_interface.py - it is fairly straightforward. Note that it will not log stuff in those functions (ie the test things there already) unless you change the log level (see run.py near the top).
