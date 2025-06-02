# Gate Opener
## About
A python web application for opening and closing a gate with AI detection of... whatever you want to trigger it. Robot lawnmowers, ride on lawnmowers people, animals, cars and so on. It is also possible to control the gate manually.

You can choose to use the existing categories provided by the model, but also provided is a really simple training interface that allows you to get good results by training it on real things in your yard, so how you trigger it is entirely up to you!

Other useful points:
- Web interface supports HTTPS & HTTP
- Username/password optional
- Supports local USB and various different types of network camera e.g. RTSP and others
- Um, opens gates

There are two empty functions waiting for you to fill with your own code to trigger relays or whatever to open the gate. I have currently only tested it on an AGX Orin, so I don't really know how it will perform on your platform, however using a smaller Yolo V8 will help speed it up.

I have not used this yet as the gate I want to use it on is not in great shape, and I have not connected it to any kind of gate opening hardware yet, hence there may or may not be issues around this.

## How to...
### Install
```
git clone https://github.com/FyrbyAdditive/gate-opener
cd gate-opener
pip install -r requirements.txt
python run.py
```
### Models
If you have performance issues, you may like to try a smaller model. By default, yolov8s.pt is used which is a small sized model. Here are the available models, in order of size:
- yolov8n.pt
- yolov8s.pt
- yolov8m.pt
- yolov8l.pt
- yolov8x.pt
Enter your choice into the relevant bit of config.ini:
```
[YOLO]
; Ensure this model is available
ModelPath = yolov8s.pt
```
### Use


### Behavior Notes
Please note that if a recognised object is in the activation area, the manual gate controls will be disabled and the gate will open. This is mostly because I didn't really decide how this should otherwise work yet.

### Coding

The functions you need to fill in for however your gate control works are in gate_control_interface.py - it is fairly straightforward. Note that it will not log stuff in those functions (ie the test things there already) unless you change the log level (see run.py near the top).
