# Date and time : 2020-06-03 13:48:03

[TRIGGERING SETTINGS]
# triggering mode, 0 - random t2, 1 - random t2 + zero modification, 2 - modulo t2, 3 - modulo t2 + zero modification
mode = 0
# count, total number of trigger signals for each of the cameras
count = 8192
# deltat, in microsecond minimum time delay between two triggers
deltat = 30000
# the 'n' parameter
n = 1
# twidth, in microseconds trigger pulse width - must be lower than deltat
twidth = 30
# swidth, in microseconds strobe pulse width - must be lower than deltat
swidth = 80
# sdelay, in microseconds strobe delay, can be negative or positive
sdelay = -20

[CAMERA SETTINGS]
# exposure time in microseconds
exposure = 50
# frame rate in Hz
framerate = 100
# pixel format, 0 for Mono8, 1 for Mono16
pixelformat = 0
# ADC bit depth, 0 for Bit8, 1 for Bit10, 2 for Bit12
adcbitdepth = 0
# image width in pixels
imgwidth = 720
# image height in pixels
imgheight = 540
# x offset in pixels
xoffset = 0
# y offset in pixels
yoffset = 0
# black level clamping, 0 for OFF, 1 for ON
blacklevelclamping = 0
# auto gain, 0 for Off, 1 for Once, 2 for Continuous
autogain = 0
# gain
gain = 0
# enable gamma, 0 for OFF, 1 for ON
gammaenable = 0
# 1 for Trigger ON, 0 for trigger OFF
trigger = 1
# 0 for software, 1 for Line0, 2 for Line1, 3 for Line2, 4 for Line3
triggersource = 1
# camera 1 serial number
cam1serial = 20045478
# camera 2 serial number
cam2serial = 20045476
# camera to reverse: 0 for none, 1 for camera 1, 2 for camera 2 
reversecam = 0
# x for reverse in x direction, y for reverse in y direction
reversedirection = 0

