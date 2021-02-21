#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
from time import sleep

viewDepth = False
spiOut = True
if spiOut:
    viewDepth = False

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - two mono (grayscale) cameras
left = pipeline.createMonoCamera()
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)

right = pipeline.createMonoCamera()
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
stereo = pipeline.createStereoDepth()
stereo.setConfidenceThreshold(255)
stereo.setOutputDepth(True)

lrcheck  = True   # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled 
subpixel = True   # Better accuracy for longer distance, fractional disparity 32-levels
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 
median   = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7

# Sanitize some incompatible options
if lrcheck or extended or subpixel:
    median   = dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF # TODO

stereo.setMedianFilter(median) # KERNEL_7x7 default
stereo.setLeftRightCheck(lrcheck)
stereo.setExtendedDisparity(extended)
stereo.setSubpixel(subpixel)

left.out.link(stereo.left)
right.out.link(stereo.right)

depthcalculator = pipeline.createDepthCalculator()
config = dai.DepthCalculatorConfig()
config.lower_threshold = 100
config.upper_threshold = 5000
config.roi = dai.Rect(0.4, 0.4, 0.5, 0.5)
depthcalculator.addROI(config)
config.lower_threshold = 200
config.upper_threshold = 5000
config.roi = dai.Rect(0.5, 0.5, 0.7, 0.7)
depthcalculator.addROI(config)

stereo.depth.link(depthcalculator.depthInput)

# Create output
if(viewDepth):
    xout_depth = pipeline.createXLinkOut()
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

if spiOut:
    spiOut_depth_average = pipeline.createSPIOut()
    spiOut_depth_average.setStreamName("spimetaout")
    spiOut_depth_average.setBusId(0)
    depthcalculator.out.link(spiOut_depth_average.input)
else:
    xout_depth_average = pipeline.createXLinkOut()
    xout_depth_average.setStreamName("depth_avg")
    depthcalculator.out.link(xout_depth_average.input)

# Pipeline defined, now the device is assigned and pipeline is started
device = dai.Device(pipeline)
device.startPipeline()

# Output queue will be used to get the depth frames from the outputs defined above
if(viewDepth):
    q = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
if not spiOut:
    q2 = device.getOutputQueue(name="depth_avg", maxSize=4, blocking=False)


while True:
    if spiOut:
        sleep(1)
        continue


    in_depth_avg = q2.get() # blocking call, will wait until a new data has arrived
    
    depth_avg = in_depth_avg.getDepthData()

    for avg in depth_avg:
        roi = avg.config.roi
        average = avg.depth_avg
        print(f"Average depth {average}")

    if(viewDepth):
        in_depth = q.get() # blocking call, will wait until a new data has arrived

        # data is originally represented as a flat 1D array, it needs to be converted into HxW form
        frame = np.array(in_depth.getData()).astype(np.uint8).view(np.uint16).reshape((in_depth.getHeight(), in_depth.getWidth()))

        frame = np.ascontiguousarray(frame)
        color = (255, 255, 255)
        for avg in depth_avg:
            roi = avg.config.roi
            xmin = int(roi.xmin * frame.shape[1])
            ymin = int(roi.ymin * frame.shape[0])
            xmax = int(roi.xmax * frame.shape[1])
            ymax = int(roi.ymax * frame.shape[0])

            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            pt_middle = (xmin + 20, int((ymax+ymin)/2))
            cv2.rectangle(frame, pt1, pt2, color, cv2.FONT_HERSHEY_TRIPLEX)

            average = avg.depth_avg
            cv2.putText(frame, "{:.2f} mm".format(average), pt_middle, cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

        # frame is transformed, the color map will be applied to highlight the depth info
        # frame is ready to be shown
        cv2.imshow("depth", frame)

    if cv2.waitKey(1) == ord('q'):
        break
