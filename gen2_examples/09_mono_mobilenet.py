#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np

# Start defining a pipeline
pipeline = dai.Pipeline()

cam_rgb = pipeline.createColorCamera()
cam_rgb.setCamId(0)
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)

# Create output
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)
cam_rgb.preview.link(xout_rgb.input)




# Define a source - mono (grayscale) camera
cam_left = pipeline.createMonoCamera()
cam_left.setCamId(1)
cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

cam_right = pipeline.createMonoCamera()
cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
cam_right.setCamId(2)


depth = pipeline.createStereoDepth()
depth.setConfidenceThreshold(200)
depth.setOutputDepth(False)
depth.setLeftRightCheck(False)
depth.setExtendedDisparity(False)
depth.setSubpixel(False)
depth.setOutputRectified(False)
depth.setMedianFilter(dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF)
# Note: the rectified streams are horizontally mirrored by default

cam_left.out.link(depth.left)
cam_right.out.link(depth.right)


xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
depth.disparity.link(xout_depth.input)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(str((Path(__file__).parent / Path('models/mobilenet-ssd.blob')).resolve().absolute()))
cam_rgb.preview.link(detection_nn.input)

# Create a node to convert the grayscale frame into the nn-acceptable form
# manip = pipeline.createImageManip()
# manip.setResize(300, 300)
# # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
# manip.setFrameType(dai.RawImgFrame.Type.BGR888p)
# cam_rgb.preview.link(manip.inputImage)
# manip.out.link(detection_nn.input)

# # Create outputs
# xout_manip = pipeline.createXLinkOut()
# xout_manip.setStreamName("left")
# manip.out.link(xout_manip.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# Pipeline defined, now the device is assigned and pipeline is started
device = dai.Device(pipeline)
device.startPipeline()

# Output queues will be used to get the grayscale frames and nn data from the outputs defined above
q_left = device.getOutputQueue("left")
q_nn = device.getOutputQueue("nn")
q_rgb_enc = device.getOutputQueue(name="h265", maxSize=8, overwrite=True)
q_depth = device.getOutputQueue(name="depth", maxSize=8, overwrite=True)
q_rgb = device.getOutputQueue(name="rgb", maxSize=4, overwrite=True)

frame = None
bboxes = []


# nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
def frame_norm(frame, bbox):
    return (np.array(bbox) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)

frame_depth = None

while True:
    # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
    in_left = None
    # q_left.tryGet()
    in_nn = q_nn.tryGet()
    in_rgb_enc = None
    # q_rgb_enc.tryGet()
    in_depth = q_depth.tryGet()

    in_rgb = q_rgb.tryGet()  # blocking call, will wait until a new data has arrived

    if in_rgb is not None:
        # data is originally represented as a flat 1D array, it needs to be converted into HxWxC form
        shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
        frame_rgb = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        frame_rgb = np.ascontiguousarray(frame_rgb)
        cv2.imshow("rgb", frame_rgb)


    if in_depth is not None:
        frame_depth = in_depth.getData().reshape((in_depth.getHeight(), in_depth.getWidth())).astype(np.uint8)
        frame_depth = np.ascontiguousarray(frame_depth)
        frame_depth = cv2.applyColorMap(frame_depth, cv2.COLORMAP_JET)

    if in_left is not None:
        # if the grayscale frame data is available, transform the 1D data into a HxWxC frame
        shape = (3, in_left.getHeight(), in_left.getWidth())
        frame = in_left.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

    if in_nn is not None:
        # one detection has 7 numbers, and the last detection is followed by -1 digit, which later is filled with 0
        bboxes = np.array(in_nn.getFirstLayerFp16())
        # take only the results before -1 digit
        bboxes = bboxes[:np.where(bboxes == -1)[0][0]]
        # transform the 1D array into Nx7 matrix
        bboxes = bboxes.reshape((bboxes.size // 7, 7))
        # filter out the results which confidence less than a defined threshold
        bboxes = bboxes[bboxes[:, 2] > 0.5][:, 3:7]

    if frame is not None:
        # if the frame is available, draw bounding boxes on it and show the frame
        for raw_bbox in bboxes:
            bbox = frame_norm(frame, raw_bbox)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        cv2.imshow("left", frame)

    if frame_depth is not None:
        cv2.imshow("depth", frame_depth)

    if cv2.waitKey(1) == ord('q'):
        break
