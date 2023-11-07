# me - this DAT
# scriptOp - the OP which is cooking

import numpy as np
import cv2

# import cv2.aruco as aruco
import mediapipe as mp
import math
import imutils

print("Imutils: ", imutils.__version__)
print("Media Pipe: ", mp.__version__)
print("Open CV: ", cv2.__version__)
print("ARUCO: ", cv2.aruco)
print("ME:", me)


Resolution = (640, 360)


# press 'Setup Parameters' in the OP to call this function to re-create the pa
# rameters.
def onSetupParameters(scriptOp):
    return


# called whenever custom pulse parameter is pushed
def onPulse(par):
    return


def onCook(scriptOp):
    print("[COOK]")
    # grab the input to the scriptTOP with a frame delayed
    # for faster operation (compare TopTo CHOP)
    # rgba values as 0-1
    video_feed = scriptOp.inputs[0].numpyArray(delayed=True)

    if video_feed is None:
        return

    print("[RUN - ARUCO]")
    image = cv2.cvtColor(video_feed, cv2.COLOR_BGR2RGB)

    # Remap the values to the range [0, 255]
    image = np.interp(image, (0, 1), (0, 255))
    # convert data to uint8

    image = np.uint8(image)

    get_aruco_marker_data(image=image)

    export_data = []
    # running multiple objectrons, one for each item type we are looking for

    scriptOp.store("detection", export_data)
    # image = cv2.flip(image, 1)
    image = np.interp(image, (0, 255), (0, 1))
    image = np.float32(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    scriptOp.copyNumpyArray(image)

    return


# https://pyimagesearch.com/2021/01/04/opencv-augmented-reality-ar/
def get_aruco_marker_data(image):
    if image is None:
        return None, None

    (imgH, imgW) = image.shape[:2]
    # image = cv2.flip(image, 1)
    #
    # load the ArUCo dictionary, grab the ArUCo parameters, and detect
    # the markers
    print("[INFO] detecting markers...")
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    # arucoParams = cv2.aruco
    # print(cv2.aruco.DetectorParameters())
    # arucoParams = aruco.DetectorParameters()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict)
    # if we have not found four markers in the input image then we cannot
    # apply our augmented reality technique
    if len(corners) != 4:
        print("[INFO] could not find 4 corners...exiting")
        return None
    else:
        print("[INFO] Markers detected")
        print("CORNERS: ", corners)
        print("CORNERS: ", len(corners))
        print("CORNER IDS: ", ids)

    # otherwise, we've found the four ArUco markers, so we can continue
    # by flattening the ArUco IDs list and initializing our list of
    # reference points
    ids = ids.flatten()

    print("ids: ", ids)
    refPts = []
    # loop over the IDs of the ArUco markers in top-left, top-right,
    # bottom-right, and bottom-left order
    # TODO - UPDATE THESE IDS with MINE
    # for i in (0, 1, 2, 3):
    # for i in (0, 256, 512, 768):
    #     # grab the index of the corner with the current ID and append the
    #     # corner (x, y)-coordinates to our list of reference points
    #     j = np.squeeze(np.where(ids == i))
    #     corner = np.squeeze(corners[j])
    #     refPts.append(corner)

    for markerCorner, markerID in zip(corners, ids):
        # extract the marker corners (which are always returned in
        # top-left, top-right, bottom-right, and bottom-left order)
        corners = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners
        # convert each of the (x, y)-coordinate pairs to integers
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))

        # draw the bounding box of the ArUCo detection
        line_color = (0, 255, 0)
        line_thickness = 2
        cv2.line(image, topLeft, topRight, line_color, line_thickness)
        cv2.line(image, topRight, bottomRight, line_color, line_thickness)
        cv2.line(image, bottomRight, bottomLeft, line_color, line_thickness)
        cv2.line(image, bottomLeft, topLeft, line_color, line_thickness)

        # compute and draw the center (x, y)-coordinates of the ArUco
        # marker
        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
        cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
        # draw the ArUco marker ID on the image
        # cv2.flip(image, 1)
        cv2.putText(
            image,
            str(markerID),
            (topLeft[0], topLeft[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.25,
            (0, 255, 0),
            2,
        )

        print("[INFO] ArUco marker ID: {}".format(markerID))

    # unpack our ArUco reference points and use the reference points to
    # define the *destination* transform matrix, making sure the points
    # are specified in top-left, top-right, bottom-right, and bottom-left
    # order
