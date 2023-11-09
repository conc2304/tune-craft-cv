# me - this DAT
# scriptOp - the OP which is cooking

import numpy as np
import cv2

# import cv2.aruco as aruco
import mediapipe as mp
import math

print("Media Pipe: ", mp.__version__)
print("Open CV: ", cv2.__version__)
print("ARUCO: ", cv2.aruco)
print("ME:", me)


Resolution = (640, 360)

LOG_ON = True
DO_DRAW = True
Storage_Op = "aruco_detector"
Storage_Loc = "centroids"
# op(Storage_Op).store(Storage_Loc, {})


def log(str):
    if LOG_ON:
        print(str)


# press 'Setup Parameters' in the OP to call this function to re-create the pa
# rameters.
def onSetupParameters(scriptOp):
    return


# called whenever custom pulse parameter is pushed
def onPulse(par):
    log("[PULSE]")
    return


def onCook(scriptOp):
    log("[COOK]")

    corners_store = scriptOp.fetch(Storage_Loc, [], storeDefault=True)
    log(f"corners_store {len(corners_store)}")
    # If we have all 4 corners we dont need to get them again
    if corners_store != None and len(corners_store) == 4:
        scriptOp.store("num_corners", 4)
        log(f"Corners Storage Found")

        return

    scriptOp.store("num_corners", 0)

    # grab the input to the scriptTOP with a frame delayed
    # for faster operation (compare TopTo CHOP)
    # rgba values as 0-1
    video_feed = scriptOp.inputs[0].numpyArray(delayed=True)

    if video_feed is None:
        return

    log("[RUN - ARUCO]")
    image = cv2.cvtColor(video_feed, cv2.COLOR_BGR2RGB)

    # Remap the values to the range [0, 255]
    image = np.interp(image, (0, 1), (0, 255))

    # convert data to uint8
    image = np.uint8(image)

    centroids = get_aruco_marker_data(image=image)

    scriptOp.store(Storage_Loc, centroids)

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

    # load the ArUCo dictionary, grab the ArUCo parameters, and detect
    # the markers
    log("[INFO] detecting markers...")
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    # arucoParams = cv2.aruco
    # print(cv2.aruco.DetectorParameters())
    # arucoParams = aruco.DetectorParameters()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict)
    # if we have not found four markers in the input image then we cannot
    # apply our augmented reality technique
    if len(corners) != 4:
        log("[INFO] could not find 4 corners...exiting")
        return None
    else:
        log(f"[INFO] Markers detected")
        log(f"CORNERS: {corners}")
        log(f"CORNERS: {len(corners)}")
        log(f"CORNER IDS: {ids}")

    # otherwise, we've found the four ArUco markers, so we can continue
    # by flattening the ArUco IDs list and initializing our list of
    # reference points
    ids = ids.flatten()

    log(f"ids: {ids}")

    centroids = []
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
        cv2.putText(
            image,
            str(markerID),
            (topLeft[0], topLeft[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.25,
            (0, 255, 0),
            2,
        )

        centroids.append((cX, cY))

        # log("[INFO] ArUco marker ID: {}".format(markerID))

    return centroids
