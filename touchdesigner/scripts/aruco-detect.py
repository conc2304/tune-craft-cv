# me - this DAT
# scriptOp - the OP which is cooking

import numpy as np
import cv2
import mediapipe as mp
import math

print("Media Pipe: ", mp.__version__)
print("Open CV: ", cv2.__version__)
print("ARUCO: ", cv2.aruco)
print("ME:", me)


# Define storage operator and location for storing centroids
Storage_Op = "aruco_detector"
Storage_Loc = "centroids"
op(Storage_Op).store(Storage_Loc, {})


def log(*args):
    LOG_ON = False
    if LOG_ON:
        print(*args)


# Function called to process each frame
def onCook(scriptOp):
    log("[COOK]")

    # Retrieve stored corner data
    corners_store = scriptOp.fetch(Storage_Loc, [], storeDefault=True)

    # Check if all four corners are already stored
    if corners_store is not None and len(corners_store) == 4:
        scriptOp.store("num_corners", 4)
        log("Corners Storage Found")
        return

    # Reset the number of corners to 0
    scriptOp.store("num_corners", 0)

    # Retrieve the video feed with a frame delay for improved performance
    video_feed = scriptOp.inputs[0].numpyArray(delayed=True)

    # If the video feed is empty, return without processing
    if video_feed is None:
        return

    log("[RUN - ARUCO]")

    # Prepare the image for use with OpenCV
    image = cv2.cvtColor(video_feed, cv2.COLOR_BGR2RGB)
    image = np.interp(image, (0, 1), (0, 255))
    image = np.uint8(image)

    # Get the centroids of ArUco markers and the image with added detections
    centroids, image = get_aruco_marker_data(image)

    # Convert the image back to a format suitable for TouchDesigner
    image = cv2.flip(image, 1)
    image = np.interp(image, (0, 255), (0, 1))
    image = np.float32(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Store the centroids and return the image with marker drawings
    # We will access the centroid data in "./aruco-channels.py"
    scriptOp.store(Storage_Loc, centroids)
    scriptOp.copyNumpyArray(image)

    return


# Function to detect ArUco markers and compute their centroids
# Adapted from https://pyimagesearch.com/2021/01/04/opencv-augmented-reality-ar/
def get_aruco_marker_data(image):
    """
    Detects ArUco markers in a given image and computes their centroids.

    This function detects ArUco markers in the input image using a predefined dictionary
    of ArUco markers (DICT_ARUCO_ORIGINAL). For each detected marker, it computes the centroid
    coordinates and draws the marker's bounding box and ID on the image. If no markers are
    detected, the function returns None along with the original image.

    Args:
        image (numpy.ndarray): The input image in which to detect ArUco markers. The image
                              should be in a format compatible with OpenCV (e.g., BGR or RGB).

    Returns:
        tuple: A tuple containing two elements:
              - centroids (list of tuples): A list of centroid coordinates (x, y) for each detected marker.
              - image (numpy.ndarray): The input image with drawn marker IDs and bounding boxes.
              If no markers are detected, returns None for centroids and the original image.
    """

    log("[INFO] Get Markers")
    log(image.shape)

    # If the image is None, return without processing
    if image is None:
        return None, None

    # Flip the image for processing
    image = cv2.flip(image, 1)

    # Load the ArUco dictionary and detect markers
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

    log("[INFO] detecting markers...")

    # Detect markers in the image
    (corners, ids, rejectedImgPoints ) = cv2.aruco.detectMarkers(image, arucoDict)

    # if we have not found four markers in the input image then we cannot
    # apply our augmented reality technique

    log(f"ids: {ids}")

    # Print the corners for debugging
    for i, corner in enumerate(corners):
        log(i, corner, corner.shape)
    if len(corners) == 0:
        log("[INFO] could not find corners...exiting")
        return None, image

    log("[INFO] Markers detected")
    log(f"CORNERS: {len(corners)}")
    log(f"CORNER IDS: {ids}")

    # Flatten the IDs list and initialize list of centroids
    ids = ids.flatten()
    centroids = []
    log(f"ids: {ids}")

    # Process each detected marker
    for markerCorner, markerID in zip(corners, ids):
        # Reshape the marker corner coordinates and convert to integers
        corners = markerCorner.reshape((4, 2))
        (top_left, top_right, bottom_right, bottom_left) = corners
        # convert each of the (x, y) coordinate pairs to integers
        top_right = (int(top_right[0]), int(top_right[1]))
        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
        bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
        top_left = (int(top_left[0]), int(top_left[1]))

        # Draw bounding box and compute the center of the ArUco marker
        line_color = (0, 255, 0)
        line_thickness = 2
        cv2.line(image, top_left, top_right, line_color, line_thickness)
        cv2.line(image, top_right, bottom_right, line_color, line_thickness)
        cv2.line(image, bottom_right, bottom_left, line_color, line_thickness)
        cv2.line(image, bottom_left, top_left, line_color, line_thickness)

        cX = int((top_left[0] + bottom_right[0]) / 2.0)
        cY = int((top_left[1] + bottom_right[1]) / 2.0)

        cv2.circle(image, (cX, cY), 2, (0, 0, 255), 2)

        # Draw the marker ID on the image
        cv2.putText(
            image,
            str(markerID),
            (top_left[0] + 25, top_left[1] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Append the centroid coordinates to the list
        centroids.append((cX, cY))

    return centroids, image
