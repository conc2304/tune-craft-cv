# me - this DAT
# scriptOp - the OP which is cooking

import numpy as np
import cv2
import mediapipe as mp
import math
import imutils

print("Imutils: ", imutils.__version__)
print("Media Pipe: ", mp.__version__)
print("ME:", me)

mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

Resolution = (640, 360)
objectron_shoe = mp_objectron.Objectron(
    static_image_mode=False,
    max_num_objects=2,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.70,
    model_name="Shoe",
    image_size=Resolution,
)

objectron_cup = mp_objectron.Objectron(
    static_image_mode=False,
    max_num_objects=2,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.70,
    model_name="Cup",
    image_size=Resolution,
)


print("ME:", me)


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

    objectrons = [{"shoe": objectron_shoe}, {"cup": objectron_cup}]

    if not (video_feed is None or objectron_shoe is None):
        print("[RUN]")
        image = cv2.cvtColor(video_feed, cv2.COLOR_BGR2RGB)

        # Remap the values to the range [0, 255]
        image = np.interp(image, (0, 1), (0, 255))
        # convert data to uint8
        image = np.uint8(image)

        export_data = []
        for item in objectrons:
            for object_name, objectron in item.items():
                results = objectron.process(image)
                objectDetected = results.detected_objects

                # print(f"Detected {object_name}: ", objectDetected)

                if results.detected_objects:
                    for detected_object in results.detected_objects:
                        mp_drawing.draw_landmarks(
                            image,
                            detected_object.landmarks_2d,
                            mp_objectron.BOX_CONNECTIONS,
                        )

                        mp_drawing.draw_axis(
                            image, detected_object.rotation, detected_object.translation
                        )

                        # export data to table for touch to use
                        angles = rotationMatrixToEulerAngles(
                            np.array(detected_object.rotation)
                        )

                        # print(detected_object.translation)

                        num_decimals = 4
                        # feature_data = [
                        #     object_name,
                        #     concat_to_decimals(
                        #         detected_object.translation[0], num_decimals
                        #     ),
                        #     concat_to_decimals(
                        #         detected_object.translation[1], num_decimals
                        #     ),
                        #     concat_to_decimals(
                        #         detected_object.translation[2], num_decimals
                        #     ),
                        #     concat_to_decimals(angles[0], num_decimals),
                        #     concat_to_decimals(angles[1], num_decimals),
                        #     concat_to_decimals(angles[2], num_decimals),
                        # ]

                        feature_data = {
                            "item": object_name,
                            "tx": concat_to_decimals(
                                detected_object.translation[0], num_decimals
                            ),
                            "ty": concat_to_decimals(
                                detected_object.translation[1], num_decimals
                            ),
                            "tz": concat_to_decimals(
                                detected_object.translation[2], num_decimals
                            ),
                            "rx": concat_to_decimals(angles[0], num_decimals),
                            "ry": concat_to_decimals(angles[1], num_decimals),
                            "rz": concat_to_decimals(angles[2], num_decimals),
                        }

                        # op("translation_table").appendRow(detected_object.translation)
                        # # for arr in detected_object.rotation:
                        # op("rotation_table").appendRow(angles)
                        # op("feature_data").appendRow(feature_data)
                        export_data.append(feature_data)

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
    # load the ArUCo dictionary, grab the ArUCo parameters, and detect
    # the markers
    print("[INFO] detecting markers...")
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(
        image, arucoDict, parameters=arucoParams
    )
    # if we have not found four markers in the input image then we cannot
    # apply our augmented reality technique
    if len(corners) != 4:
        print("[INFO] could not find 4 corners...exiting")
        # sys.exit(0)
    else:
        print("[INFO] Markers detected")
        print(corners, ids)

    # otherwise, we've found the four ArUco markers, so we can continue
    # by flattening the ArUco IDs list and initializing our list of
    # reference points
    print("[INFO] constructing augmented reality visualization...")
    ids = ids.flatten()
    refPts = []
    # loop over the IDs of the ArUco markers in top-left, top-right,
    # bottom-right, and bottom-left order
    # TODO - UPDATE THESE IDS with MINE
    for i in (923, 1001, 241, 1007):
        # grab the index of the corner with the current ID and append the
        # corner (x, y)-coordinates to our list of reference points
        j = np.squeeze(np.where(ids == i))
        corner = np.squeeze(corners[j])
        refPts.append(corner)

    # unpack our ArUco reference points and use the reference points to
    # define the *destination* transform matrix, making sure the points
    # are specified in top-left, top-right, bottom-right, and bottom-left
    # order
    (refPtTL, refPtTR, refPtBR, refPtBL) = refPts
    dstMat = [refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]]
    dstMat = np.array(dstMat)
    # grab the spatial dimensions of the source image and define the
    # transform matrix for the *source* image in top-left, top-right,
    # bottom-right, and bottom-left order
    # TODO source is the UI that needs to be mapped in space
    (srcH, srcW) = source.shape[:2]
    srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])
    # compute the homography matrix and then warp the source image to the
    # destination based on the homography
    (H, _) = cv2.findHomography(srcMat, dstMat)
    warped = cv2.warpPerspective(source, H, (imgW, imgH))

    # construct a mask for the source image now that the perspective warp
    # has taken place (we'll need this mask to copy the source image into
    # the destination)
    mask = np.zeros((imgH, imgW), dtype="uint8")
    cv2.fillConvexPoly(mask, dstMat.astype("int32"), (255, 255, 255), cv2.LINE_AA)
    # this step is optional, but to give the source image a black border
    # surrounding it when applied to the source image, you can apply a
    # dilation operation
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, rect, iterations=2)
    # create a three channel version of the mask by stacking it depth-wise,
    # such that we can copy the warped source image into the input image
    maskScaled = mask.copy() / 255.0
    maskScaled = np.dstack([maskScaled] * 3)
    # copy the warped source image into the input image by (1) multiplying
    # the warped image and masked together, (2) multiplying the original
    # input image with the mask (giving more weight to the input where
    # there *ARE NOT* masked pixels), and (3) adding the resulting
    # multiplications together
    warpedMultiplied = cv2.multiply(warped.astype("float"), maskScaled)
    imageMultiplied = cv2.multiply(image.astype(float), 1.0 - maskScaled)
    output = cv2.add(warpedMultiplied, imageMultiplied)
    output = output.astype("uint8")

    return corners, ids


def concat_to_decimals(number, decimals):
    format_string = "{:.{}f}".format(number, decimals)
    formatted_number = format_string.rstrip("0")  # Remove trailing zeros
    if formatted_number.endswith("."):
        formatted_number = formatted_number[:-1]  # Remove trailing dot if no decimals
    return formatted_number


# https://learnopencv.com/rotation-matrix-to-euler-angles/
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    # assert isRotationMatrix(R)

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])
