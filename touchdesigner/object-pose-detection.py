# me - this DAT
# scriptOp - the OP which is cooking

print("POSE DETECTOR")
import numpy as np
import cv2
import cv2.aruco as aruco
import mediapipe as mp
import math

print("ME:", me)

# import imutils

# print("Imutils: ", imutils.__version__)
print("Media Pipe: ", mp.__version__)
print("Open CV: ", cv2.__version__)
print("ARUCO: ", aruco)
print("ME:", me)


mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

Resolution = (640, 360)
objectron_shoe = mp_objectron.Objectron(
    static_image_mode=True,
    max_num_objects=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.70,
    model_name="Shoe",
    image_size=Resolution,
)

objectron_cup = mp_objectron.Objectron(
    static_image_mode=True,
    max_num_objects=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.70,
    model_name="Chair",
    image_size=Resolution,
)

LOG_ON = False


def log(*args):
    if LOG_ON:
        print(*args)


# press 'Setup Parameters' in the OP to call this function to re-create the pa
# rameters.
def onSetupParameters(scriptOp):
    return


# called whenever custom pulse parameter is pushed
def onPulse(par):
    return


def onCook(scriptOp):
    log("[COOK - OPD]")
    # grab the input to the scriptTOP with a frame delayed
    # for faster operation (compare TopTo CHOP)
    # rgba values as 0-1
    video_feed = scriptOp.inputs[0].numpyArray(delayed=True)

    objectrons = [{"shoe": objectron_shoe}, {"cup": objectron_cup}]

    if not (video_feed is None or objectron_shoe is None):
        log("[RUN]")
        image = cv2.cvtColor(video_feed, cv2.COLOR_BGR2RGB)

        # Remap the values to the range [0, 255]
        image = np.interp(image, (0, 1), (0, 255))
        # convert data to uint8
        image = np.uint8(image)

        export_data = []
        # running multiple objectrons, one for each item type we are looking for
        for item in objectrons:
            for object_name, objectron in item.items():
                results = objectron.process(image)

                if results.detected_objects:
                    for detected_object in results.detected_objects:
                        landmarks = []
                        for i, landmark in enumerate(
                            detected_object.landmarks_2d.landmark
                        ):
                            l = {
                                "x": landmark.x,
                                "y": landmark.y,
                            }
                            # print("l: ", landmark.x)
                            landmarks.append(l)

                        # print("----T----", detected_object.translation)
                        # print("----R----", detected_object.rotation)

                        mp_drawing.draw_landmarks(
                            image,
                            detected_object.landmarks_2d,
                            mp_objectron.BOX_CONNECTIONS,
                        )
                        mp_drawing.draw_axis(
                            image, detected_object.rotation, detected_object.translation
                        )

                        angles = rotationMatrixToEulerAngles(
                            np.array(detected_object.rotation)
                        )

                        log(detected_object)

                        num_decimals = 4
                        center = find_center_landmark(landmarks)

                        # print("----C----", center)

                        feature_data = {
                            "item": object_name,
                            "cx": concat_to_decimals(center["x"], num_decimals),
                            "cy": concat_to_decimals(center["y"], num_decimals),
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

                        export_data.append(feature_data)

        image = np.interp(image, (0, 255), (0, 1))
        image = np.float32(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        scriptOp.copyNumpyArray(image)
        scriptOp.store("detection", export_data)

    return


def concat_to_decimals(number, decimals):
    format_string = "{:.{}f}".format(number, decimals)
    formatted_number = format_string.rstrip("0")  # Remove trailing zeros
    if formatted_number.endswith("."):
        formatted_number = formatted_number[:-1]  # Remove trailing dot if no decimals
    return formatted_number


def find_center_landmark(landmarks):
    # Initialize sums of x and y coordinates
    log("find_center_landmark")
    log(landmarks)
    sum_x = 0
    sum_y = 0

    # Iterate over all landmarks to sum up the coordinates
    for landmark in landmarks:
        sum_x += landmark["x"]
        sum_y += landmark["y"]

    # Calculate the average x and y coordinates
    center_x = sum_x / len(landmarks)
    center_y = sum_y / len(landmarks)

    return {"x": center_x, "y": center_y}


# https://learnopencv.com/rotation-matrix-to-euler-angles/
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
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
