# me - this DAT
# scriptOp - the OP which is cooking

print("POSE DETECTOR")
import numpy as np
import cv2
import cv2.aruco as aruco
import mediapipe as mp
import math

print("Media Pipe: ", mp.__version__)
print("Open CV: ", cv2.__version__)
print("ARUCO: ", aruco)
print("ME:", me)


# """The 9 3D box landmarks."""
#
#       3 + + + + + + + + 7
#       +\                +\          UP
#       + \               + \
#       +  \              +  \        |
#       +   4 + + + + + + + + 8       | y
#       +   +             +   +       |
#       +   +             +   +       |
#       +   +     (0)     +   +       .------- x
#       +   +             +   +        \
#       1 + + + + + + + + 5   +         \
#        \  +              \  +          \ z
#         \ +               \ +           \
#          \+                \+
#           2 + + + + + + + + 6

CENTER = 0
BACK_BOTTOM_LEFT = 1
FRONT_BOTTOM_LEFT = 2
BACK_TOP_LEFT = 3
FRONT_TOP_LEFT = 4
BACK_BOTTOM_RIGHT = 5
FRONT_BOTTOM_RIGHT = 6
BACK_TOP_RIGHT = 7
FRONT_TOP_RIGHT = 8

mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

Resolution = (640, 360)
objectron_shoe = mp_objectron.Objectron(
    static_image_mode=True,
    max_num_objects=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.70,
    model_name="Shoe",
    # image_size=Resolution,
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


def onCook(scriptOp):
    log("[COOK - OPD]")
    # grab the input to the scriptTOP with a frame delayed
    # for faster operation (compare TopTo CHOP)
    # rgba values as 0-1
    video_feed = scriptOp.inputs[0].numpyArray(delayed=True)

    objectrons = [
        {"shoe": objectron_shoe},
        # {"cup": objectron_cup}
    ]

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
                        landmarks_2d = []
                        for i, landmark in enumerate(
                            detected_object.landmarks_2d.landmark
                        ):
                            l = {
                                "x": landmark.x,
                                "y": landmark.y,
                            }

                            landmarks_2d.append(l)

                        landmarks_3d = []
                        for i, landmark in enumerate(
                            detected_object.landmarks_3d.landmark
                        ):
                            l = [landmark.x, landmark.y, landmark.z]
                            landmarks_3d.append(l)

                        mp_drawing.draw_landmarks(
                            image,
                            detected_object.landmarks_2d,
                            mp_objectron.BOX_CONNECTIONS,
                        )
                        mp_drawing.draw_axis(
                            image, detected_object.rotation, detected_object.translation
                        )

                        log(detected_object)

                        num_decimals = 4
                        # TODO - this does not seem right
                        # angles = calculate_rotation_angles(landmarks_3d)
                        x_angle, y_angle, z_angle = rotation_matrix_to_angles(
                            detected_object.rotation
                        )

                        log("x_angle, y_angle, z_angle")
                        log(x_angle, y_angle, z_angle)

                        feature_data = {
                            "item": object_name,
                            "cx": concat_to_decimals(
                                landmarks_2d[CENTER]["x"], num_decimals
                            ),
                            "cy": concat_to_decimals(
                                landmarks_2d[CENTER]["y"], num_decimals
                            ),
                            "rx": concat_to_decimals(x_angle, num_decimals),
                            "ry": concat_to_decimals(y_angle, num_decimals),
                            "rz": concat_to_decimals(z_angle, num_decimals),
                        }

                        export_data.append(feature_data)

        image = np.interp(image, (0, 255), (0, 1))
        image = np.float32(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        scriptOp.copyNumpyArray(image)
        scriptOp.store("detection", export_data)

    return


def rotation_matrix_to_angles(r_matrix):
    log(["[INFO] Rotation Matrix to Angles"])
    # https://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
    r_matrix[2, 0]  # = -sinB
    theta1 = -1 * np.arcsin(r_matrix[2, 0])
    theta2 = np.pi + 1 * np.arcsin(r_matrix[2, 0])

    log(np.degrees(theta1), np.degrees(theta2))
    R32 = r_matrix[2, 1]
    R32
    R33 = r_matrix[2, 2]
    R33
    phi1 = np.arctan2(R32 / np.cos(theta1), R33 / np.cos(theta1))
    phi2 = np.arctan2(R32 / np.cos(theta2), R33 / np.cos(theta2))

    log(np.degrees(phi1), np.degrees(phi2))
    R21 = r_matrix[1, 0]
    R21
    R11 = r_matrix[0, 0]
    R11

    row1 = np.arctan2(R21 / np.cos(theta1), R11 / np.cos(theta1))
    row2 = np.arctan2(R21 / np.cos(theta2), R11 / np.cos(theta2))

    log(np.degrees(row1), np.degrees(row2))
    return np.degrees(theta1), np.degrees(phi1), np.degrees(row1)


def concat_to_decimals(number, decimals):
    format_string = "{:.{}f}".format(number, decimals)
    formatted_number = format_string.rstrip("0")  # Remove trailing zeros
    if formatted_number.endswith("."):
        formatted_number = formatted_number[:-1]  # Remove trailing dot if no decimals
    return formatted_number


def calculate_rotation_angles(landmarks_3d):
    # Define vectors for cube's edges
    vector_x = np.array(landmarks_3d[BACK_BOTTOM_RIGHT]) - np.array(
        landmarks_3d[BACK_BOTTOM_LEFT]
    )  # BACK_BOTTOM_RIGHT - BACK_BOTTOM_LEFT
    vector_y = np.array(landmarks_3d[BACK_TOP_LEFT]) - np.array(
        landmarks_3d[BACK_BOTTOM_LEFT]
    )
    vector_z = np.array(landmarks_3d[FRONT_BOTTOM_LEFT]) - np.array(
        landmarks_3d[BACK_BOTTOM_LEFT]
    )

    # Normalize vectors
    unit_vector_x = vector_x / np.linalg.norm(vector_x)
    unit_vector_y = vector_y / np.linalg.norm(vector_y)
    unit_vector_z = vector_z / np.linalg.norm(vector_z)

    # Calculate rotation angles using atan2
    angle_x = math.atan2(unit_vector_x[1], unit_vector_x[0])  # Rotation in XY plane
    angle_y = math.atan2(unit_vector_y[2], unit_vector_y[1])  # Rotation in YZ plane
    angle_z = math.atan2(unit_vector_z[0], unit_vector_z[2])  # Rotation in ZX plane

    # Convert angles to degrees and adjust range to [0, 360]
    angle_x_degrees = (math.degrees(angle_x) + 360) % 360
    angle_y_degrees = (math.degrees(angle_y) + 360) % 360
    angle_z_degrees = (math.degrees(angle_z) + 360) % 360

    # return angles from 0 to 180
    return angle_x_degrees, angle_y_degrees, angle_z_degrees


def calculate_adjacent_angle(point1, point2):
    """
    Calculates the adjacent angle of a right triangle formed by two points in 2D space.
    Args:
    point1 (tuple): The first point (x, y).
    point2 (tuple): The second point (x, y).

    Returns:
    float: The adjacent angle in degrees.
    """

    # Calculate the differences in x and y coordinates
    delta_x = point2["x"] - point1["x"]
    delta_y = point2["y"] - point1["y"]

    # Calculate the angle using atan2
    angle_radians = math.atan2(delta_y, delta_x)

    # Convert the angle to degrees
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees
