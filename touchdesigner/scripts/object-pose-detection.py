# me - this DAT
# scriptOp - the OP which is cooking

# Printing version information of the libraries
print("POSE DETECTOR")
import numpy as np
import cv2
import cv2.aruco as aruco
import mediapipe as mp

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

# Defining constants for identifying landmarks on a 3D box
CENTER = 0
BACK_BOTTOM_LEFT = 1
FRONT_BOTTOM_LEFT = 2
BACK_TOP_LEFT = 3
FRONT_TOP_LEFT = 4
BACK_BOTTOM_RIGHT = 5
FRONT_BOTTOM_RIGHT = 6
BACK_TOP_RIGHT = 7
FRONT_TOP_RIGHT = 8

# Initializing MediaPipe solutions for object detection
mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

# Configuring Objectron models for detecting specific objects
# We have this outside of the onCook method so that we dont reinitialize objectron on every frame
# Notice we have static_image_mode set to true, ordinarily for video feeds using OpenCV we would set to false
# However, since we are using Touchdesigner we don't have the ability to do a while loop without blocking further execution
# So we just send in one frame at a time, its more compute expensive but it is more accurate

# higher confidence means more accuracy - ie less likely to detect non-shoes
# but that also makes it less likely to detect the shoe to begin with
objectron_shoe = mp_objectron.Objectron(
    static_image_mode=True,
    max_num_objects=1,
    min_detection_confidence=0.1,
    min_tracking_confidence=0.2,
    model_name="Shoe",
)

objectron_2 = mp_objectron.Objectron(
    static_image_mode=True,
    max_num_objects=1,
    min_detection_confidence=0.1,
    min_tracking_confidence=0.1,
    model_name="Cup",  # Options are ['Chair', 'Bicycle', 'Cup, 'Shoe']
)


# Main function to process each frame of video
def onCook(scriptOp):
    log("[COOK - OPD]")

    # Retrieve the input video frame with a frame delay for performance optimization
    video_feed = scriptOp.inputs[0].numpyArray(delayed=True)

    # List of object detection models to apply
    objectrons = [
        {"shoe": objectron_shoe},
        # {"cup": objectron_2}  # this allows us to run multiple objectrons,
        # but it is compute intensive and this script/touchdesigner runs synchronously so it blocks other execution causing lag
    ]

    # Process the video feed if it's not empty and objectron model is loaded
    if not (video_feed is None or objectron_shoe is None):
        log("[RUN]")

        # Convert color space from BGR to RGB
        image = cv2.cvtColor(video_feed, cv2.COLOR_BGR2RGB)
        # Normalize pixel values to range [0, 255] and convert to uint8
        image = np.interp(image, (0, 1), (0, 255))
        image = np.uint8(image)

        # List to store detection data
        export_data = []

        # Process each object detection model
        for item in objectrons:
            for object_name, objectron in item.items():
                # Detect objects in the image
                results = objectron.process(image)

                if results.detected_objects:
                    for detected_object in results.detected_objects:
                        # Extract 2D landmarks
                        landmarks_2d = []
                        for i, landmark in enumerate(
                            detected_object.landmarks_2d.landmark
                        ):
                            l = {
                                "x": landmark.x,
                                "y": landmark.y,
                            }

                            landmarks_2d.append(l)

                        # Extract 3D landmarks
                        landmarks_3d = []
                        for i, landmark in enumerate(
                            detected_object.landmarks_3d.landmark
                        ):
                            l = [landmark.x, landmark.y, landmark.z]
                            landmarks_3d.append(l)

                        # Draw landmarks and axis on the image
                        mp_drawing.draw_landmarks(
                            image,
                            detected_object.landmarks_2d,
                            mp_objectron.BOX_CONNECTIONS,
                        )
                        mp_drawing.draw_axis(
                            image, detected_object.rotation, detected_object.translation
                        )

                        # Convert rotation matrix to angles
                        x_angle, y_angle, z_angle = rotation_matrix_to_angles(
                            detected_object.rotation
                        )

                        log(detected_object)
                        log("x_angle, y_angle, z_angle")
                        log(x_angle, y_angle, z_angle)

                        # Prepare feature data for export
                        num_decimals = 4
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

                        # Add feature data to the export list
                        export_data.append(feature_data)

        # Convert the image back to the original color space and data type
        image = np.interp(image, (0, 255), (0, 1))
        image = np.float32(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Return the new image back to Touchdesigner as a video frame
        scriptOp.copyNumpyArray(image)
        # Store the detection data in this Script Operator's storage
        # We will be accessing it from "./object-pose-channels.py"
        scriptOp.store("detection", export_data)

    return


def rotation_matrix_to_angles(r_matrix):
    """
    Converts a rotation matrix to Euler angles (in degrees).

    This function computes the Euler angles from a given rotation matrix using the
    method described in https://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf.
    It calculates two possible solutions for the angles and logs them. The function
    returns the first set of calculated angles.

    Args:
        r_matrix (numpy.ndarray): A 3x3 rotation matrix.

    Returns:
        tuple: A tuple containing the Euler angles (theta, phi, row) in degrees.
    """
    log(["[INFO] Rotation Matrix to Angles"])

    # Compute the first angle (theta) based on the rotation matrix
    # This represents rotation around the Z-axis
    # -sinB is at position [2,0] in the rotation matrix
    r_matrix[2, 0]  # = -sinB
    theta1 = -1 * np.arcsin(r_matrix[2, 0])

    # Extracting matrix elements for further calculations
    R32 = r_matrix[2, 1]
    R33 = r_matrix[2, 2]
    phi1 = np.arctan2(R32 / np.cos(theta1), R33 / np.cos(theta1))

    # Compute the second angle (phi) based on the rotation matrix
    # This represents rotation around the Y-axis
    R21 = r_matrix[1, 0]
    R11 = r_matrix[0, 0]
    row1 = np.arctan2(R21 / np.cos(theta1), R11 / np.cos(theta1))

    # Return the set of calculated angles in degrees
    return np.degrees(theta1), np.degrees(phi1), np.degrees(row1)


def concat_to_decimals(number, decimals):
    """
    Formats a number to a specified number of decimal places, removing any trailing zeros.

    This function takes a number and formats it to have a specified number of decimal
    places. If the resulting formatted number ends with trailing zeros, they are
    removed. Additionally, if the formatted number ends with a decimal point (with no
    digits following it), the decimal point is also removed. This is useful for
    displaying numbers in a more readable format without unnecessary zeros.

    Args:
        number (float): The number to be formatted.
        decimals (int): The number of decimal places to format the number to.

    Returns:
        str: The formatted number as a string.
    """

    # Create a format string specifying the number of decimal places
    format_string = "{:.{}f}".format(number, decimals)

    # Remove any trailing zeros from the formatted number
    formatted_number = format_string.rstrip("0")

    # If the formatted number ends with a decimal point, remove it
    if formatted_number.endswith("."):
        formatted_number = formatted_number[:-1]

    return formatted_number


def log(*args):
    LOG_ON = False
    if LOG_ON:
        print(*args)
