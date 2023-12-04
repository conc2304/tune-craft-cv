import numpy as np
import cv2
import cv2.aruco as aruco
import mediapipe as mp
import math

# Methods that are no longer used but keeping around for posterity

def log(*args):
    LOG_ON = False
    if LOG_ON:
        print(*args)

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
    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]

    # Calculate the angle using atan2
    angle_radians = math.atan2(delta_y, delta_x)

    # Convert the angle to degrees
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees


# https://learnopencv.com/rotation-matrix-to-euler-angles/
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


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
