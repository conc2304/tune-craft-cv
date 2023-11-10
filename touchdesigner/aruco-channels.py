# me - this DAT
# scriptOp - the OP which is cooking
import numpy as np

Storage_Op = "aruco_detector"
Storage_Loc = "centroids"


# press 'Setup Parameters' in the OP to call this function to re-create the parameters.
def onSetupParameters(scriptOp):
    return


# called whenever custom pulse parameter is pushed
def onPulse(par):
    return


def onCook(scriptOp):
    # Get the data we stored in the Aruco Marker Detection Script
    centroids = op(Storage_Op).fetch(Storage_Loc)
    # print("CENTROIDS", centroids)

    if len(centroids) != 4:
        # Exit out
        # We dont have a way to only update the 'detected' value in storage while keeping everything intact

        # Get the corners from storage
        stored_corners = op("script2").chans("c_*")
        # Clear the Storage
        scriptOp.clear()

        # Rewrite the storage with detection set to false
        chan = scriptOp.appendChan("detected")
        chan[0] = 0

        labeled_corners = label_corners(stored_corners)
        for corner_name, (x, y) in labeled_corners.items():
            x_chan = scriptOp.appendChan(f"{corner_name}_x")
            x_chan[0] = x
            y_chan = scriptOp.appendChan(f"{corner_name}_y")
            y_chan[0] = y

        return
    else:
        # Clear and set detected flag
        scriptOp.clear()
        chan = scriptOp.appendChan("detected")
        chan[0] = 1

    centroids = np.array(centroids)

    x = centroids[:, 0]
    y = centroids[:, 1]

    # Using argsort to handle cases with multiple points having extreme values
    sorted_indices_x = np.argsort(x)
    sorted_indices_y = np.argsort(y)

    # The lowest x will be our left, and the highest x will be our right
    left_indices = sorted_indices_x[:2]  # This could be lower left or upper left
    right_indices = sorted_indices_x[-2:]  # This could be lower right or upper right

    # Among the left_indices, the one with lower y is lower left
    lower_left_index = left_indices[np.argmin(y[left_indices])]
    upper_left_index = left_indices[np.argmax(y[left_indices])]

    # Among the right_indices, the one with lower y is lower right
    lower_right_index = right_indices[np.argmin(y[right_indices])]
    upper_right_index = right_indices[np.argmax(y[right_indices])]

    corners = {
        "c_upper_left": tuple(centroids[upper_left_index]),
        "c_upper_right": tuple(centroids[upper_right_index]),
        "c_lower_left": tuple(centroids[lower_left_index]),
        "c_lower_right": tuple(centroids[lower_right_index]),
    }

    for corner_name, (x, y) in corners.items():
        x_chan = scriptOp.appendChan(f"{corner_name}_x")
        x_chan[0] = x
        y_chan = scriptOp.appendChan(f"{corner_name}_y")
        y_chan[0] = y

    return


import numpy as np


def label_corners(coordinates):
    # Assuming the coordinates are in the format [x1, y1, x2, y2, ..., xn, yn]
    # We reshape the list into a 2D array of (x, y) pairs
    points = np.array(coordinates).reshape(-1, 2)

    # Sort the points into corners
    # Upper left will have min(x) and max(y), upper right will have max(x) and max(y),
    # lower left will have min(x) and min(y), lower right will have max(x) and min(y).
    ul = points[np.argmin(points[:, 0]) + np.argmax(points[:, 1])]  # upper left
    ur = points[np.argmax(points[:, 0]) + np.argmax(points[:, 1])]  # upper right
    ll = points[np.argmin(points[:, 0]) + np.argmin(points[:, 1])]  # lower left
    lr = points[np.argmax(points[:, 0]) + np.argmin(points[:, 1])]  # lower right

    # Define the labels
    labels = [
        "c_upper_left",
        "c_upper_right",
        "c_lower_left",
        "c_lower_right",
    ]

    # Map the coordinates to their respective labels
    corners = [ul, ur, ll, lr]
    labeled_corners = {
        labels[i]: corners[i % len(corners)].tolist() for i in range(len(corners))
    }

    return labeled_corners
