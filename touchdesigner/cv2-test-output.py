# me - this DAT
# scriptOp - the OP which is cooking

import numpy as np
import cv2

# import matplotlib

# import mediapipe as mp


print(cv2.__version__)


# press 'Setup Parameters' in the OP to call this function to re-create the parameters.
def onSetupParameters(scriptOp):
    return


# called whenever custom pulse parameter is pushed
def onPulse(par):
    return


def onCook(scriptOp):
    # grab the input to the scriptTOP with a frame delayed
    # for faster operation (compare TopTo CHOP)
    img = scriptOp.inputs[0].numpyArray(delayed=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find Harris corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    # print(corners)
    # clear the table we are writing to
    op("table1").clear()
    for i in corners:
        op("table1").appendRow([i[0], i[1]])

    scriptOp.copyNumpyArray(img)

    return
