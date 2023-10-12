# me - this DAT
# scriptOp - the OP which is cooking

import numpy as np
import cv2
import mediapipe as mp

print(mp.__version__)


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

    if not img is None:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = np.float32(gray)

        dst = cv2.cornerHarris(gray, 2, 3, 0.04)

        dst = cv2.dilate(dst, None)

        # Threshold for an optimal value, it may vary depending on the image.
        # touch is working in RGBA
        img[dst > 0.01 * dst.max()] = [255, 0, 0, 255]

        scriptOp.copyNumpyArray(img)

    return
