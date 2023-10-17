# me - this DAT
# scriptOp - the OP which is cooking

import numpy as np
import cv2
import mediapipe as mp

print("Media Pipe: ", mp.__version__)

mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

objectron = mp_objectron.Objectron(
    static_image_mode=False,
    max_num_objects=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.70,
    model_name="Cup",
)


# press 'Setup Parameters' in the OP to call this function to re-create the parameters.
def onSetupParameters(scriptOp):
    return


# called whenever custom pulse parameter is pushed
def onPulse(par):
    return


def onCook(scriptOp):
    # grab the input to the scriptTOP with a frame delayed
    # for faster operation (compare TopTo CHOP)
    # rgba values as 0-1
    video_feed = scriptOp.inputs[0].numpyArray(delayed=True)

    if not video_feed is None:
        image = cv2.cvtColor(video_feed, cv2.COLOR_BGR2RGB)

        # Remap the values to the range [0, 255]
        image = np.interp(image, (0, 1), (0, 255))
        # convert data to uint8
        image = np.uint8(image)
        results = objectron.process(image)
        objectDetected = results.detected_objects

        print(objectDetected)
        if results.detected_objects:
            for detected_object in results.detected_objects:
                mp_drawing.draw_landmarks(
                    image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS
                )

                mp_drawing.draw_axis(
                    image, detected_object.rotation, detected_object.translation
                )

                # export data to table for touch to use

        image = cv2.flip(image, 1)
        image = np.interp(image, (0, 255), (0, 1))
        image = np.float32(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        scriptOp.copyNumpyArray(image)

    return
