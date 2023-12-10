# me - this DAT
# scriptOp - the OP which is cooking

# We are running this as a separate script to separate concerns
# The pose detection script is only responsible for getting the raw data
# This script is for renaming the data channels so that we have more granular ways to access the different channels in our Touchdesigner patch

def onCook(scriptOp):
    # Clear any existing data in scriptOp
    scriptOp.clear()

    # Fetch the object detection data stored in the "object_detector" operator from "./object-pose-detection.py"
    object_detection = op("object_detector").fetch("detection")

    # Count the number of objects detected
    num_object_detected = len(object_detection)
    # Set the number of samples in scriptOp to the number of objects detected
    scriptOp.numSamples = num_object_detected

    # Initialize an identifier for each object
    id = 0
    # Iterate through each detected object
    for object in object_detection:
        # Retrieve the name of the object
        object_name = object["item"]

        # Iterate through each attribute of the detected object
        for key, value in object.items():
            # Skip the 'item' attribute as it's the object's name
            if key == "item":
                continue

            # Create a unique channel name for each attribute of the object
            channel_name = f"{object_name}_{id}_{key}"
            # Append a new channel to scriptOp with the created channel name
            chan = scriptOp.appendChan(channel_name)
            # Set the value of the first sample in the channel to the attribute value
            chan[0] = value

        id += 1
