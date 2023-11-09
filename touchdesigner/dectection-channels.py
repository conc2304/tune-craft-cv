# me - this DAT
# scriptOp - the OP which is cooking


# press 'Setup Parameters' in the OP to call this function to re-create the parameters.
def onSetupParameters(scriptOp):
    return


# called whenever custom pulse parameter is pushed
def onPulse(par):
    return


def onCook(scriptOp):
    scriptOp.clear()

    # get the data we stored in the
    object_detection = op("object_detector").fetch("detection")

    # create channels with the names we want
    num_object_detected = len(object_detection)
    scriptOp.numSamples = num_object_detected

    id = 0
    for object in object_detection:
        object_name = object["item"]

        for key, value in object.items():
            if key == "item":
                continue

            channel_name = f"{object_name}_{id}_{key}"
            chan = scriptOp.appendChan(channel_name)
            chan[0] = value
        id += 1

    return
