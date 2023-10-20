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

    object_detection = op("object_detector").fetch("detection")
    # print(object_detection)

    # create chanels
    num_object_detected = len(object_detection)
    scriptOp.numSamples = num_object_detected
    print(scriptOp.numSamples)
    i = 0
    for object in object_detection:
        object_name = object["item"]

        for key, value in object.items():
            if key == "item":
                continue

            channel_name = f"{object_name}_{i}_{key}"
            chan = scriptOp.appendChan(channel_name)
            chan[0] = value
            print(channel_name, value)
        i += 1

    return
