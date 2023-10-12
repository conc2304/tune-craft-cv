import sys
import os
import platform

# C:/Users/josec/anaconda3/envs/td-cv/Lib/site-packages
# C:/Users/josec/anaconda3/envs/td-cv/Lib/site-packages


def onStart():
    user = "josec"
    condaEnv = "td-cv"

    if platform.system() == "Windows":
        print(f"Python Version: {sys.version_info.major}.{sys.version_info.minor}")
        if sys.version_info.major >= 3 and sys.version_info.minor >= 8:
            """
            Double check all the following paths, it could be that your anaconda 'envs' folder is not in your User folder depending on your conda install settings and conda version.
            """
            os.add_dll_directory(
                "C:/Users/" + user + "/anaconda3/envs/" + condaEnv + "/DLLs"
            )
            os.add_dll_directory(
                "C:/Users/" + user + "/anaconda3/envs/" + condaEnv + "/Library/bin"
            )
        else:
            """
            Double check all the following paths, it could be that your anaconda 'envs' folder is not in your User folder depending on your conda install settings and conda version.
            """
            # Not the most elegant solution, but we need to control load order
            os.environ["PATH"] = (
                "C:/Users/"
                + user
                + "/miniconda3/envs/"
                + condaEnv
                + "/DLLs"
                + os.pathsep
                + os.environ["PATH"]
            )
            os.environ["PATH"] = (
                "C:/Users/"
                + user
                + "/miniconda3/envs/"
                + condaEnv
                + "/Library/bin"
                + os.pathsep
                + os.environ["PATH"]
            )

        sys.path = [
            "C:/Users/" + user + "/anaconda3/envs/" + condaEnv + "/Lib/site-packages"
        ] + sys.path

        sys.path = list(set(sys.path))

        print("PATH: ", sys.path)

    else:
        """
        MacOS users should include path to .dlybs / MacOS binaries, site-packages
        """
        os.environ["PATH"] = (
            "/Users/"
            + user
            + "/opt/miniconda3/envs/"
            + condaEnv
            + "/lib"
            + os.pathsep
            + os.environ["PATH"]
        )
        os.environ["PATH"] = (
            "/Users/"
            + user
            + "/opt/miniconda3/envs/"
            + condaEnv
            + "/bin"
            + os.pathsep
            + os.environ["PATH"]
        )
        # The following path might need editing (python3.9) based on the python version used with conda
        sys.path = [
            "/Users/"
            + user
            + "/opt/miniconda3/envs/"
            + condaEnv
            + "/lib/python3.9/site-packages"
        ] + sys.path

    return


def onCreate():
    return


def onExit():
    return


def onFrameStart(frame):
    return


def onFrameEnd(frame):
    return


def onPlayStateChange(state):
    return


def onDeviceChange():
    return


def onProjectPreSave():
    return


def onProjectPostSave():
    return
