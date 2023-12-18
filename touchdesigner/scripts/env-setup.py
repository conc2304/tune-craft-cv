import sys
import os
import platform

# example path -> C:/Users/josec/anaconda3/envs/td-cv/Lib/site-packages


# Function to execute on script start in Touchdesigner
def onStart():
    # User and conda environment name configuration
    user = "josec"
    condaEnv = "td-cv2"  # this is the conda env name from environment.yml

    # Check if the operating system is Windows
    if platform.system() == "Windows":
        print(f"Python Version: {sys.version_info.major}.{sys.version_info.minor}")

        # Check Python version for compatibility (3.8 or higher)
        if sys.version_info.major >= 3 and sys.version_info.minor >= 8:
            """
            Double check all the following paths, it could be that your anaconda 'envs' folder is not in your User folder depending on your conda install settings and conda version.
            """

            # Add directories to DLL search paths for Python 3.8 or higher
            # These paths may vary depending on Anaconda installation and settings
            os.add_dll_directory(
                "C:/Users/" + user + "/anaconda3/envs/" + condaEnv + "/DLLs"
            )
            os.add_dll_directory(
                "C:/Users/" + user + "/anaconda3/envs/" + condaEnv + "/Library/bin"
            )
            os.add_dll_directory(
                "C:/Users/"
                + user
                + "/anaconda3/envs/"
                + condaEnv
                + "/Lib/site-packages"
            )

        else:
            """
            Double check all the following paths, it could be that your anaconda 'envs' folder is not in your User folder depending on your conda install settings and conda version.
            """
            # Not the most elegant solution, but we need to control load order
            # For Python versions lower than 3.8, manually adjust PATH for DLL loading
            # This is a workaround to control DLL load order
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

        # Modify sys.path to include the site-packages directory for Python imports
        sys.path = [
            "C:/Users/" + user + "/anaconda3/envs/" + condaEnv + "/Lib/site-packages"
        ] + sys.path

        # Remove any duplicate entries in sys.path
        sys.path = list(set(sys.path))

        print("PATH: ", sys.path)

    else:
        # MacOS-specific configuration
        # Set PATH to include directories for dylibs and binaries

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

        # Modify sys.path to include the site-packages directory
        # Note: Python version in the path may need to be updated based on the conda environment
        sys.path = [
            "/Users/"
            + user
            + "/opt/miniconda3/envs/"
            + condaEnv
            + "/lib/python3.9/site-packages"
        ] + sys.path

    return
