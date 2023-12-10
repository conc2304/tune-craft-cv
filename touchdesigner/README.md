# Directory Overview - ./touchdesigner

## Folder Structure
This folder structure represents the organization of various components for our TouchDesigner project,  focusing on computer vision and environment setup:


```bash
touchdesigner/
│
├── scripts/
│   ├── aruco-channels.py
│   ├── aruco-detect.py
│   ├── env-setup.py
│   ├── object-pose-channels.py
│   └── object-pose-detection.py
│
├── aruco-markers/
│   ├── aruco-marker-ID-[0-9].jpg
│   └── aruco-marker-ID-[0-9].svg
│
└── cold-storage/
    └── utils.py
```

The `touchdesigner/` directory is the main folder containing all TouchDesigner-related files for the project, organized as follows:

- `scripts/`: This subfolder contains several Python scripts essential for the project's functionality:
  - `aruco-channels.py` and `aruco-detect.py` for processing and detecting ArUco markers.
  - `env-setup.py` for setting up the project environment.
  - `object-pose-channels.py` and `object-pose-detection.py` for real-time object pose detection and tracking.

- `aruco-markers/`: This subfolder stores resources related to ArUco markers:
  - Includes `.jpg` and `.svg` files for each ArUco marker (ID 0 to 9), used for marker generation and recognition.

- `cold-storage/`: A subfolder for deprecated scripts:
  - Contains `utils.py`, which contains various attempts at getting converting the rotation matrix data into degree angle values

