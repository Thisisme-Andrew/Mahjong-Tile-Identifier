import os

from roboflow import Roboflow

api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise RuntimeError("Set ROBOFLOW_API_KEY before running this script.")

rf = Roboflow(api_key=api_key)
project = rf.workspace(os.getenv("WORKSPACE_NAME")).project(os.getenv("PROJECT_NAME"))
version = project.version(1)
# Download into a fixed folder so you can copy into mahjong-1/ without guessing paths.
dataset = version.download("yolov8", location="dataset/_roboflow_download")