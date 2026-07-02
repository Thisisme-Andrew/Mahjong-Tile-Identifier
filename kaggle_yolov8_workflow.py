# %% [markdown]
# # Mahjong Tile Detection (YOLOv8) - Kaggle Workflow

# %%
from pathlib import Path

DATASET_ROOT = Path("PATH_TO_DATASET")
PROJECT_ROOT = Path("PATH_TO_PROJECT")

original_yaml_path = DATASET_ROOT / "data.yaml"

print(f"Dataset root set to: {DATASET_ROOT}")
print(f"Checking for data.yaml... {'FOUND!' if original_yaml_path.exists() else 'NOT FOUND'}")

# %%
!pip install -q --upgrade ultralytics pyyaml

# %%
import yaml

PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

with open(original_yaml_path, encoding="utf-8") as f:
    yaml_content = yaml.safe_load(f)

# Point the base path directly to the folder containing train, valid, test
yaml_content["path"]  = str(DATASET_ROOT)
yaml_content["train"] = "train/images"
yaml_content["val"]   = "valid/images"
yaml_content["test"]  = "test/images"

writable_data_yaml = PROJECT_ROOT / "data.yaml"
with open(writable_data_yaml, "w", encoding="utf-8") as f:
    yaml.dump(yaml_content, f, sort_keys=False)

print(f"Dynamic image directory base path set to: {yaml_content['path']}")

# %%
data_yaml = PROJECT_ROOT / "data.yaml"
with open(data_yaml, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

dataset_path = Path(cfg["path"])
for split_key, label in (("train", "train"), ("val", "valid"), ("test", "test")):
    split_dir = dataset_path / cfg[split_key]
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")
    image_count = len([
        f for f in split_dir.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])
    print(f"{label}: {image_count} images at {split_dir}")
print("Preflight passed.")

# %%
from ultralytics import YOLO

# Use checkout point and model instead of the yolov8s.pt model if starting from checkpoint
# Note that you will need to have best.pt and last.pt file from last run
# CHECKPOINT = Path("/kaggle/input/datasets/khebrahapps/mahjong-tile-identifier/last.pt")

# model = YOLO(str(CHECKPOINT))

model = YOLO("yolov8s.pt")
results = model.train(
    data=str(data_yaml),
    epochs=100,
    imgsz=640,
    batch=32,
    workers=4,
    device=0,
    name="mahjong_hand_yolov8s",
    save=True,
    plots=True,
    cache=False,
    patience=15,
    save_period=10,         # This is if training takes longer than 12 hours kaggle gives.
    # resume=True,          # This is if you want to resume training from the last checkpoint (if it got stopped at last 12 hour run on kaggle)
)

# %%
# Validate on best weights — not last
best_model = YOLO(results.save_dir / "weights" / "best.pt")
metrics = best_model.val(data=str(data_yaml))   # ✅ best_model

# %%
# ONNX — cross-platform for onnxruntime-react-native
onnx_path = best_model.export(        # get onyx model for react native
    format="onnx",
    imgsz=640,
    dynamic=False,
    simplify=True,
    opset=12,                                
    nms=True,
)
print("ONNX:", onnx_path)

# %%
# Native mobile exports (keep if you prefer platform-native inference)
tflite_path = best_model.export(
    format="tflite",
    imgsz=640,
    int8=True,
    data=str(data_yaml),
    nms=True,
)
coreml_path = best_model.export(
    format="coreml",
    imgsz=640,
    int8=True,
    data=str(data_yaml),
    nms=True,
)
print("Android TFLite:", tflite_path)
print("iOS CoreML:",     coreml_path)

# %%
# Smoke test
test_images = sorted((DATASET_ROOT / "test/images").glob("*.jpg"))
pred = best_model.predict(                     # ✅ best_model
    source=str(test_images[0]),
    conf=0.25,
    save=True,
)
print("Predicted:", test_images[0])

# %%
print(f"Outputs at: {results.save_dir}")
!ls -lah {results.save_dir}