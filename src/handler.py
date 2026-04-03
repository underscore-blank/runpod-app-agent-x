"""
AppAgentX RunPod Serverless Worker
Routes between two services based on job["input"]["service"]:
  - "omni"  : OmniParser screen parsing
  - "embed" : Image feature extraction
"""

import base64
import io
import os
import tempfile
import time
from typing import List

import runpod
import torch

# ── Shared config ─────────────────────────────────────────────────────────────
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/weights")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── OmniParser — load once at startup ────────────────────────────────────────
from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
import pandas as pd

YOLO_PATH = f"{WEIGHTS_DIR}/icon_detect_v1_5/best.pt"
FLORENCE_PATH = f"{WEIGHTS_DIR}/icon_caption_florence"

print("Loading OmniParser models...")
som_model = get_yolo_model(model_path=YOLO_PATH)
som_model.to(DEVICE)
caption_model_processor = get_caption_model_processor(
    model_name="florence2",
    model_name_or_path=FLORENCE_PATH,
    device=DEVICE,
)
print("OmniParser ready.")

# ── ImageEmbedding — lazy model cache ────────────────────────────────────────
import timm
from torchvision import transforms
from PIL import Image

_embed_cache = {}

MODELS_CONFIG = {
    "resnet50":                      {"image_size": 224},
    "vit_base_patch16_224":          {"image_size": 224},
    "efficientnet_b0":               {"image_size": 224},
    "efficientnet_b4":               {"image_size": 380},
    "swin_base_patch4_window7_224":  {"image_size": 224},
    "convnext_base":                 {"image_size": 224},
    "eva02_base_patch14_448":        {"image_size": 448},
}

def _get_embed_model(model_name: str):
    if model_name not in _embed_cache:
        model = timm.create_model(model_name, pretrained=True, num_classes=0).to(DEVICE)
        model.eval()
        sz = MODELS_CONFIG.get(model_name, {}).get("image_size", 224)
        transform = transforms.Compose([
            transforms.Resize(sz),
            transforms.CenterCrop(sz),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        _embed_cache[model_name] = (model, transform)
    return _embed_cache[model_name]

def _extract_features(image_b64: str, model_name: str) -> List[float]:
    model, transform = _get_embed_model(model_name)
    img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = model(tensor)
    return features.cpu().numpy().tolist()[0]


# ── Handlers ──────────────────────────────────────────────────────────────────

def handle_omni(job_input: dict) -> dict:
    image_bytes = base64.b64decode(job_input["image"])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    box_threshold = job_input.get("box_threshold", 0.05)
    iou_threshold = job_input.get("iou_threshold", 0.1)
    imgsz = job_input.get("imgsz", 640)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    try:
        start = time.time()

        ocr_result, _ = check_ocr_box(
            tmp_path,
            display_img=False,
            output_bb_format="xyxy",
            goal_filtering=None,
            easyocr_args={"paragraph": False, "text_threshold": 0.5},
            use_paddleocr=False,
        )
        text, ocr_bbox = ocr_result

        draw_bbox_config = {
            "text_scale":     0.8 * (max(image.size) / 3200),
            "text_thickness": max(int(2 * (max(image.size) / 3200)), 1),
            "text_padding":   max(int(3 * (max(image.size) / 3200)), 1),
            "thickness":      max(int(3 * (max(image.size) / 3200)), 1),
        }

        labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            tmp_path,
            som_model,
            BOX_TRESHOLD=box_threshold,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=caption_model_processor,
            ocr_text=text,
            use_local_semantics=True,
            iou_threshold=iou_threshold,
            scale_img=False,
            batch_size=128,
            imgsz=imgsz,
        )

        df = pd.DataFrame(parsed_content_list)
        df["ID"] = range(len(df))

        return {
            "status": "success",
            "parsed_content": df.to_dict(orient="records"),
            "labeled_image": labeled_img,
            "e_time": time.time() - start,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        os.unlink(tmp_path)


def handle_embed(job_input: dict) -> dict:
    model_name = job_input.get("model_name", "resnet50")
    start = time.time()
    try:
        if "image" in job_input:
            features = _extract_features(job_input["image"], model_name)
            return {
                "features": [features],
                "time_taken": time.time() - start,
                "shape": [1, len(features)],
                "model_name": model_name,
            }
        if "images" in job_input:
            all_features = [_extract_features(img, model_name) for img in job_input["images"]]
            return {
                "features": all_features,
                "time_taken": time.time() - start,
                "shape": [len(all_features), len(all_features[0])],
                "model_name": model_name,
            }
        return {"error": "Provide 'image' (single) or 'images' (batch) in input."}
    except Exception as e:
        return {"error": str(e)}


# ── Main router ───────────────────────────────────────────────────────────────

def handler(job):
    job_input = job["input"]
    service = job_input.get("service")

    if service == "omni":
        return handle_omni(job_input)
    elif service == "embed":
        return handle_embed(job_input)
    else:
        return {"error": f"Unknown service '{service}'. Use 'omni' or 'embed'."}


runpod.serverless.start({"handler": handler})
