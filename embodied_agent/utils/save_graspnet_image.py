"""
Save Gemini segmentation results in Contact-GraspNet format.

Contact-GraspNet expects:
- 'depth': (H, W) float32 depth in meters  
- 'segmap': (H, W) int32 with object IDs (1, 2, 3...), 0 for background
- 'K': (9,) flat camera intrinsics [fx, 0, cx, 0, fy, cy, 0, 0, 1]
- 'rgb': (H, W, 3) uint8

NOTE: This function is always called via _state["last_segmentation"] from tools.py,
so the numpy mask arrays are always real - never serialized strings.
"""

import numpy as np
import os


def save_graspnet_image(
    segmentation_results: dict,
    rgbd_path: str = "captures/rgbd/rgbd_image.npz",
    output_path: str = "/ros-ai-agent/captures/segmentation/rgbd_sgmtd/rgbd_sgmtd.npz"
) -> dict:
    """
    Save segmentation in Contact-GraspNet format with integer object IDs.
    
    Args:
        segmentation_results: Output from GeminiSegmentor.segment() with real numpy masks
        rgbd_path: Path to original RGBD .npz
        output_path: Where to save for Contact-GraspNet
        
    Returns:
        {"success": bool, "path": str, "num_objects": int}
    """
    objects = segmentation_results.get("objects", [])
    
    if not objects:
        return {
            "success": False,
            "error": "No objects found in segmentation results"
        }
    
    # Load original RGBD data
    try:
        data = np.load(rgbd_path)
        rgb   = data["rgb"]
        depth = data["depth"]
        K_raw = data["K"]
    except Exception as e:
        return {
            "success": False,
            "error": f"Could not load RGBD file '{rgbd_path}': {e}"
        }
    
    H, W = rgb.shape[:2]
    
    # Validate depth shape
    if depth.ndim == 1:
        if depth.size == H * W:
            depth = depth.reshape(H, W)
        else:
            return {
                "success": False,
                "error": f"Depth size {depth.size} doesn't match RGB {H}x{W}"
            }
    
    # Convert K to flat 9-element array if needed
    if K_raw.ndim == 2 and K_raw.shape == (3, 3):
        K_flat = K_raw.flatten()
    elif K_raw.ndim == 1 and K_raw.size == 9:
        K_flat = K_raw
    else:
        return {
            "success": False,
            "error": f"K has unexpected shape: {K_raw.shape}"
        }
    
    # Build segmentation map: 0=background, 1=first object, 2=second, etc.
    segmap = np.zeros((H, W), dtype=np.int32)
    num_valid = 0

    for i, obj in enumerate(objects):
        mask = obj.get("mask")

        # Validate mask is a real numpy bool array
        if mask is None:
            continue
        if not isinstance(mask, np.ndarray):
            continue
        if mask.shape != (H, W):
            continue

        object_id = i + 1
        segmap[mask] = object_id
        num_valid += 1

    if num_valid == 0:
        # Fallback: use bounding boxes if all masks were invalid
        for i, obj in enumerate(objects):
            box = obj.get("box_pixels")
            if box:
                x0, y0 = int(box["x_min"]), int(box["y_min"])
                x1, y1 = int(box["x_max"]), int(box["y_max"])
                segmap[y0:y1, x0:x1] = i + 1
                num_valid += 1

    if num_valid == 0:
        return {
            "success": False,
            "error": "Could not build segmap: no valid masks or bounding boxes"
        }

    # Save to output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        depth=depth.astype(np.float32),
        segmap=segmap,
        K=K_flat,
        rgb=rgb
    )

    unique_ids = [int(x) for x in np.unique(segmap)]

    return {
        "success": True,
        "path": output_path,
        "num_objects": num_valid,
        "segmap_shape": list(segmap.shape),
        "unique_ids": unique_ids,
    }