import os
import json
import base64
import shutil
from io import BytesIO

import numpy as np
from PIL import Image

from google import genai
from google.genai import types
from .utils import get_gemini_api_key

class GeminiSegmentor:
    """
    Open-vocabulary object segmentation using Gemini Robotics model.

    Uses gemini-robotics-er-1.5-preview which returns pixel-level
    segmentation masks (not just bounding boxes). Loads RGBD from .npz,
    segments the query object(s), and back-projects through depth to
    produce 3D grasp points.

    Usage:
        segmentor = GeminiSegmentor()
        results = segmentor.segment("captures/rgbd/rgbd_image.npz", "blue cube")
    """

    MODEL_NAME = "gemini-robotics-er-1.5-preview"

    def __init__(self):
        self._client = genai.Client(api_key=get_gemini_api_key())


    def segment(self, npz_path: str, query: str,
                save_visualizations: bool = True,
                output_dir: str = "captures/segmentation",
                save_npz_with_segmap: bool = False,
                output_npz_path: str | None = None) -> dict:
        """
        Full pipeline: load .npz then segment with Gemini to get the 3D back-projection.

        Args:
            npz_path: Path to .npz with 'rgb', 'depth', 'K'
            query: What to segment, e.g. "blue cube", "red objects"
            save_visualizations: If True, saves mask, overlay, segmented images
            output_dir: Where to save visualization outputs
            save_npz_with_segmap: If True, saves segmap with the rgbd image for further processing
            output_npz_path: Path to .npz with 'segmap', 'rgb', 'depth', 'K'

        Returns:
            {
                "objects": [
                    {
                        "label": str,
                        "box_pixels": {"x_min", "y_min", "x_max", "y_max"},
                        "mask": np.ndarray (H, W) bool,
                        "grasp_center_3d": {"x", "y", "z"} | None
                    },
                    ...
                ],
                "count": int,
                "visualizations": {
                    "mask_path": str,
                    "overlay_path": str,
                    "segmented_path": str
                } (only if save_visualizations=True)
            }
        """
        rgb, depth, K = self._load_npz(npz_path)
        masks_data    = self._query_gemini(rgb, query)

        if not masks_data:
            return {
                "objects": [],
                "count": 0,
                "message": f"No objects matching '{query}' were detected."
            }

        results = self._process_masks(rgb, depth, K, masks_data)

        response = {"objects": results, "count": len(results)}
        
        if results:
                segmap = self._make_combined_segmap(results)  # uint8 (H,W)
                response["segmap_shape"] = list(segmap.shape)
                response["segmap_unique"] = [int(x) for x in np.unique(segmap)]

                if save_npz_with_segmap:
                    out_npz = output_npz_path
                    if out_npz is None:
                        # default next to output_dir
                        out_npz = os.path.join(output_dir, "rgbd_sgmtd", "rgbd_sgmtd.npz")
                    saved_path = self._save_npz_with_segmap(npz_path, segmap, out_npz)
                    response["npz_with_segmap_path"] = saved_path

        if save_visualizations and results:
            viz_paths = self._save_visualizations(
                rgb, results, output_dir
            )
            response["visualizations"] = viz_paths

        return response


    @staticmethod
    def _load_npz(npz_path: str):
        """Load and validate RGBD data from .npz."""
        data  = np.load(npz_path)
        rgb   = data["rgb"]
        depth = data["depth"]
        K_raw = data["K"]

        # K is saved as a 1D array (9,) from CameraInfo.k
        # Need to reshape to (3, 3)
        if K_raw.ndim == 1 and K_raw.size == 9:
            K = K_raw.reshape(3, 3)
        elif K_raw.shape == (3, 3):
            K = K_raw
        else:
            raise ValueError(f"K has unexpected shape: {K_raw.shape}")

        # Reshape depth if it's flattened 
        H, W = rgb.shape[:2]
        if depth.ndim == 1:
            if depth.size == H * W:
                depth = depth.reshape(H, W)
            else:
                raise ValueError(
                    f"Depth array is 1D with size {depth.size}, "
                    f"but expected {H * W} (H={H}, W={W})"
                )
        elif depth.shape != (H, W):
            raise ValueError(
                f"Depth shape {depth.shape} doesn't match RGB shape ({H}, {W})"
            )

        return rgb, depth, K


    def _query_gemini(self, rgb: np.ndarray, query: str) -> list[dict]:
        """Query Gemini robotics model for segmentation masks."""
        # Resize for API (max 1024x1024)
        image = Image.fromarray(rgb, mode="RGB")
        image.thumbnail([1024, 1024], Image.Resampling.LANCZOS)

        prompt = (
            f'Give the segmentation masks for only the {query} that you can identify.\n'
            f'Output a JSON list where each entry contains:\n'
            f'- "box_2d": bounding box [y_min, x_min, y_max, x_max] in 0-1000 coords\n'
            f'- "mask": segmentation mask as base64 PNG\n'
            f'- "label": descriptive text label\n'
            f'Only include objects matching: {query}'
        )

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )

        response = self._client.models.generate_content(
            model=self.MODEL_NAME,
            contents=[prompt, image],
            config=config
        )

        return self._parse_response(response.text)


    def _process_masks(self, rgb: np.ndarray, depth: np.ndarray, 
                       K: np.ndarray, masks_data: list[dict]) -> list[dict]:
        """Convert Gemini masks to pixel masks + 3D grasp centers."""
        H, W          = rgb.shape[:2]
        fx, fy        = K[0, 0], K[1, 1]
        cx, cy        = K[0, 2], K[1, 2]

        results = []
        for item in masks_data:
            # Bounding box (0-1000 → pixels)
            box = item["box_2d"]
            y0 = int(box[0] / 1000 * H)
            x0 = int(box[1] / 1000 * W)
            y1 = int(box[2] / 1000 * H)
            x1 = int(box[3] / 1000 * W)

            if y0 >= y1 or x0 >= x1:
                continue

            # Decode mask
            png_str = item["mask"]
            if not png_str.startswith("data:image/png;base64,"):
                continue

            png_str   = png_str.removeprefix("data:image/png;base64,")
            mask_data = base64.b64decode(png_str)
            mask_img  = Image.open(BytesIO(mask_data))

            # Resize mask to bbox, then embed in full-res array
            mask_img = mask_img.resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR)
            mask_array = np.array(mask_img) > 128  # Binary threshold

            # Create full-size mask
            full_mask = np.zeros((H, W), dtype=bool)
            full_mask[y0:y1, x0:x1] = mask_array

            # Compute 3D grasp center (centroid of all pixels)
            grasp_3d = self._compute_grasp_center(full_mask, depth, fx, fy, cx, cy)
            
            # Compute 3D placement surface (top surface for placing on)
            placement_3d = self._compute_placement_surface(full_mask, depth, fx, fy, cx, cy)

            results.append({
                "label":              item["label"],
                "box_pixels":         {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1},
                "mask":               full_mask,           # Boolean numpy array (H, W)
                "mask_array":         full_mask.astype(np.uint8) * 255,  # uint8 for saving/viewing
                "grasp_center_3d":    grasp_3d,           # Centroid (for grasping)
                "placement_surface_3d": placement_3d,      # Top surface (for placing on)
            })

        return results


    @staticmethod
    def _compute_grasp_center(mask: np.ndarray, depth: np.ndarray,
                              fx: float, fy: float, cx: float, cy: float):
        """Back-project mask pixels through depth to get 3D centroid."""
        ys, xs = np.where(mask)
        if ys.size == 0:
            return None

        zs = depth[ys, xs]
        valid = (zs > 0) & np.isfinite(zs)
        if not valid.any():
            return None

        xs, ys, zs = xs[valid], ys[valid], zs[valid]

        # Back-project to 3D
        X = (xs - cx) * zs / fx
        Y = (ys - cy) * zs / fy
        Z = zs

        # Centroid
        return {
            "x": round(float(X.mean()), 4),
            "y": round(float(Y.mean()), 4),
            "z": round(float(Z.mean()), 4),
        }  


    def _compute_placement_surface(self, mask, depth, fx, fy, cx, cy,
                                    surface_depth_percentile: float = 5.0):
        """
        Stable placement point in CAMERA frame.
        
        For DOWNWARD-FACING cameras (looking down at workspace):
        - Use LOW percentile (5th) to get NEAREST surface
        - This is the TOP of the object (what camera sees first)
        - This is the correct placement surface
        
        For FORWARD-FACING cameras (looking horizontally):
        - Use HIGH percentile (95th) to get FARTHEST surface  
        - This is the BOTTOM/support surface of the object
        
        Note: The percentile should match your camera mounting orientation.
        """

        ys, xs = np.where(mask)
        if xs.size < 20:
            return {"x": 0.0, "y": 0.0, "z": 0.0}

        zs = depth[ys, xs].astype(np.float32)
        valid = np.isfinite(zs) & (zs > 0.05) & (zs < 5.0)

        if valid.sum() < 20:
            return {"x": 0.0, "y": 0.0, "z": 0.0}

        xs_v = xs[valid].astype(np.float32)
        ys_v = ys[valid].astype(np.float32)
        zs_v = zs[valid].astype(np.float32)

        # Stable image-space center
        u = float(np.median(xs_v))
        v = float(np.median(ys_v))

        # Robust depth - percentile depends on camera orientation
        z = float(np.percentile(zs_v, surface_depth_percentile))

        # Standard Pinhole Camera Model
        X = (u - cx) * z / fx
        Y = (v - cy) * z / fy
        Z = z

        return {"x": float(X), "y": float(Y), "z": float(Z)}

    @staticmethod
    def _save_visualizations(rgb: np.ndarray, results: list[dict], 
                            output_dir: str) -> dict:
        """Save mask, overlay, segmented images + individual masks per object."""
        output_dir = os.path.join(output_dir, "masks")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        
        H, W = rgb.shape[:2]
        image = Image.fromarray(rgb, mode="RGB")

        # Save individual masks for each object
        individual_masks = []
        for i, res in enumerate(results):
            # Binary mask for this object only
            obj_mask = (res["mask"].astype(np.uint8) * 255)
            mask_filename = "mask.png"
            mask_path = os.path.join(output_dir,  mask_filename)
            Image.fromarray(obj_mask, mode="L").save(mask_path)
            
            # Segmented RGB for this object only
            obj_segmented = np.where(res["mask"][:, :, None], rgb, 0)
            seg_filename = "segmented.png"
            seg_path = os.path.join(output_dir, seg_filename)
            Image.fromarray(obj_segmented.astype(np.uint8), mode="RGB").save(seg_path)
            
            individual_masks.append({
                "label": res["label"],
                "mask_path": mask_path,
                "segmented_path": seg_path
            })

        # Combined binary mask (all objects)
        combined_mask = np.zeros((H, W), dtype=np.uint8)
        for res in results:
            combined_mask = np.maximum(combined_mask, res["mask"].astype(np.uint8) * 255)

        # Colored segmentation (RGB only where any mask is True)
        colored_seg = np.zeros((H, W, 3), dtype=np.uint8)
        for res in results:
            mask_3ch = res["mask"][:, :, None]
            colored_seg = np.where(mask_3ch, rgb, colored_seg)

        # Overlay (RGB with green highlight on all masks)
        overlay = rgb.copy()
        for res in results:
            mask = res["mask"]
            overlay[mask, 1] = np.clip(overlay[mask, 1] * 0.6 + 255 * 0.4, 0, 255).astype(np.uint8)
            overlay[mask, 0] = (overlay[mask, 0] * 0.6).astype(np.uint8)
            overlay[mask, 2] = (overlay[mask, 2] * 0.6).astype(np.uint8)

        # Save combined outputs
        combined_mask_path = os.path.join(output_dir, "mask_combined.png")
        combined_seg_path  = os.path.join(output_dir, "segmented_combined.png")
        overlay_path       = os.path.join(output_dir, "overlay.png")

        Image.fromarray(combined_mask, mode="L").save(combined_mask_path)
        Image.fromarray(colored_seg, mode="RGB").save(combined_seg_path)
        Image.fromarray(overlay, mode="RGB").save(overlay_path)
        
        

        return {
            "individual_masks": individual_masks,  # List of {label, mask_path, segmented_path}
            "combined_mask_path": combined_mask_path,
            "combined_segmented_path": combined_seg_path,
            "overlay_path": overlay_path,
        }

    @staticmethod
    def _make_combined_segmap(results: list[dict]) -> np.ndarray:
        """
        Combine all object masks into a single binary segmap (0 background, 1 object).
        """
        # results[i]["mask"] is bool (H,W)
        combined = None
        for r in results:
            m = r["mask"]
            if combined is None:
                combined = m.copy()
            else:
                combined |= m
        if combined is None:
            # no results
            return np.zeros((1, 1), dtype=np.uint8)
        return combined.astype(np.uint8)


    @staticmethod
    def _save_npz_with_segmap(original_npz_path: str, segmap: np.ndarray, out_npz_path: str) -> str:
        """
        Load original npz and save a new npz with segmap added.
        """
        data = dict(np.load(original_npz_path, allow_pickle=False))

        # Basic sanity: match rgb size if present
        if "rgb" in data:
            H, W = data["rgb"].shape[:2]
            if segmap.shape != (H, W):
                raise ValueError(f"segmap shape {segmap.shape} does not match rgb shape {(H, W)}")

        data["segmap"] = segmap.astype(np.uint8)  # enforce 0/1 uint8

        os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)
        np.savez_compressed(out_npz_path, **data)
        return out_npz_path


    @staticmethod
    def _parse_response(text: str) -> list[dict]:
        """Parse Gemini JSON response, stripping markdown fences."""
        raw = text.strip()
        
        # Remove markdown code fences
        lines = raw.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "```json":
                raw = "\n".join(lines[i+1:])
                raw = raw.split("```")[0]
                break
        else:
            # Handle ```json on same line or just ```
            if raw.startswith("```"):
                raw = raw.split("```")[1].split("```")[0]
                if raw.startswith("json"):
                    raw = raw[4:]
        
        raw = raw.strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []