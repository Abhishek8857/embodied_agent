import os
import json
import cv2

from io import BytesIO

import numpy as np
from PIL import Image

from google import genai
from google.genai import types
from .utils import get_gemini_api_key


class GeminiSegmentor:
    """
    Open-vocabulary object segmentation using Gemini Robotics model.

    Uses gemini-robotics-er-1.6-preview which returns point-based detections
    (not bounding boxes + masks). Each point is expanded into a circular mask
    (~5% of the smallest image dimension) and back-projected through depth to
    produce 3D grasp points.

    Usage:
        segmentor = GeminiSegmentor()
        results = segmentor.segment("captures/rgbd/rgbd_image.npz", "blue cube")
    """

    MODEL_NAME = "gemini-robotics-er-1.6-preview"

    def __init__(self):
        self._client = genai.Client(api_key=get_gemini_api_key())

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def segment(self, npz_path: str, query: str,
                save_visualizations: bool = True,
                output_dir: str = "captures/segmentation",
                save_npz_with_segmap: bool = False,
                output_npz_path: str | None = None) -> dict:
        """
        Full pipeline: load .npz → query Gemini for points → build masks → 3D back-project.

        Args:
            npz_path: Path to .npz with 'rgb', 'depth', 'K'
            query: What to segment, e.g. "blue cube", "red objects"
            save_visualizations: If True, saves mask, overlay, segmented images
            output_dir: Where to save visualization outputs
            save_npz_with_segmap: If True, saves segmap alongside the rgbd data
            output_npz_path: Override path for the segmap .npz

        Returns:
            {
                "objects": [
                    {
                        "label": str,
                        "box_pixels": {"x_min", "y_min", "x_max", "y_max"},
                        "mask": np.ndarray (H, W) bool,
                        "grasp_center_3d": {"x", "y", "z"} | None,
                        "placement_surface_3d": {"x", "y", "z"} | None,
                    },
                    ...
                ],
                "count": int,
                "visualizations": { ... }  # only if save_visualizations=True
            }
        """
        rgb, depth, K = self._load_npz(npz_path)
        points_data = self._query_gemini(rgb, query)

        if not points_data:
            return {
                "objects": [],
                "count": 0,
                "message": f"No objects matching '{query}' were detected.",
            }

        results = self._process_points(rgb, depth, K, points_data)

        response = {"objects": results, "count": len(results)}

        if results:
            segmap = self._make_combined_segmap(results)
            response["segmap_shape"] = list(segmap.shape)
            response["segmap_unique"] = [int(x) for x in np.unique(segmap)]

            if save_npz_with_segmap:
                out_npz = output_npz_path or os.path.join(
                    output_dir, "rgbd_sgmtd", "rgbd_sgmtd.npz"
                )
                saved_path = self._save_npz_with_segmap(npz_path, segmap, out_npz)
                response["npz_with_segmap_path"] = saved_path

        if save_visualizations and results:
            response["visualizations"] = self._save_visualizations(rgb, results, output_dir)

        return response


    @staticmethod
    def _load_npz(npz_path: str):
        """Load and validate RGBD data from .npz."""
        data = np.load(npz_path)
        rgb = data["rgb"]
        depth = data["depth"]
        K_raw = data["K"]

        if K_raw.ndim == 1 and K_raw.size == 9:
            K = K_raw.reshape(3, 3)
        elif K_raw.shape == (3, 3):
            K = K_raw
        else:
            raise ValueError(f"K has unexpected shape: {K_raw.shape}")

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
        """
        Query Gemini 1.6 for object points.

        The model returns a JSON array of {"point": [y, x], "label": "..."}
        where coordinates are normalised to 0–1000.
        """
        image = Image.fromarray(rgb, mode="RGB")
        image.thumbnail([1024, 1024], Image.Resampling.LANCZOS)

        # 1.6 requires bytes, not a PIL object
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

        prompt = (
            f'Get all points matching the following objects: {query}. '
            f'The label returned should be an identifying name for the object detected. '
            f'The answer should follow the json format: '
            f'[{{"point": [y, x], "label": "..."}}]. '
            f'The points are in [y, x] format normalized to 0-1000. '
            f'Return ONLY the JSON array, no markdown, no explanation.'
        )

        config = types.GenerateContentConfig(
            temperature=1.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        response = self._client.models.generate_content(
            model=self.MODEL_NAME,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt,
            ],
            config=config,
        )

        return self._parse_response(response.text)


    def _process_points(self, rgb: np.ndarray, depth: np.ndarray,
                        K: np.ndarray, points_data: list[dict],
                        depth_tol_m: float = 0.03,
                        color_tol_lab: float = 30.0) -> list[dict]:
        """
        Convert Gemini point detections to object-shaped masks.
 
        Strategy
        --------
        Candidates must satisfy ALL three constraints simultaneously:
          1. Within a moderate search window (~15 % of min dimension) of seed.
          2. Depth within ±depth_tol_m of the seed pixel's depth.
          3. Color (CIE-Lab) within color_tol_lab of the seed pixel's color.
 
        Using color prevents the flood fill from spilling onto flat surfaces
        (table, floor) that share the same depth as the object's base pixels.
        Connected-components then isolates the object's own blob.
 
        Args:
            depth_tol_m:    Depth tolerance in metres  (default 3 cm).
            color_tol_lab:  CIE-Lab ΔE tolerance       (default 30). Lower = stricter.
        """
        import cv2
 
        H, W = rgb.shape[:2]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
 
        # Convert full image to Lab once (float32, L in [0,100])
        rgb_u8  = rgb.astype(np.uint8)
        lab_img = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2Lab).astype(np.float32)
 
        results = []
        for item in points_data:
            point = item.get("point")
            if not point or len(point) < 2:
                continue
 
            # 1. De-normalise 0–1000 → pixel coordinates
            py = int(np.clip(point[0] / 1000.0 * H, 0, H - 1))
            px = int(np.clip(point[1] / 1000.0 * W, 0, W - 1))
 
            z_seed = float(depth[py, px])
 
            # Sample seed color from a small patch (5 px radius) so we don't
            # land on a single highlight / shadow pixel
            ph = 5
            y0p = max(0, py - ph);  y1p = min(H, py + ph + 1)
            x0p = max(0, px - ph);  x1p = min(W, px + ph + 1)
            lab_seed = np.median(
                lab_img[y0p:y1p, x0p:x1p].reshape(-1, 3), axis=0
            )  # shape (3,)
 
            full_mask = None
 
            if z_seed > 0 and np.isfinite(z_seed):
                # 2. Search window — moderate radius keeps bleed-out in check
                search_radius = int(min(H, W) * 0.15)
                ys_grid, xs_grid = np.ogrid[:H, :W]
                in_radius = (
                    (xs_grid - px) ** 2 + (ys_grid - py) ** 2
                ) <= search_radius ** 2
 
                # 3. Depth gate
                depth_ok = (
                    np.abs(depth - z_seed) <= depth_tol_m
                ) & (depth > 0) & np.isfinite(depth)
 
                # 4. Color gate (Euclidean ΔE in Lab space)
                delta_lab   = lab_img - lab_seed           # (H, W, 3)
                color_dist  = np.linalg.norm(delta_lab, axis=2)  # (H, W)
                color_ok    = color_dist <= color_tol_lab
 
                candidate = (in_radius & depth_ok & color_ok).astype(np.uint8)
 
                # 5. Connected components → keep seed component
                _, labels = cv2.connectedComponents(candidate)
                seed_label = int(labels[py, px])
 
                if seed_label > 0:
                    full_mask = labels == seed_label
 
            # Fallback: small circle when depth / color unavailable
            if full_mask is None or not np.any(full_mask):
                fallback_radius = int(min(H, W) * 0.05)
                ys_grid, xs_grid = np.ogrid[:H, :W]
                full_mask = (
                    (xs_grid - px) ** 2 + (ys_grid - py) ** 2
                ) <= fallback_radius ** 2
 
            # Bounding box from actual mask extent
            ys_m, xs_m = np.where(full_mask)
            x0, x1 = int(xs_m.min()), int(xs_m.max())
            y0, y1 = int(ys_m.min()), int(ys_m.max())
 
            grasp_3d = self._compute_grasp_center(full_mask, depth, fx, fy, cx, cy)
            if grasp_3d is None:
                continue
 
            placement_3d = self._compute_placement_surface(
                full_mask, depth, fx, fy, cx, cy
            )
 
            results.append({
                "label":                item.get("label", ""),
                "box_pixels":           {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1},
                "mask":                 full_mask,
                "mask_array":           full_mask.astype(np.uint8) * 255,
                "grasp_center_3d":      grasp_3d,
                "placement_surface_3d": placement_3d,
            })
 
        return results

    @staticmethod
    def _compute_grasp_center(mask: np.ndarray, depth: np.ndarray,
                               fx: float, fy: float,
                               cx: float, cy: float) -> dict | None:
        """Back-project mask pixels through depth to get 3D centroid."""
        ys, xs = np.where(mask)
        if ys.size == 0:
            return None

        zs = depth[ys, xs]
        valid = (zs > 0) & np.isfinite(zs)
        if not valid.any():
            return None

        xs_v, ys_v, zs_v = xs[valid], ys[valid], zs[valid]

        X = (xs_v - cx) * zs_v / fx
        Y = (ys_v - cy) * zs_v / fy
        Z = zs_v

        return {
            "x": round(float(X.mean()), 4),
            "y": round(float(Y.mean()), 4),
            "z": round(float(Z.mean()), 4),
        }

    def _compute_placement_surface(self, mask: np.ndarray, depth: np.ndarray,
                                    fx: float, fy: float,
                                    cx: float, cy: float,
                                    surface_depth_percentile: float = 5.0) -> dict:
        """
        Stable placement point in camera frame.

        Uses a low depth percentile (nearest surface) suited to downward-facing
        cameras.  Adjust surface_depth_percentile for other orientations.
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

        u = float(np.median(xs_v))
        v = float(np.median(ys_v))
        z = float(np.percentile(zs_v, surface_depth_percentile))

        return {
            "x": float((u - cx) * z / fx),
            "y": float((v - cy) * z / fy),
            "z": float(z),
        }


    @staticmethod
    def _save_visualizations(rgb: np.ndarray, results: list[dict],
                              output_dir: str) -> dict:
        """Save mask, overlay, and segmented images."""
        output_dir = os.path.join(output_dir, "masks")
        os.makedirs(output_dir, exist_ok=True)

        H, W = rgb.shape[:2]
        individual_masks = []

        for i, res in enumerate(results):
            obj_mask = res["mask"].astype(np.uint8) * 255
            mask_path = os.path.join(output_dir, "mask.png")
            Image.fromarray(obj_mask, mode="L").save(mask_path)

            obj_segmented = np.where(res["mask"][:, :, None], rgb, 0)
            seg_path = os.path.join(output_dir, "segmented.png")
            Image.fromarray(obj_segmented.astype(np.uint8), mode="RGB").save(seg_path)

            individual_masks.append({
                "label": res["label"],
                "mask_path": mask_path,
                "segmented_path": seg_path,
            })

        # Combined binary mask
        combined_mask = np.zeros((H, W), dtype=np.uint8)
        for res in results:
            combined_mask = np.maximum(combined_mask, res["mask"].astype(np.uint8) * 255)

        # Coloured segmentation
        colored_seg = np.zeros((H, W, 3), dtype=np.uint8)
        for res in results:
            colored_seg = np.where(res["mask"][:, :, None], rgb, colored_seg)

        # Green overlay
        overlay = rgb.copy()
        for res in results:
            mask = res["mask"]
            overlay[mask, 1] = np.clip(overlay[mask, 1] * 0.6 + 255 * 0.4, 0, 255).astype(np.uint8)
            overlay[mask, 0] = (overlay[mask, 0] * 0.6).astype(np.uint8)
            overlay[mask, 2] = (overlay[mask, 2] * 0.6).astype(np.uint8)

        combined_mask_path = os.path.join(output_dir, "mask_combined.png")
        combined_seg_path  = os.path.join(output_dir, "segmented_combined.png")
        overlay_path       = os.path.join(output_dir, "overlay.png")

        Image.fromarray(combined_mask, mode="L").save(combined_mask_path)
        Image.fromarray(colored_seg, mode="RGB").save(combined_seg_path)
        Image.fromarray(overlay, mode="RGB").save(overlay_path)

        return {
            "individual_masks": individual_masks,
            "combined_mask_path": combined_mask_path,
            "combined_segmented_path": combined_seg_path,
            "overlay_path": overlay_path,
        }


    @staticmethod
    def _make_combined_segmap(results: list[dict]) -> np.ndarray:
        """Combine all object masks into a single binary segmap (0=bg, 1=object)."""
        combined = None
        for r in results:
            m = r["mask"]
            combined = m.copy() if combined is None else (combined | m)
        if combined is None:
            return np.zeros((1, 1), dtype=np.uint8)
        return combined.astype(np.uint8)

    @staticmethod
    def _save_npz_with_segmap(original_npz_path: str,
                               segmap: np.ndarray,
                               out_npz_path: str) -> str:
        """Load original .npz and save a new one with segmap added."""
        data = dict(np.load(original_npz_path, allow_pickle=False))

        if "rgb" in data:
            H, W = data["rgb"].shape[:2]
            if segmap.shape != (H, W):
                raise ValueError(
                    f"segmap shape {segmap.shape} does not match rgb shape {(H, W)}"
                )

        data["segmap"] = segmap.astype(np.uint8)
        os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)
        np.savez_compressed(out_npz_path, **data)
        return out_npz_path


    @staticmethod
    def _parse_response(text: str) -> list[dict]:
        """Parse Gemini 1.6 JSON response, stripping any markdown fences."""
        raw = text.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            return []