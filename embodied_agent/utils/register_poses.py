"""
pose_registry.py
----------------
Manages a JSON file of named robot poses (joint states).

File format (poses.json):
{
  "home": {
    "joints": [0.049, -0.4882, 3.1227, -2.0745, 0.0112, -0.9870, 1.55],
    "description": "Safe home position",
    "saved_at": "2026-03-27T12:00:00"
  },
  "retract": {
    "joints": [0.0, 0.0, 3.1227, -1.5, 0.0, -1.6, 1.55],
    "description": "Retract pose above workspace",
    "saved_at": "2026-03-27T12:01:00"
  }
}
"""

import json
import os
from datetime import datetime
from typing import Optional


DEFAULT_PATH = "poses/poses.json"


class RegisterPoses:

    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        self._poses: dict = {}
        self._load()

    # ── Private ───────────────────────────────────────────────────────────────
    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    raw = json.load(f)
                # Normalise keys on load so manually edited files always work
                self._poses = {
                    k.strip().lower().replace(" ", "_"): v
                    for k, v in raw.items()
                }
            except (json.JSONDecodeError, OSError) as e:
                print(f"[PoseRegistry] Failed to load {self.path}: {e}. Starting empty.")
                self._poses = {}
        else:
            self._poses = {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._poses, f, indent=2)

    # ── Public API ────────────────────────────────────────────────────────────

    def save_pose(self, name: str, joints: list[float], description: str = "") -> dict:
        """Save or overwrite a named pose."""
        name = name.strip().lower().replace(" ", "_")
        self._poses[name] = {
            "joints":      [float(j) for j in joints],
            "description": description,
            "saved_at":    datetime.now().isoformat()[:19],
        }
        self._save()
        return {"success": True, "name": name, "joints": self._poses[name]["joints"]}

    def get_pose(self, name: str) -> Optional[dict]:
        """Return the pose dict for a given name, or None if not found."""
        return self._poses.get(name.strip().lower().replace(" ", "_"))

    def get_joints(self, name: str) -> Optional[list[float]]:
        """Return just the joint list for a given name, or None if not found."""
        pose = self.get_pose(name)
        return pose["joints"] if pose else None

    def delete_pose(self, name: str) -> dict:
        """Delete a named pose."""
        name = name.strip().lower().replace(" ", "_")
        if name not in self._poses:
            return {"success": False, "error": f"Pose '{name}' not found."}
        del self._poses[name]
        self._save()
        return {"success": True, "deleted": name}

    def list_poses(self) -> dict:
        """Return all pose names with their descriptions and joint counts."""
        return {
            name: {
                "description": entry.get("description", ""),
                "num_joints":  len(entry.get("joints", [])),
                "saved_at":    entry.get("saved_at", ""),
            }
            for name, entry in self._poses.items()
        }

    def rename_pose(self, old_name: str, new_name: str) -> dict:
        """Rename a pose."""
        old = old_name.strip().lower().replace(" ", "_")
        new = new_name.strip().lower().replace(" ", "_")
        if old not in self._poses:
            return {"success": False, "error": f"Pose '{old}' not found."}
        if new in self._poses:
            return {"success": False, "error": f"Pose '{new}' already exists."}
        self._poses[new] = self._poses.pop(old)
        self._save()
        return {"success": True, "renamed": f"{old} -> {new}"}