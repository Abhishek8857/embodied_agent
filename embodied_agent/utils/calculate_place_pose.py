import numpy as np
from scipy.spatial.transform import Rotation



def get_placement_pose(
    segmentation_results: dict,
    target_object_label: str = None,
    object_index: int = 0,
    height_offset: float = 0.23,  # Reduced for better accuracy (was 0.2)
    gripper_orientation: str = "downward",
    tf_transform: dict = None,
    apply_tf: bool = True,
) -> dict:
    """
    Extract placement pose from segmentation results.
    
    Args:
        segmentation_results: Output from segment_objects tool
        target_object_label: Label to search for (e.g., "red cube"). If None, uses object_index.
        object_index: Which object to use if label not specified (default: 0 = first object)
        height_offset: How high above the surface to place (meters, default: 0.03 = 3cm)
        gripper_orientation: "downward" or "identity" - how gripper should be oriented
        tf_transform: Transform from base_link to camera frame
        apply_tf: Whether to apply the transform
    
    Returns:
        {
            "success": bool,
            "x": float,
            "y": float,
            "z": float,  # surface + height_offset
            "qx": float,
            "qy": float,
            "qz": float,
            "qw": float,
            "object_label": str,
            "surface_z": float,  # Original surface height
        }
    """
    objects = segmentation_results.get("objects", [])
    
    if not objects:
        return {
            "success": False,
            "error": "No objects found in segmentation results"
        }
    
    # Find target object
    target_obj = None
    if target_object_label:
        target_label_lower = target_object_label.lower()
        for obj in objects:
            if target_label_lower in obj["label"].lower():
                target_obj = obj
                break
        if target_obj is None:
            available = [o["label"] for o in objects]
            return {
                "success": False,
                "error": f"No object matching '{target_object_label}' found. Available: {available}"
            }
    else:
        if object_index >= len(objects):
            return {
                "success": False,
                "error": f"object_index {object_index} out of range (only {len(objects)} objects found)"
            }
        target_obj = objects[object_index]
    
    # Get the placement surface pose
    surface_pose = target_obj.get("placement_surface_3d")
    if surface_pose is None:
        return {
            "success": False,
            "error": f"Object '{target_obj['label']}' has no placement_surface_3d. Did segmentation fail?"
        }
    

    
    # Extract camera-frame position
    x = float(surface_pose["x"]) 
    y = float(surface_pose["y"])
    z_surface = float(surface_pose["z"])


    # Apply TF transform if requested
    if apply_tf:
        if tf_transform is None or not tf_transform.get("success", False):
            return {"success": False, "error": "TF transform missing/invalid but apply_tf=True"}

        t = tf_transform["translation"]
        q = tf_transform["quaternion"]

        # Use DIRECT transform, not inverse
        # The TF appears to be camera to base, so: p_base = R @ p_cam + t
        R_transform = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]])
        t_transform = np.array([t["x"], t["y"], t["z"]], dtype=float)
        
        p_cam = np.array([x, y, z_surface], dtype=float)
        
        print(f"[DEBUG] Camera optical frame: x={x:.4f}, y={y:.4f}, z={z_surface:.4f}")
        print(f"[DEBUG] TF translation: {t_transform}")
        
        # Apply transform
        p_base = R_transform.apply(p_cam) + t_transform
        x, y, z_surface = float(p_base[0]), float(p_base[1]), float(p_base[2])
        
        
        X_OFFSET = -0.0125
        Y_OFFSET_LEFT = 0.028
        Y_OFFSET_RIGHT = 0.015
        
        
        x += X_OFFSET
        
        if y > 0.0:
            y -= Y_OFFSET_LEFT
        else:
            y -= Y_OFFSET_RIGHT
            
            
        # Sanity check: Expand lower bound to -0.20m to allow surfaces at/below base level
        if x < -0.2 or x > 0.8:
            print(f"[WARNING] X={x:.4f} is outside typical reach range [-0.2, 0.8]")
        if abs(y) > 0.6:
            print(f"[WARNING] Y={y:.4f} is outside typical lateral range [-0.6, 0.6]")
        if z_surface < -0.20 or z_surface > 1.0:
            print(f"[WARNING] Unusual z_surface={z_surface:.4f}")
            return {
                "success": False,
                "error": f"Z={z_surface:.4f} is outside expected range [-0.20, 1.0]m"
            }

    # Add height offset
    z_place = z_surface + height_offset

    # Determine Orientation
    if gripper_orientation == "downward":
        # Standard downward gripper (180° around X-axis)
        R_downward = Rotation.from_euler('x', 180, degrees=True)
        quat = R_downward.as_quat()
        
    elif gripper_orientation == "downward_angled":
        # Less extreme downward (135° around X-axis) - often better for motion planning
        R_downward = Rotation.from_euler('x', 135, degrees=True)
        quat = R_downward.as_quat()
        
    elif gripper_orientation == "side_grasp":
        # Approach from side (90° around Y-axis)
        R_side = Rotation.from_euler('y', 90, degrees=True)
        quat = R_side.as_quat()
        
    elif gripper_orientation == "angled_approach":
        # 45° angled approach
        R_angled = Rotation.from_euler('xy', [45, 0], degrees=True)
        quat = R_angled.as_quat()
        
    elif gripper_orientation == "identity":
        quat = [0, 0, 0, 1]
        
    else:
        return {"success": False, "error": f"Unknown gripper_orientation: {gripper_orientation}"}

    return {
        "success": True,
        "x": round(x, 4),
        "y": round(y, 4),
        "z": round(z_place, 4),
        "qx": round(float(quat[0]), 4),
        "qy": round(float(quat[1]), 4),
        "qz": round(float(quat[2]), 4),
        "qw": round(float(quat[3]), 4),
        "object_label": target_obj["label"],
        "surface_z": round(z_surface, 4),
    }