"""
SemanticKITTI Panoptic Renderer
================================
Renders 3D panoptic occupancy voxel grids to 2D images using pyrender.
Resolution: 1440 x 2048 (matching SemCity FID evaluation setup)

Loads SemanticKITTI panoptic format (uint32):
- Lower 16 bits: semantic class
- Upper 16 bits: instance ID

Uses position-based instance coloring for FID-compatible deterministic colors.

Usage
-----
# Render real SemanticKITTI panoptic scenes
python render_semantickitti_panoptic.py \
    --mode real \
    --sequences_dir /path/to/kitti/sequences \
    --output_dir /path/to/real_images

# Render generated panoptic scenes (from .label files)
python render_semantickitti_panoptic.py \
    --mode generated \
    --generated_dir /path/to/generated_labels \
    --output_dir /path/to/gen_images

# Then compute FID with torch-fidelity
pip install torch-fidelity
python -m torch_fidelity evaluate \
    --input1 /path/to/real_images \
    --input2 /path/to/gen_images \
    --fid --isc --prc
"""
import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
# os.environ["PYOPENGL_PLATFORM"] = "egl"  # uncomment for headless rendering

import trimesh
import pyrender
from colorsys import rgb_to_hsv, hsv_to_rgb
import pickle
# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
IMG_W = 4096
IMG_H = 2880
VOXEL_SIZE = 0.2
GRID_SHAPE = (256, 256, 32)

# SemanticKITTI 20-class base colors (RGB)
KITTI_COLORS = np.array([
    [0,   0,   0  ],  #  0  free / unlabeled
    # [100, 150, 245],  #  1  car
    [255, 0,   0  ],
    [100, 230, 245],  #  2  bicycle
    [30,  60,  150],  #  3  motorcycle
    [80,  30,  180],  #  4  truck
    [0,   0,   255],  #  5  other-vehicle
    [255, 30,  30 ],  #  6  person
    [255, 40,  200],  #  7  bicyclist
    [150, 30,  90 ],  #  8  motorcyclist
    [255, 0,   255],  #  9  road
    [255, 150, 255],  # 10  parking
    [75,  0,   75 ],  # 11  sidewalk
    [175, 0,   75 ],  # 12  other-ground
    [255, 200, 0  ],  # 13  building
    [255, 120, 50 ],  # 14  fence
    [0,   175, 0  ],  # 15  vegetation
    [135, 60,  0  ],  # 16  trunk
    [150, 240, 80 ],  # 17  terrain
    [255, 240, 150],  # 18  pole
    [100, 150, 245],  # 19  traffic-sign
], dtype=np.uint8)

# Thing classes (have instances)
THING_IDS = [1, 2, 3, 4, 5, 6, 7, 8]
THING_IDS_SET = set(THING_IDS)

learning_map = {
    0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5,
    30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13,
    51: 14, 52: 0, 60: 9, 70: 15, 71: 16, 72: 17, 80: 18, 81: 19,
    99: 0, 252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading — panoptic
# ─────────────────────────────────────────────────────────────────────────────
def load_kitti_panoptic(label_path: str) -> tuple:
    """
    Load a SemanticKITTI panoptic .label file (uint32 format).
    
    Returns
    -------
    semantic_labels : (256, 256, 32) np.int32 — class IDs in 0-19 range
    instance_labels : (256, 256, 32) np.int32 — instance IDs (0 = stuff/empty)
    """
    try:
        raw = np.fromfile(label_path, dtype=np.uint32)
        
        expected = np.prod(GRID_SHAPE)
        if raw.size != expected:
            raise ValueError(
                f"Size mismatch: {raw.size} elements vs expected {expected} "
                f"for grid {GRID_SHAPE}. Check uint16 vs uint32 dtype."
            )
        
        # Decompose: lower 16 bits = semantic, upper 16 bits = instance
        semantic_raw = (raw & 0xFFFF).astype(np.int64)
        instance_raw = (raw >> 16).astype(np.int64)
        
        # Apply learning_map to semantic
        if semantic_raw.max() > 19:
            semantic_raw = np.vectorize(learning_map.get)(semantic_raw)
        # semantic_remapped = np.vectorize(learning_map.get, otypes=[np.int32])(semantic_raw)
        
        semantic_labels = semantic_raw.reshape(GRID_SHAPE)
        instance_labels = instance_raw.reshape(GRID_SHAPE).astype(np.int32)
    except:
        raw = np.fromfile(label_path, dtype=np.uint16)
        semantic_labels = raw.reshape(GRID_SHAPE)
        instance_labels = np.zeros_like(semantic_labels, dtype=np.int32)

        if semantic_labels.max() > 19:
            semantic_labels = np.vectorize(learning_map.get)(semantic_labels)
    
    return semantic_labels, instance_labels

def load_real_kitti_panoptic(label_path: str) -> tuple:
    with open(label_path, 'rb') as f:
        data = pickle.load(f)
    semantic_labels = data['semantic_labels']
    instance_labels = data['instance_labels']
    mask = semantic_labels == 255
    semantic_labels[mask] = 0
    instance_labels[mask] = 0
    print(f"Unique semantic labels: {np.unique(semantic_labels)}")
    print(f"Unique instance labels: {np.unique(instance_labels)}")
    return semantic_labels, instance_labels

def load_generated_panoptic(label_path: str) -> tuple:
    """
    Load a generated panoptic voxel grid (uint32 format).
    Same format as KITTI panoptic.
    """
    print(f"Loading generated panoptic from {label_path}")
    semantic_labels, instance_labels = load_kitti_panoptic(label_path)
    print(f"  Unique semantic: {np.unique(semantic_labels)[:10]}...")
    print(f"  Unique instances: {len(np.unique(instance_labels))}")
    return semantic_labels, instance_labels


# ─────────────────────────────────────────────────────────────────────────────
# 2. Position-based instance color generation
# ─────────────────────────────────────────────────────────────────────────────
# def get_instance_color(base_color: np.ndarray,
#                        centroid_y: float,
#                        centroid_x: float,
#                        cls_id: int,
#                        quantize: int = 4) -> np.ndarray:
#     """
#     Compute a deterministic color for an instance based on its centroid position.
#     Same (class, position) always gives the same color.
    
#     Parameters
#     ----------
#     base_color : (3,) RGB base color for the class
#     centroid_y, centroid_x : instance centroid coordinates
#     cls_id : semantic class id
#     quantize : spatial quantization for stability
    
#     Returns
#     -------
#     color : (3,) uint8 RGB
#     """
#     grid_y = int(centroid_y // quantize) * quantize
#     grid_x = int(centroid_x // quantize) * quantize
    
#     pos_seed = grid_y * 10000 + grid_x + int(cls_id) * 100000
#     rng = np.random.RandomState(pos_seed)
#     variation = 0.4 + 0.6 * rng.rand()
    
#     color = (base_color.astype(np.float32) * variation).clip(0, 255).astype(np.uint8)
#     return color

def get_instance_color(base_color: np.ndarray,
                       centroid_y: float,
                       centroid_x: float,
                       cls_id: int,
                       quantize: int = 4) -> np.ndarray:
    """
    Compute a deterministic color for an instance based on its centroid position.
    Uses hue shift (not just brightness) for visibly distinct instances.
    Same (class, position) always gives the same color.
    """
    from colorsys import rgb_to_hsv, hsv_to_rgb
    
    grid_y = int(centroid_y // quantize) * quantize
    grid_x = int(centroid_x // quantize) * quantize
    
    pos_seed = grid_y * 10000 + grid_x + int(cls_id) * 100000
    rng = np.random.RandomState(pos_seed)
    
    # Convert base color to HSV
    r, g, b = base_color / 255.0
    h, s, v = rgb_to_hsv(r, g, b)
    
    # Shift hue significantly (±60° on color wheel) — strong but stays "in family"
    h_shift = (rng.rand() - 0.5) * 0.33  # ±0.165 = ±60°
    h_new = (h + h_shift) % 1.0
    
    # Keep saturation high so colors stay vivid
    s_new = np.clip(s * (0.85 + 0.15 * rng.rand()), 0.5, 1.0)
    
    # Vary brightness moderately
    v_new = np.clip(v * (0.7 + 0.3 * rng.rand()), 0.5, 1.0)
    
    r_new, g_new, b_new = hsv_to_rgb(h_new, s_new, v_new)
    color = (np.array([r_new, g_new, b_new]) * 255).clip(0, 255).astype(np.uint8)
    return color


def build_panoptic_color_grid(semantic_labels: np.ndarray,
                              instance_labels: np.ndarray) -> np.ndarray:
    """
    Build a (H, W, D, 3) RGB color grid from panoptic labels.
    Each voxel gets a color based on its (class, instance) — instances use
    position-based coloring for FID stability.
    """
    H, W, D = semantic_labels.shape
    color_grid = np.zeros((H, W, D, 3), dtype=np.uint8)
    
    unique_classes = np.unique(semantic_labels)
    
    for cls_id in unique_classes:
        if cls_id == 0:
            continue
        
        cls_idx = int(cls_id)
        base_color = KITTI_COLORS[min(cls_idx, len(KITTI_COLORS) - 1)]
        cls_mask = (semantic_labels == cls_id)
        
        # if cls_idx in THING_IDS_SET:
        if False:
            # Thing class — color each instance separately
            unique_ins = np.unique(instance_labels[cls_mask])
            unique_ins = unique_ins[unique_ins > 0]  # skip background instance ID 0
            
            for ins_id in unique_ins:
                ins_mask = (instance_labels == ins_id) & cls_mask
                if not ins_mask.any():
                    continue
                
                # Compute centroid in 2D (X, Y) — ignore Z for color stability
                ys, xs, zs = np.where(ins_mask)
                cy = ys.mean()
                cx = xs.mean()
                
                color = get_instance_color(base_color, cy, cx, cls_idx)
                color_grid[ins_mask] = color
            
            # Handle thing voxels without instance IDs (shouldn't happen but safe)
            no_ins_mask = cls_mask & (instance_labels == 0)
            color_grid[no_ins_mask] = base_color
        else:
            # Stuff class — use base color directly
            color_grid[cls_mask] = base_color
    
    return color_grid




# ─────────────────────────────────────────────────────────────────────────────
# 3. Voxel grid → trimesh with per-voxel colors
# ─────────────────────────────────────────────────────────────────────────────
def panoptic_voxels_to_mesh(semantic_labels: np.ndarray,
                            instance_labels: np.ndarray) -> trimesh.Trimesh:
    """
    Convert a panoptic voxel grid to a colored trimesh.
    Each occupied voxel becomes a colored box with position-based instance color.
    """
    # Build color grid
    color_grid = build_panoptic_color_grid(semantic_labels, instance_labels)
    
    # Find occupied voxel positions
    xs, ys, zs = np.where(semantic_labels > 0)
    if len(xs) == 0:
        return None
    
    # World-space voxel centers
    cx = xs * VOXEL_SIZE
    cy = ys * VOXEL_SIZE
    cz = zs * VOXEL_SIZE
    
    s = VOXEL_SIZE / 2.0
    N = len(xs)
    
    # Template cube vertices
    unit_v = np.array([
        [-s, -s, -s], [ s, -s, -s], [ s,  s, -s], [-s,  s, -s],
        [-s, -s,  s], [ s, -s,  s], [ s,  s,  s], [-s,  s,  s],
    ], dtype=np.float32)
    
    # Template cube faces
    unit_f = np.array([
        [0,1,2],[0,2,3], [4,5,6],[4,6,7],
        [0,1,5],[0,5,4], [2,3,7],[2,7,6],
        [1,2,6],[1,6,5], [0,3,7],[0,7,4],
    ], dtype=np.int32)
    
    # Broadcast vertices
    offsets = np.stack([cx, cy, cz], axis=1)[:, np.newaxis, :]
    all_v = unit_v[np.newaxis, :, :] + offsets
    all_v = all_v.reshape(-1, 3)
    
    # Broadcast faces
    face_off = (np.arange(N) * 8)[:, np.newaxis, np.newaxis]
    all_f = unit_f[np.newaxis, :, :] + face_off
    all_f = all_f.reshape(-1, 3)
    
    # Per-voxel colors (from panoptic color grid)
    colors = color_grid[xs, ys, zs]  # (N, 3)
    colors = np.repeat(colors, 8, axis=0)  # (N*8, 3)
    alpha = np.full((len(colors), 1), 255, dtype=np.uint8)
    colors = np.hstack([colors, alpha])  # (N*8, 4)
    
    mesh = trimesh.Trimesh(
        vertices=all_v,
        faces=all_f,
        vertex_colors=colors,
        process=False,
    )
    return mesh


# ─────────────────────────────────────────────────────────────────────────────
# 4. Camera utilities (unchanged from semantic version)
# ─────────────────────────────────────────────────────────────────────────────
def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    z = eye - target
    z_norm = np.linalg.norm(z)
    z = z / z_norm if z_norm > 1e-8 else np.array([0.0, 0.0, 1.0])
    
    x = np.cross(up, z)
    x_norm = np.linalg.norm(x)
    if x_norm < 1e-8:
        up = np.array([1.0, 0.0, 0.0])
        x = np.cross(up, z)
        x_norm = np.linalg.norm(x)
    x = x / x_norm
    y = np.cross(z, x)
    
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = z
    pose[:3, 3] = eye
    return pose


def _get_camera_pose(centre, extents, mode):
    diag = float(np.linalg.norm(extents))
    if mode == 'bev':
        eye = centre + np.array([0.0, 0.0, diag * 0.7])
        up = np.array([0.0, 1.0, 0.0])
    elif mode == 'front':
        eye = centre + np.array([0.0, -diag * 1.2, diag * 0.4])
        up = np.array([0.0, 0.0, 1.0])
    elif mode == 'iso':
        eye = centre + np.array([diag * 0.8, -diag * 0.8, diag * 0.7])
        up = np.array([0.0, 0.0, 1.0])
    else:
        raise ValueError(f"Unknown camera mode '{mode}'")
    return eye, up


# ─────────────────────────────────────────────────────────────────────────────
# 5. Core render function
# ─────────────────────────────────────────────────────────────────────────────
def render_panoptic(semantic_labels: np.ndarray,
                    instance_labels: np.ndarray,
                    camera_mode: str = 'bev',
                    img_w: int = IMG_W,
                    img_h: int = IMG_H) -> Image.Image:
    """
    Render a panoptic voxel grid to a PIL Image.
    """
    mesh = panoptic_voxels_to_mesh(semantic_labels, instance_labels)
    
    if mesh is None:
        return Image.fromarray(np.zeros((img_h, img_w, 3), dtype=np.uint8))
    
    scene = pyrender.Scene(
        bg_color=[0.1, 0.1, 0.1, 1.0],
        ambient_light=[0.4, 0.4, 0.4],
    )
    
    py_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(py_mesh)
    
    bounds = mesh.bounds
    centre = (bounds[0] + bounds[1]) / 2.0
    extents = bounds[1] - bounds[0]
    diag = float(np.linalg.norm(extents))
    
    eye, up = _get_camera_pose(centre, extents, camera_mode)
    cam_pose = _look_at(eye, centre, up)
    
    camera = pyrender.PerspectiveCamera(
        yfov=np.pi / 3.0,
        aspectRatio=img_w / img_h,
        znear=0.1,
        zfar=diag * 10.0,
    )
    scene.add(camera, pose=cam_pose)
    
    # Key light
    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(key_light, pose=cam_pose)
    
    # Fill light from opposite side
    fill_eye = centre - (eye - centre)
    fill_pose = _look_at(fill_eye, centre, up)
    fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
    scene.add(fill_light, pose=fill_pose)
    
    renderer = pyrender.OffscreenRenderer(img_w, img_h)
    color, _ = renderer.render(scene)
    renderer.delete()
    
    return Image.fromarray(color)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Batch rendering
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_SEQUENCES = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']


def render_real_dataset(sequences_dir: str,
                        output_dir: str,
                        camera_mode: str = 'bev',
                        sequences: list = None):
    """Render real SemanticKITTI panoptic scenes to PNGs."""
    os.makedirs(output_dir, exist_ok=True)
    if sequences is None:
        sequences = TRAIN_SEQUENCES
    
    total, success = 0, 0
    for seq in sequences:
        voxel_dir = Path(sequences_dir) / seq
        if not voxel_dir.exists():
            print(f"[WARN] Sequence {seq} not found at {voxel_dir}")
            continue
        
        label_files = sorted(voxel_dir.glob('*.pkl'))
        print(f"Sequence {seq}: {len(label_files)} scans")
        
        for label_path in label_files:
            total += 1
            try:
                sem_labels, ins_labels = load_real_kitti_panoptic(str(label_path))
                img = render_panoptic(sem_labels, ins_labels, camera_mode)
                out_name = f"{seq}_{label_path.stem}.png"
                img.save(Path(output_dir) / out_name)
                success += 1
                if total % 50 == 0:
                    print(f"  Rendered {total} scenes...")
            except Exception as e:
                print(f"  [ERROR] {label_path.name}: {e}")
    
    print(f"\nDone. {success}/{total} scenes rendered to {output_dir}")


def render_generated_dataset(generated_dir: str,
                             output_dir: str,
                             camera_mode: str = 'bev'):
    """Render generated panoptic .label files to PNGs."""
    os.makedirs(output_dir, exist_ok=True)
    label_files = sorted(Path(generated_dir).glob('*.label'))
    
    if len(label_files) == 0:
        print(f"[ERROR] No .label files found in {generated_dir}")
        return
    
    print(f"Found {len(label_files)} generated scenes")
    success = 0
    for i, label_path in enumerate(label_files):
        try:
            sem_labels, ins_labels = load_generated_panoptic(str(label_path))
            img = render_panoptic(sem_labels, ins_labels, camera_mode)
            img.save(Path(output_dir) / f"{i:05d}.png")
            success += 1
            if (i + 1) % 50 == 0:
                print(f"  Rendered {i+1}/{len(label_files)}")
        except Exception as e:
            print(f"  [ERROR] {label_path.name}: {e}")
    
    print(f"\nDone. {success}/{len(label_files)} scenes rendered to {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description='Render SemanticKITTI panoptic scenes for FID evaluation')
    parser.add_argument('--mode', type=str, required=True,
        choices=['real', 'generated'],
        help='real: render real dataset | generated: render .label files')
    parser.add_argument('--sequences_dir', type=str, default=None,
        help='Path to SemanticKITTI sequences folder (for --mode real)')
    parser.add_argument('--generated_dir', type=str, default=None,
        help='Path to folder of .label generated panoptic files (for --mode generated)')
    parser.add_argument('--output_dir', type=str, required=True,
        help='Where to save rendered PNG images')
    parser.add_argument('--camera', type=str, default='bev',
        choices=['bev', 'front', 'iso'],
        help='Camera mode (default: bev)')
    parser.add_argument('--sequences', type=str, nargs='+', default=None,
        help='Specific sequence ids (default: all training sequences)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    if args.mode == 'real':
        assert args.sequences_dir is not None, "--sequences_dir required"
        render_real_dataset(
            sequences_dir=args.sequences_dir,
            output_dir=args.output_dir,
            camera_mode=args.camera,
            sequences=args.sequences,
        )
    elif args.mode == 'generated':
        assert args.generated_dir is not None, "--generated_dir required"
        render_generated_dataset(
            generated_dir=args.generated_dir,
            output_dir=args.output_dir,
            camera_mode=args.camera,
        )