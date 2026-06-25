import pyvista as pv
import numpy as np
import click
import os
import json
import pickle
from utils.data_map import color_map, learning_map
import time

GRID_SHAPE = (256, 256, 32)
VOXEL_SIZE = 0.2

with open("./color_palette.json", 'r') as f:
    thing_color = json.load(f)


def load_voxels(pcd_file, panoptic=False, voxel_size=VOXEL_SIZE, thing_only=False ):
    raw = np.fromfile(pcd_file, dtype=np.uint16)
    if raw.size == np.prod(GRID_SHAPE):
        labels = raw.reshape(GRID_SHAPE).astype(np.uint32)
    else:
        labels = np.fromfile(pcd_file, dtype=np.uint32).reshape(GRID_SHAPE)

    sem = labels & 0xFFFF
    if sem.max() > 19:
        sem = np.vectorize(lambda k: learning_map.get(k, 0))(sem)

    xs, ys, zs = np.where(sem > 0)
    sem_labels = sem[xs, ys, zs].astype(int)
    centers = np.stack([xs, ys, zs], axis=1).astype(np.float32) * voxel_size

    color_array = np.array(list(color_map.values()))
    sem_labels = np.clip(sem_labels, 0, len(color_array) - 1)
    rgb = color_array[sem_labels][:, ::-1] / 255.0  # BGR->RGB

    if thing_only:
        # Filter out stuff classes (assuming thing classes have labels > 0 and <= 19)
        thing_mask = (sem_labels > 0) & (sem_labels <= 9)
        rgb[thing_mask] = [1.0, 1.0, 1.0]  # Set thing classes to white
    else:
        thing_mask = np.ones(len(sem_labels), dtype=bool)  # All voxels are considered "things" if not filtering

    if panoptic:
        ins = (labels[xs, ys, zs] >> 16).astype(np.int64)
        for label in np.unique(ins):
            if label == 0:
                continue
            mask = ins == label
            c = thing_color.get(str(int(label) + np.random.randint(0, 15)))
            if c is not None:
                rgb[mask] = c
    return centers, rgb, thing_mask

def load_gt(pcd_file, panoptic=False, thing_only=False, voxel_size=VOXEL_SIZE):
    with open(pcd_file, 'rb') as f:
        raw = pickle.load(f)
    sem = raw['semantic_labels'].astype(np.uint8)
    sem[sem >= 250] = 0  # remove unknown labels


    if sem.max() > 19:
        sem = np.vectorize(lambda k: learning_map.get(k, 0))(sem)

    xs, ys, zs = np.where(sem > 0)
    sem_labels = sem[xs, ys, zs].astype(int)
    centers = np.stack([xs, ys, zs], axis=1).astype(np.float32) * voxel_size

    color_array = np.array(list(color_map.values()))
    sem_labels = np.clip(sem_labels, 0, len(color_array) - 1)
    rgb = color_array[sem_labels][:, ::-1] / 255.0  # BGR->RGB

    if thing_only:
        # Filter out stuff classes (assuming thing classes have labels > 0 and <= 19)
        thing_mask = (sem_labels < 9)
        rgb[~thing_mask] = [1.0, 1.0, 1.0]  # Set thing classes to white
    else:
        thing_mask = np.ones(len(sem_labels), dtype=bool)  # All voxels are considered "things" if not filtering

    if panoptic:
        ins = (raw['instance_labels'][xs, ys, zs]).astype(np.uint8)
        for label in np.unique(ins):
            if label == 0:
                continue
            mask = ins == label
            c = thing_color.get(str(int(label) ))
            if c is not None:
                rgb[mask] = c
    return centers, rgb, thing_mask


def build_glyph_cubes(centers, rgb, voxel_size=VOXEL_SIZE, gap=0.08):
    pts = pv.PolyData(centers)
    pts['rgb'] = (rgb * 255).astype(np.uint8)
    cube = pv.Cube(x_length=voxel_size * (1 - gap),
                   y_length=voxel_size * (1 - gap),
                   z_length=voxel_size * (1 - gap))
    return pts.glyph(geom=cube, scale=False, orient=False)


def save_figure(pl, out_stem, fmt, scale=3, transparent=False):
    fmt = fmt.lower().lstrip('.')
    out_path = f"{out_stem}.{fmt}"
    if fmt == "png":
        pl.screenshot(out_path, scale=scale, transparent_background=transparent)
        note = " (transparent)" if transparent else ""
        print(f"saved {out_path}  ({scale}x raster{note})")
    elif fmt in ("jpg", "jpeg"):
        if transparent:
            print("[!] JPG has no alpha channel — saving opaque. Use PNG for transparency.")
        pl.screenshot(out_path, scale=scale)
        print(f"saved {out_path}  ({scale}x raster)")
    elif fmt in ("svg", "pdf", "eps"):
        print(f"[!] Vector export of dense voxels can be huge/slow. Working...")
        pl.save_graphic(out_path)
        print(f"saved {out_path}  (vector)")
    else:
        print(f"[!] Unknown format '{fmt}', skipping.")


@click.command()
@click.option('--path', '-p', type=str, default="data/sample_npz")
@click.option('--panoptic', is_flag=True)
@click.option('--thing_only', is_flag=True, help="Show only thing classes ")
@click.option('--out-dir', type=str, default="renders")
@click.option('--fmt', type=click.Choice(['png', 'jpg', 'svg', 'pdf']),
              default='png', help="Default save format for the S key.")
@click.option('--cam', type=str, default=None,
              help="Initial camera pose as a Python tuple string, "
                   "e.g. \"[(x,y,z),(fx,fy,fz),(ux,uy,uz)]\"")
@click.option('--gt', is_flag=True, help="Load ground truth voxel data instead of label files.")
@click.option('--gt_scene', type=str, default=None, help="Path to the ground truth scene file (pkl) if --gt is used.")

def main(path, panoptic, thing_only, out_dir, fmt, cam, gt, gt_scene):
    os.makedirs(out_dir, exist_ok=True)
    if not gt:
        files = sorted([f for f in os.listdir(path) if f.endswith('.label')])
    else:
        if gt_scene is None:
            files = sorted([f for f in os.listdir(path) if f.endswith('.pkl')])
        else:
            files = [f"{gt_scene}.pkl"]
    print(f"Found {len(files)} label files.")
    print("Keys:  N/B = next/prev   C = print camera   "
          "S = save   1/2/3 = png/jpg/svg   Q = quit")

    # mutable state shared with key callbacks
    state = {"idx": 0, "fmt": fmt, "quit": False, "reload": True, "transparent": False, "stuff_opacity": 0.0}

    pl = pv.Plotter(window_size=[1600, 1000])
    pl.background_color = 'white'

    # optional starting camera pose from CLI
    init_cam = None
    if cam:
        try:
            init_cam = eval(cam)
        except Exception as e:
            print(f"[!] Could not parse --cam: {e}")

    def load_current(gt=False):
        """Clear and load the file at state['idx'] into the open window."""
        pl.clear()
        stem = files[state["idx"]]
        full_path = os.path.join(path, stem)
        if not gt:
            centers, rgb, is_thing = load_voxels(full_path, panoptic=panoptic, thing_only=thing_only)
        else:
            centers, rgb, is_thing = load_gt(full_path, panoptic=panoptic, thing_only=thing_only)
        print(f"\n[{state['idx']+1}/{len(files)}] {stem}: {len(centers)} voxels")
        common = dict(scalars='rgb', rgb=True, show_scalar_bar=False,
                      smooth_shading=True, ambient=0.7, diffuse=0.5, specular=0.2)

        # things: fully opaque
        if is_thing.any():
            g_thing = build_glyph_cubes(centers[is_thing], rgb[is_thing])
            pl.add_mesh(g_thing, opacity=1.0, **common)

        # non-things: low opacity
        non = ~is_thing
        if non.any():
            g_stuff = build_glyph_cubes(centers[non], rgb[non])
            pl.add_mesh(g_stuff, opacity=state["stuff_opacity"], **common)

        pl.enable_eye_dome_lighting()
        pl.add_text(stem, font_size=10, name="title")
        if init_cam is not None:
            pl.camera_position = init_cam

    def next_file():
        if state["idx"] < len(files) - 1:
            state["idx"] += 1
            load_current(gt=gt)
        else:
            print("[!] Already at last file.")

    def prev_file():
        if state["idx"] > 0:
            state["idx"] -= 1
            load_current(gt=gt)
        else:
            print("[!] Already at first file.")

    def get_camera():
        print("camera_position =", pl.camera_position)

    def do_save():
        # pl remove text
        pl.remove_actor("title")
        stem = os.path.splitext(files[state["idx"]])[0]
        time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
        save_figure(pl, os.path.join(out_dir, f"{stem}_{time_str}"), state["fmt"], transparent=state["transparent"])

    def set_fmt_png():
        state["fmt"] = "png"; print("save format -> png")
    def set_fmt_jpg():
        state["fmt"] = "jpg"; print("save format -> jpg")
    def set_fmt_svg():
        state["fmt"] = "svg"; print("save format -> svg")

    def toggle_transparent():
        state["transparent"] = not state["transparent"]
        print(f"transparent background -> {'on' if state['transparent'] else 'off'}")

    def do_quit():
        state["quit"] = True
        pl.close()
        os._exit(0)

    pl.add_key_event('n', next_file)
    pl.add_key_event('b', prev_file)
    pl.add_key_event('c', get_camera)
    pl.add_key_event('s', do_save)
    pl.add_key_event('t', toggle_transparent)
    pl.add_key_event('1', set_fmt_png)
    pl.add_key_event('2', set_fmt_jpg)
    pl.add_key_event('3', set_fmt_svg)
    pl.add_key_event('q', do_quit)

    load_current(gt=gt)
    pl.show()  # blocks until window closed / Q pressed


if __name__ == "__main__":
    main()