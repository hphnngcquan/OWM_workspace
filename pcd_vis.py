import open3d as o3d
import click
import os
from render_lidar import simulate_lidar
from utils.data_map import color_map, learning_map
import numpy as np
from tqdm import tqdm
from random import shuffle
import json
GRID_SHAPE = (256, 256, 32)
VOXEL_SIZE = 0.2
with open("./color_palette.json", 'r') as f:
    thing_color = json.load(f)

def visualize_pcd(pcd, window_name):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)

    def close_callback(vis):
        vis.close()

    vis.register_key_callback(ord("Q"), close_callback)

    while vis.poll_events():
        vis.update_renderer()
    vis.destroy_window()


def load_pcd(pcd_file, xyz_range, panoptic=False, voxel_size=VOXEL_SIZE):
    if pcd_file.endswith('.npz'):
        points = np.load(pcd_file)['arr_0']
    elif pcd_file.endswith('.npy'):
        points = np.load(pcd_file)
    elif pcd_file.endswith('.label'):
        try:
            labels = np.fromfile(pcd_file, dtype=np.uint32).reshape(GRID_SHAPE)
            print(np.unique(labels))
        except:
            labels = np.fromfile(pcd_file, dtype=np.uint16).reshape(GRID_SHAPE)
            labels = np.vectorize(learning_map.get)(labels)
            # Process the label data to extract points and labels

        xs, ys, zs = np.where(labels > 0)
        sem_labels = labels[xs, ys, zs].astype(np.uint8)
        x = xs.astype(np.float32) * voxel_size
        y = ys.astype(np.float32) * voxel_size
        z = zs.astype(np.float32) * voxel_size
        if panoptic:
            ins_labels = labels[xs, ys, zs] >> 16
            print(np.unique(ins_labels))
            points = np.stack([x, y, z, sem_labels, ins_labels], axis=1)
        else:
            points = np.stack([x, y, z, sem_labels], axis=1)

    if xyz_range is None:
        return points
    grid_fov = (
        (points[:, 0] > xyz_range[0][0]) & (points[:, 0] < xyz_range[0][1]) &
        (points[:, 1] > xyz_range[1][0]) & (points[:, 1] < xyz_range[1][1]) &
        (points[:, 2] > xyz_range[2][0]) & (points[:, 2] < xyz_range[2][1])
    )

    return points[grid_fov]


def npy_to_pcd(pcd_file, panoptic=False, outpainting=False):
    if pcd_file.endswith('.ply'):
        return o3d.io.read_point_cloud(pcd_file)

    # if outpainting:
    #     xyz_range = [[-51.2, 51.2], [-51.2, 51.2], [0, 6.4]]
    # else:
    #     xyz_range = [[-25.6, 25.6], [-25.6, 25.6], [-2.2, 4.2]]
    pcd_ = load_pcd(pcd_file, xyz_range=None, panoptic=panoptic)

    points = np.round(pcd_[:, :3] / 0.1) * 0.1
    sem_labels = pcd_[:, 3].astype(int)
    if panoptic:
        ins_labels = pcd_[:, 4].astype(int)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # if 'x0' in pcd_file:
    if sem_labels.max() > 19:
        sem_labels = np.vectorize(learning_map.get)(sem_labels)
    color_array = np.array(list(color_map.values()))
    colors = color_array[sem_labels, ::-1] /255.0

    if panoptic:    
        for label in np.unique(ins_labels):
            if label == 0:
                continue
            mask = ins_labels == label
            colors[mask] = thing_color.get(str(label + np.random.randint(0, 15)))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


@click.command()
@click.option('--path', '-p', type=str, default="data/sample_npz")
@click.option('--panoptic', is_flag=True, help="Whether to visualize panoptic labels (default: False)")
@click.option('--outpainting', is_flag=True, help="Whether to visualize outpainting data (default: False)")
def main(path, panoptic, outpainting):
    pcd_list = os.listdir(path)
    # shuffle(pcd_list)

    for pcd_file in tqdm(pcd_list):
        print(f"\n{pcd_file}")
        full_path = os.path.join(path, pcd_file)

        # Main PCD
        pcd = npy_to_pcd(full_path, panoptic=panoptic, outpainting=outpainting)
        pcd.estimate_normals()
        visualize_pcd(pcd, f"PCD{pcd_file}")

        # # Optional LiDAR comparison
        # if 'single_scan' in path:
        #     lidar_path = os.path.join(path, 'cond', pcd_file)
        #     if os.path.exists(lidar_path):
        #         pcd_lidar = npy_to_pcd(lidar_path, panoptic=panoptic, outpainting=outpainting)
        #         pcd_lidar.estimate_normals()
        #         visualize_pcd(pcd_lidar, "PCD LiDAR")


if __name__ == "__main__":
    main()