VOXEL_SIZE = 0.2
GRID_SHAPE = (256, 256, 32)
# GRID_SHAPE = (512, 512, 32)
import numpy as np
import os

def save_voxel_as_npz_pan(labels, save_path, voxel_size=VOXEL_SIZE):
    """
    Save occupied voxels as single (N,4) array:
    [x, y, z, label]
    """

    xs, ys, zs = np.where(labels > 0)

    sem_labels = labels[xs, ys, zs].astype(np.uint8)

    ins_labels = labels[xs, ys, zs] >> 16
    print(np.unique(ins_labels))

    x = xs.astype(np.float32) * voxel_size
    y = ys.astype(np.float32) * voxel_size
    z = zs.astype(np.float32) * voxel_size

    arr = np.stack([x, y, z, sem_labels, ins_labels], axis=1)   # (N,5)

    np.savez_compressed(save_path, arr)

    print(f"Saved {arr.shape} to {save_path}")

def save_voxel_as_npz_sem(labels, save_path, voxel_size=VOXEL_SIZE):
    """
    Save occupied voxels as single (N,4) array:
    [x, y, z, label]
    """

    xs, ys, zs = np.where(labels > 0)

    sem_labels = labels[xs, ys, zs].astype(np.uint8)


    x = xs.astype(np.float32) * voxel_size
    y = ys.astype(np.float32) * voxel_size
    z = zs.astype(np.float32) * voxel_size

    arr = np.stack([x, y, z, sem_labels], axis=1)   # (N,4)

    np.savez_compressed(save_path, arr)

    print(f"Saved {arr.shape} to {save_path}")

if __name__ == "__main__":
    for i in range(1300):
        print(f"./data/gen_pan_from_sem_DBSCAN/sample/{i}.label")
        labels = np.fromfile(f"./data/gen_pan_from_sem_DBSCAN/sample/{i}.label", dtype=np.uint32).reshape(GRID_SHAPE)
        SAVE_PATH = "./data/gen_pan_from_sem_DBSCAN/sample_npz"
        os.makedirs(SAVE_PATH, exist_ok=True)
        save_voxel_as_npz_pan(labels, os.path.join(SAVE_PATH, f"{i}.npz"))