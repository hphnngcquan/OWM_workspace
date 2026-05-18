import matplotlib.pyplot as plt
import numpy as np

GRID_SHAPE = (512, 512, 32)
labels = np.fromfile("./data/outpainting_sample/outpainting.label", dtype=np.uint16).reshape(GRID_SHAPE)

# Show bird's-eye view (max projection over Z)
bev = labels.max(axis=2)  # (512, 512)

plt.figure(figsize=(10, 10))
plt.imshow(bev, cmap='tab20', interpolation='nearest')
plt.title('Outpainted Scene (BEV)')
plt.colorbar(label='Class ID')
plt.axis('equal')
plt.savefig('outpainting_bev.png', dpi=150, bbox_inches='tight')
plt.show()

# Show individual Z slices
fig, axes = plt.subplots(4, 8, figsize=(20, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(labels[:, :, i], cmap='tab20')
    ax.set_title(f'Z={i}')
    ax.axis('off')
plt.tight_layout()
plt.savefig('outpainting_slices.png', dpi=120, bbox_inches='tight')