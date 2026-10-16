"""Publication figure of a layout condition as a floating 3D tile.

    python fig_layout3d.py --scene 00/000995                  # -> figs/00_000995_3d.png
    python fig_layout3d.py --scene 00/000995 --transparent    # alpha background
    python fig_layout3d.py --scene 00/000995 --interactive    # pick a camera, press C

The BEV layout is drawn as a thin slab floating above a shadow-catching floor:
    top      ground classes in the SemanticKITTI colours; the class heatmaps the
             model is conditioned on rise out of it as smooth coloured mounds
    pins     one glossy pin per requested object (centers_from_layout), stem
             from the mound peak, head in the class colour
Orientation matches fig_layout.py / visualize_layout_2d.py when seen from above.
"""
import os

import click
import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter, zoom

from fig_layout import palette, THING_NAMES
from utils.layout import (GROUND_IDS, N_GROUND, centers_from_layout, dequantise,
                          set_thing_ids, thing_ids_for_channels)

VOXEL_SIZE = 0.2
TILE = np.array([0.975, 0.972, 0.962])        # empty cells of the slab
SIDE = '#b9b9c2'                               # slab edge
DEFAULT_CAM = 'auto'


def surface_fields(lay, pal, thing_ids, factor, smooth, blend, halo=3.0):
    """Per-point RGB and heat on a (S*factor)^2 grid."""
    def up(a, s=0.0):
        a = gaussian_filter(a, s) if s > 0 else a
        return zoom(a, factor, order=1)

    H = lay.shape[1] * factor
    rgb = np.ones((H, H, 3), np.float32) * TILE
    for i, g in enumerate(GROUND_IDS):
        m = up(lay[i], smooth) > 0.5
        rgb[m] = (1 - blend) * pal[g] + blend * TILE

    heat = np.stack([up(lay[N_GROUND + k]) for k in range(len(thing_ids))])
    dom, a = heat.argmax(0), np.clip(heat.max(0), 0, 1)
    cls_rgb = pal[np.array(thing_ids)][dom]

    # halo: a wider, decorative pool of the class colour around each centre, so
    # objects read at figure scale. The sharp core on top is the real heatmap.
    if halo > 0:
        hal = np.stack([up(gaussian_filter(lay[N_GROUND + k], halo))
                        for k in range(len(thing_ids))])
        hal = hal / max(hal.max(), 1e-6)
        hdom, ha = hal.argmax(0), np.clip(hal.max(0), 0, 1)
        hw = (ha ** 0.8)[..., None] * 0.45
        rgb = (1 - hw) * rgb + hw * pal[np.array(thing_ids)][hdom]

    w = (a ** 1.2)[..., None] * 0.9
    rgb = (1 - w) * rgb + w * cls_rgb
    return rgb, a


def build_scene(pl, lay, req, pal, thing_ids, factor=4, smooth=0.7, blend=0.35,
                mound=0.25, pin_h=3.4, head_r=0.8, thickness=1.2, floor_gap=7.0,
                floor=True, halo=3.0):
    S = lay.shape[1]
    L = S * VOXEL_SIZE
    rgb, heat = surface_fields(lay, pal, thing_ids, factor, smooth, blend, halo)
    H = rgb.shape[0]

    # image row r -> voxel x -> world -y ; image col c -> voxel y -> world x
    vox = (np.arange(H) + 0.5) / factor - 0.5
    wx = vox * VOXEL_SIZE
    wy = -vox * VOXEL_SIZE
    X, Y = np.meshgrid(wx, wy, indexing='xy')        # X[r, c] = wx[c], Y[r, c] = wy[r]
    Z = heat * mound
    top = pv.StructuredGrid(X, Y, Z)
    # StructuredGrid flattens in Fortran order over (x, y); match it
    top['rgb'] = (rgb.transpose(1, 0, 2).reshape(-1, 3) * 255).astype(np.uint8)
    top = top.extract_surface().compute_normals(split_vertices=False)
    pl.add_mesh(top, scalars='rgb', rgb=True, smooth_shading=True,
                ambient=0.30, diffuse=0.70, specular=0.10, specular_power=20)

    # slab body
    x0, x1 = wx[0], wx[-1]
    y0, y1 = wy[-1], wy[0]
    slab = pv.Box(bounds=(x0, x1, y0, y1, -thickness, -0.01))
    pl.add_mesh(slab, color=SIDE, ambient=0.35, diffuse=0.7, specular=0.1)

    # pins
    for xv, yv, c in req:
        px, py = yv * VOXEL_SIZE, -xv * VOXEL_SIZE
        r = int(round(xv * factor)), int(round(yv * factor))
        z0 = float(heat[min(r[0], H - 1), min(r[1], H - 1)] * mound)
        color = pal[c]
        stem = pv.Cylinder(center=(px, py, z0 + pin_h / 2), direction=(0, 0, 1),
                           radius=0.1, height=pin_h, resolution=24)
        pl.add_mesh(stem, color='#3c3c44', ambient=0.3, diffuse=0.7, specular=0.4)
        head = pv.Sphere(radius=head_r, center=(px, py, z0 + pin_h),
                         theta_resolution=48, phi_resolution=48)
        pl.add_mesh(head, color=color, smooth_shading=True, ambient=0.3, diffuse=0.75,
                    specular=0.9, specular_power=60)

    if floor:
        add_soft_shadow(pl, (x0, x1, y0, y1), -thickness - floor_gap)
    return (x0 + x1) / 2, (y0 + y1) / 2, L


def add_soft_shadow(pl, bounds, z, offset=(1.5, -2.5), blur=0.06, strength=0.38, res=512):
    """Baked drop shadow on a white floor: a blurred slab silhouette.

    VTK shadow maps are hard-edged and alias on large planes; a texture gives the
    soft contact-free shadow that makes the tile read as floating.
    """
    x0, x1, y0, y1 = bounds
    L = max(x1 - x0, y1 - y0)
    span = 3.0 * L
    cx, cy = (x0 + x1) / 2 + offset[0], (y0 + y1) / 2 + offset[1]
    u = np.linspace(-span / 2, span / 2, res)
    U, V = np.meshgrid(u, u)
    m = ((np.abs(U) <= (x1 - x0) / 2) & (np.abs(V) <= (y1 - y0) / 2)).astype(np.float32)
    m = gaussian_filter(m, blur * res)
    img = (np.clip(1 - strength * m, 0, 1) * 255).astype(np.uint8)
    tex = pv.numpy_to_texture(np.dstack([img] * 3))
    fl = pv.Plane(center=(cx, cy, z), direction=(0, 0, 1), i_size=span, j_size=span)
    pl.add_mesh(fl, texture=tex, ambient=1.0, diffuse=0.0, specular=0.0, lighting=False)


def setup_lights(pl, center, L):
    pl.remove_all_lights()
    cx, cy = center
    key = pv.Light(position=(cx - 0.5, cy + 0.8, 1.6), focal_point=(cx, cy, 0),
                   intensity=0.62, light_type='scene light', positional=False)
    key.position = (cx - 0.5 * L, cy + 0.8 * L, 1.6 * L)
    fill = pv.Light(position=(cx + L, cy - L, 0.9 * L), focal_point=(cx, cy, 0),
                    intensity=0.28, light_type='scene light', positional=False)
    head = pv.Light(light_type='headlight', intensity=0.22)
    for l in (key, fill, head):
        pl.add_light(l)


def default_camera(center, L, elev=38.0, azim=-28.0, dist=2.9):
    cx, cy = center
    el, az = np.radians(elev), np.radians(azim)
    d = dist * L
    # looking from "below" the image (world -y), rotated by azim
    pos = (cx + d * np.cos(el) * np.sin(az), cy - d * np.cos(el) * np.cos(az),
           d * np.sin(el))
    return [pos, (cx, cy, -0.5), (0, 0, 1)]


def autocrop(img, transparent, margin=0.03, tol=4):
    """Trim uniform background around the render, keeping a small margin."""
    if transparent and img.shape[2] == 4:
        fg = img[..., 3] > tol
    else:
        fg = np.abs(img[..., :3].astype(int) - img[0, 0, :3].astype(int)).max(-1) > tol
    rows, cols = np.where(fg.any(1))[0], np.where(fg.any(0))[0]
    if len(rows) == 0:
        return img
    m = int(margin * max(img.shape[:2]))
    r0, r1 = max(rows[0] - m, 0), min(rows[-1] + m + 1, img.shape[0])
    c0, c1 = max(cols[0] - m, 0), min(cols[-1] + m + 1, img.shape[1])
    return img[r0:r1, c0:c1]


@click.command()
@click.option('--path', '-p', type=str,
              default="Q:/OWM_workspace/panscene_Dev_savings/exp/Layout/panscene_layout_v2")
@click.option('--scene', type=str, default='00/000995', help="<seq>/<frame>")
@click.option('--out', type=str, default=None, help="Output PNG. Default figs/<seq>_<frame>_3d.png")
@click.option('--elev', type=float, default=38.0, help="Camera elevation (deg).")
@click.option('--azim', type=float, default=-28.0, help="Camera azimuth (deg).")
@click.option('--zoom', 'zoom_', type=float, default=1.0)
@click.option('--cam', type=str, default=None,
              help="Exact camera as printed by --interactive / C key.")
@click.option('--mound', type=float, default=0.25, help="Heatmap mound height (m).")
@click.option('--pin', type=float, default=3.4, help="Pin height (m).")
@click.option('--halo', type=float, default=3.0,
              help="Decorative glow radius around centres (voxels). 0 = heatmap only.")
@click.option('--no_floor', is_flag=True, help="No shadow floor; slab floats on the background.")
@click.option('--shadows', is_flag=True, help="Also cast VTK shadow maps (pins onto tile).")
@click.option('--transparent', is_flag=True, help="Transparent PNG background.")
@click.option('--size', type=str, default='2400x1800', help="Render size WxH (px).")
@click.option('--interactive', is_flag=True, help="Open a window to choose the camera.")
def main(path, scene, out, elev, azim, zoom_, cam, mound, pin, halo, no_floor, shadows,
         transparent, size, interactive):
    seq, frame = scene.split('/')
    lay = np.load(os.path.join(path, seq, f'{frame}.npz'))['layout']
    lay = dequantise(lay) if lay.dtype == np.uint8 else lay.astype(np.float32)
    thing_ids = thing_ids_for_channels(lay.shape[0])
    set_thing_ids(thing_ids)
    req = centers_from_layout(lay)
    pal = palette()

    W, Hh = (int(v) for v in size.lower().split('x'))
    pl = pv.Plotter(off_screen=not interactive, window_size=[W, Hh], lighting='none')
    pl.set_background('white')
    cx, cy, L = build_scene(pl, lay, req, pal, thing_ids, mound=mound, pin_h=pin,
                            floor=not no_floor, halo=halo)
    setup_lights(pl, (cx, cy), L)
    if shadows:
        pl.enable_shadows()       # hard VTK shadow maps; the floor shadow is baked anyway
    pl.enable_anti_aliasing('ssaa')

    pl.camera_position = eval(cam) if cam else default_camera((cx, cy), L, elev, azim)
    pl.camera.view_angle = 24.0
    if zoom_ != 1.0:
        pl.camera.zoom(zoom_)

    print(f'{len(req)} requested objects: '
          + ', '.join(THING_NAMES.get(c, str(c)) for _, _, c in req))
    if interactive:
        pl.add_key_event('c', lambda: print('--cam "', pl.camera_position, '"'))
        print('Rotate to taste, press C to print the camera, close to exit.')
        pl.show()
        return

    out = out or f'figs/{seq}_{frame}_3d.png'
    os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)
    img = pl.screenshot(None, transparent_background=transparent, return_img=True)
    pl.close()
    img = autocrop(img, transparent)
    import imageio.v2 as imageio
    imageio.imwrite(out, img)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
