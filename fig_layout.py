"""Publication figure of a layout condition (one BEV panel).

    python fig_layout.py --scene 00/000995                       # -> figs/00_000995.pdf + .png
    python fig_layout.py --scene 00/000995 --theme dark --legend none
    python fig_layout.py --legend_only --out figs/legend          # shared legend strip

What is drawn, back to front:
    ground   road / parking / sidewalk / other-ground in the SemanticKITTI colours,
             blended toward the background so the objects carry the eye
    glow     the class heatmaps themselves -- the gaussians the model is
             conditioned on -- as a soft halo in the class colour
    markers  each requested object as read back by centers_from_layout, one
             marker shape per class so the figure survives grayscale print
    scale    a 10 m bar (0.2 m voxels)

Orientation matches visualize_layout_2d.py (imshow, origin='upper').
"""
import os

import click
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.ndimage import gaussian_filter, zoom

from utils.layout import (GROUND_IDS, N_GROUND, centers_from_layout, dequantise,
                          set_thing_ids, thing_ids_for_channels)

SEMKITTI_YAML_PATH = './semantic_kitti.yaml'
VOXEL_SIZE = 0.2

GROUND_NAMES = ('road', 'parking', 'sidewalk', 'other-ground')
THING_NAMES = {1: 'car', 2: 'bicycle', 3: 'motorcycle', 4: 'truck',
               5: 'other-vehicle', 6: 'person', 7: 'bicyclist', 8: 'motorcyclist'}
MARKERS = {1: 'o', 2: 'D', 3: 'd', 4: 's', 5: 'p', 6: '^', 7: 'v', 8: '<'}
MARKER_SIZE = {1: 6.5, 2: 5.0, 3: 6.5, 4: 6.5, 5: 7.5, 6: 7.0, 7: 7.0, 8: 7.0}

THEMES = {
    #           background         ground blend   frame       text
    'light': dict(bg=(1.0, 1.0, 1.0), blend=0.45, frame='#9a9a9a', text='#222222',
                  edge='white', glow=0.55),
    'dark':  dict(bg=(0.06, 0.06, 0.08), blend=0.50, frame='#3a3a42', text='#e8e8e8',
                  edge='white', glow=0.65),
}


def palette():
    """SemanticKITTI colours indexed by train id, RGB float. color_map is BGR."""
    y = yaml.safe_load(open(SEMKITTI_YAML_PATH))
    pal = np.zeros((256, 3), np.float32)
    for t, raw in y['learning_map_inv'].items():
        pal[t] = np.array(y['color_map'][raw][::-1], np.float32) / 255.0
    return pal


def upsample(a, factor, smooth):
    """Bilinear upsampling, optionally after a sub-voxel blur that rounds staircase edges."""
    if smooth > 0:
        a = gaussian_filter(a, smooth)
    return zoom(a, factor, order=1) if factor > 1 else a


def ground_rgba(lay, pal, theme, factor, smooth):
    t = THEMES[theme]
    bg = np.array(t['bg'], np.float32)
    H, W = lay.shape[1] * factor, lay.shape[2] * factor
    img = np.ones((H, W, 3), np.float32) * bg
    for i, g in enumerate(GROUND_IDS):
        m = upsample(lay[i], factor, smooth) > 0.5
        img[m] = (1 - t['blend']) * pal[g] + t['blend'] * bg
    return img


def glow_rgba(lay, thing_ids, pal, theme, factor):
    """One RGBA layer: dominant class colour, alpha from heatmap strength."""
    heat = np.stack([upsample(lay[N_GROUND + k], factor, 0) for k in range(len(thing_ids))])
    dom, a = heat.argmax(0), heat.max(0)
    rgba = np.zeros(a.shape + (4,), np.float32)
    rgba[..., :3] = pal[np.array(thing_ids)][dom]
    rgba[..., 3] = np.clip(a, 0, 1) ** 1.3 * THEMES[theme]['glow']
    return rgba


def draw_scalebar(ax, size, theme, meters=10.0):
    t = THEMES[theme]
    n = meters / VOXEL_SIZE
    x1 = size - 12
    x0 = x1 - n
    yb = size - 12
    ax.add_patch(Rectangle((x0, yb - 1.6), n, 3.2, facecolor=t['text'], edgecolor='none',
                           zorder=6))
    ax.text((x0 + x1) / 2, yb - 5, f'{meters:g} m', ha='center', va='bottom',
            fontsize=8, color=t['text'], zorder=6,
            path_effects=[pe.withStroke(linewidth=2.2, foreground=t['bg'])])


def legend_handles(pal, thing_ids, present=None):
    ground = [Patch(facecolor=pal[g], edgecolor='none', label=n)
              for g, n in zip(GROUND_IDS, GROUND_NAMES)]
    ids = [c for c in thing_ids if present is None or c in present]
    things = [Line2D([], [], ls='none', marker=MARKERS[c], ms=MARKER_SIZE[c],
                     mfc=pal[c], mec='white', mew=0.9, label=THING_NAMES[c])
              for c in ids]
    return ground, things


def add_legend(fig, pal, thing_ids, present, theme, loc='lower center', y=0.0):
    t = THEMES[theme]
    ground, things = legend_handles(pal, thing_ids, present)
    handles = ground + things
    leg = fig.legend(handles=handles, loc=loc, bbox_to_anchor=(0.5, y),
                     ncol=min(len(handles), 6), frameon=False, fontsize=8,
                     handlelength=1.1, handleheight=1.0, columnspacing=1.1,
                     handletextpad=0.45, labelcolor=t['text'])
    return leg


def render(lay, out, theme='light', legend='present', factor=4, smooth=0.7,
           scalebar=True, threshold=0.3, size_in=3.4, dpi=300, formats=('pdf', 'png')):
    pal = palette()
    thing_ids = thing_ids_for_channels(lay.shape[0])
    set_thing_ids(thing_ids)
    req = centers_from_layout(lay, threshold=threshold)
    t = THEMES[theme]
    S = lay.shape[1]

    has_legend = legend != 'none'
    fig_h = size_in + (0.55 if has_legend else 0)
    fig = plt.figure(figsize=(size_in, fig_h), facecolor=t['bg'])
    bottom = 0.55 / fig_h if has_legend else 0
    ax = fig.add_axes([0, bottom, 1, size_in / fig_h])
    ax.set_facecolor(t['bg'])

    # pixel centres stay on voxel coordinates whatever the upsampling factor
    ext = (-0.5, S - 0.5, S - 0.5, -0.5)
    ax.imshow(ground_rgba(lay, pal, theme, factor, smooth), extent=ext,
              interpolation='bilinear', zorder=1)
    ax.imshow(glow_rgba(lay, thing_ids, pal, theme, factor), extent=ext,
              interpolation='bilinear', zorder=2)

    shadow = [pe.SimpleLineShadow(offset=(0.7, -0.7), alpha=0.35, shadow_color='black'),
              pe.Normal()]
    for x, yy, c in req:
        ax.plot(yy, x, ls='none', marker=MARKERS.get(c, 'o'), ms=MARKER_SIZE.get(c, 6.5),
                mfc=pal[c], mec=t['edge'], mew=0.9, zorder=5, path_effects=shadow)

    if scalebar:
        draw_scalebar(ax, S, theme)

    ax.set_xlim(-0.5, S - 0.5); ax.set_ylim(S - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(t['frame']); s.set_linewidth(0.6)

    if has_legend:
        present = None if legend == 'all' else {c for _, _, c in req}
        add_legend(fig, pal, thing_ids, present, theme, y=0.0)

    os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)
    written = []
    for f in formats:
        p = f'{out}.{f}'
        fig.savefig(p, dpi=dpi, facecolor=t['bg'], bbox_inches='tight', pad_inches=0.02)
        written.append(p)
    plt.close(fig)
    return written, req


def render_legend_only(out, n_channels=11, theme='light', dpi=300, formats=('pdf', 'png')):
    """A standalone legend strip, for figures that tile several layout panels."""
    pal = palette()
    thing_ids = thing_ids_for_channels(n_channels)
    t = THEMES[theme]
    fig = plt.figure(figsize=(7.0, 0.35), facecolor=t['bg'])
    add_legend(fig, pal, thing_ids, None, theme, loc='center', y=0.5)
    written = []
    for f in formats:
        p = f'{out}.{f}'
        fig.savefig(p, dpi=dpi, facecolor=t['bg'], bbox_inches='tight', pad_inches=0.02)
        written.append(p)
    plt.close(fig)
    return written


@click.command()
@click.option('--path', '-p', type=str,
              default="Q:/OWM_workspace/panscene_Dev_savings/exp/Layout/panscene_layout_e3",
              help="Root holding <seq>/<frame>.npz layouts.")
@click.option('--scene', type=str, default='08/004040', help="<seq>/<frame>")
@click.option('--out', type=str, default=None,
              help="Output stem, extensions added. Default figs/<seq>_<frame>.")
@click.option('--theme', type=click.Choice(list(THEMES)), default='dark')
@click.option('--legend', type=click.Choice(['present', 'all', 'none']), default='none',
              help="Legend classes: only those in the frame, all of them, or no legend.")
@click.option('--legend_only', is_flag=True, help="Write just the legend strip.")
@click.option('--smooth', type=float, default=0.7,
              help="Sub-voxel blur (voxels) before upsampling the ground masks. 0 = raw voxels.")
@click.option('--factor', type=int, default=4, help="Upsampling factor for the raster layers.")
@click.option('--no_scalebar', is_flag=True)
@click.option('--size', type=float, default=3.4, help="Panel width in inches (3.4 = one column).")
@click.option('--fmt', type=str, default='pdf,png', help="Comma-separated output formats.")
def main(path, scene, out, theme, legend, legend_only, smooth, factor, no_scalebar, size, fmt):
    formats = tuple(f.strip() for f in fmt.split(','))
    if legend_only:
        for p in render_legend_only(out or 'figs/legend', theme=theme, formats=formats):
            print(f'wrote {p}')
        return

    seq, frame = scene.split('/')
    lay = np.load(os.path.join(path, seq, f'{frame}.npz'))['layout']
    lay = dequantise(lay) if lay.dtype == np.uint8 else lay.astype(np.float32)
    written, req = render(lay, out or f'figs/{seq}_{frame}', theme=theme, legend=legend,
                          factor=factor, smooth=smooth, scalebar=not no_scalebar,
                          size_in=size, formats=formats)
    for p in written:
        print(f'wrote {p}')
    print(f'{len(req)} requested objects: '
          + ', '.join(THING_NAMES.get(c, str(c)) for _, _, c in req))


if __name__ == '__main__':
    main()
