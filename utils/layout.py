"""Bird's-eye layout condition: road structure plus the objects to place.

A layout is a *set* of objects, so it cannot be conditioned on as a list any
more than instance ids could be generated as labels -- renumbering the requests
must not change the condition. Both are resolved the same way: rasterise into a
permutation-invariant field.

    channels  0-3   ground masks   road, parking, sidewalk, other-ground
              4-11  center heatmap per thing class, gaussian at each centroid
              12    centroid height z/(Z-1) at the dominant peak

Buildings and vegetation are deliberately absent. The user specifies the road
network and the object population; the model invents everything else.
Conditioning on the full semantic scene would make the task reconstruction
rather than generation.

Requested centers come from the same `keep` mask the field supervision uses, so
the condition only ever asks for objects the model was trained to produce.
"""
import numpy as np


THING_IDS = (1, 2, 3, 4, 5, 6)
ALL_THING_IDS = (1, 2, 3, 4, 5, 6, 7, 8) 
SIGMA = 2.0

MAX_EXTENT = {
    1: 30,   # car            (1500 static examples)
    2: 20,   # bicycle        (105)
    3: 15,   # motorcycle     (53)
    4: 51,   # truck          (30)
    5: 41,   # other-vehicle  (122)
    6: 10,   # person         (61)
    7: 15,   # bicyclist      (0 static examples; 1.8 m + slop)
    8: 15,   # motorcyclist   (1 static example;  2.2 m + slop)
}

def instance_stats(ins, sem, moving=None, thing_ids=THING_IDS,
                   max_extent=MAX_EXTENT):
    """Per-instance centroid, extent, class and whether to supervise it.

    Args:
        ins:    [X, Y, Z] instance ids, 0 for stuff/empty.
        sem:    [X, Y, Z] semantic train-ids.
        moving: [X, Y, Z] bool from `moving_mask_from_raw`, or None to skip the
                dynamic-object exclusion (tests and ablations only).

    Returns dict with, per instance id in `uid`:
        coords     [M, 3] voxel coordinates of every instance voxel
        inv        [M]    index into `uid` for each coordinate
        centroid   [K, 3] float centroid
        extent     [K, 3] bounding-box size in voxels
        cls        [K]    semantic class
        moving     [K]    bool, majority of voxels carry a moving raw class
        keep       [K]    bool, supervise this instance's fields
    """
    occ = ins > 0
    coords = np.argwhere(occ).astype(np.float32)
    ids = ins[occ]

    if coords.shape[0] == 0:
        return dict(coords=coords, inv=np.zeros(0, np.int64),
                    uid=np.zeros(0, ins.dtype), centroid=np.zeros((0, 3), np.float32),
                    extent=np.zeros((0, 3), np.int64), cls=np.zeros(0, np.int64),
                    moving=np.zeros(0, bool), keep=np.zeros(0, bool))

    uid, inv = np.unique(ids, return_inverse=True)
    k = uid.shape[0]
    n = np.bincount(inv, minlength=k).astype(np.float32)

    centroid = np.stack(
        [np.bincount(inv, weights=coords[:, d], minlength=k) / n for d in range(3)],
        axis=1).astype(np.float32)

    lo = np.full((k, 3), np.inf, np.float32)
    hi = np.full((k, 3), -np.inf, np.float32)
    for d in range(3):
        np.minimum.at(lo[:, d], inv, coords[:, d])
        np.maximum.at(hi[:, d], inv, coords[:, d])
    extent = (hi - lo + 1).astype(np.int64)

    # Instances are semantically pure (verified: majority-class fraction is
    # exactly 1.0 over 1795 instances), so any voxel gives the class.
    first = np.zeros(k, np.int64)
    first[inv[::-1]] = np.arange(coords.shape[0])[::-1]
    ci = coords[first].astype(np.int64)
    cls = sem[ci[:, 0], ci[:, 1], ci[:, 2]].astype(np.int64)

    if moving is None:
        mov = np.zeros(k, bool)
    else:
        mv = moving[occ].astype(np.float32)
        mov = (np.bincount(inv, weights=mv, minlength=k) / n) > 0.5

    limit = np.array([max_extent.get(int(c), 0) for c in cls])
    keep = np.isin(cls, thing_ids) & ~mov & (extent.max(axis=1) <= limit)

    return dict(coords=coords, inv=inv, uid=uid, centroid=centroid,
                extent=extent, cls=cls, moving=mov, keep=keep)



# Train ids of the ground classes that make up the road layout.
GROUND_IDS = (9, 10, 11, 12)          # road, parking, sidewalk, other-ground

N_GROUND = len(GROUND_IDS)
N_THING = len(THING_IDS)
LAYOUT_CHANNELS = N_GROUND + N_THING + 1
HEIGHT_CHANNEL = LAYOUT_CHANNELS - 1


def set_thing_ids(ids):
    """Reconfigure the layout for a different thing-class set, at run time.

    Dropping bicyclist and motorcyclist took the layout from 13 channels to 11,
    which makes every checkpoint trained before that change unloadable: its
    layout encoder expects 13. Rather than strand those models, callers can
    restore the old configuration for the lifetime of the process.

    Mutates the module-level constants, so call it before building anything
    that reads them.
    """
    global THING_IDS, N_THING, LAYOUT_CHANNELS, HEIGHT_CHANNEL
    THING_IDS = tuple(ids)
    N_THING = len(THING_IDS)
    LAYOUT_CHANNELS = N_GROUND + N_THING + 1
    HEIGHT_CHANNEL = LAYOUT_CHANNELS - 1
    return LAYOUT_CHANNELS


def thing_ids_for_channels(n_channels):
    """The thing-class set implied by a stored layout's channel count."""
    n_thing = n_channels - N_GROUND - 1
    if n_thing == len(ALL_THING_IDS):
        return ALL_THING_IDS
    if n_thing == 6:
        return (1, 2, 3, 4, 5, 6)
    raise ValueError(f'layout has {n_channels} channels, implying {n_thing} '
                     f'thing classes, which matches no known configuration')


def rasterise_centers(centers, classes, heights, shape, sigma=SIGMA):
    """Per-class BEV gaussian heatmaps plus a height channel.

    Args:
        centers: [K, 2] float (x, y) centroid positions in voxels.
        classes: [K] thing-class train ids.
        heights: [K] centroid z, normalised to [0, 1].

    Returns [N_THING + 1, X, Y] float32.
    """
    X, Y = shape
    out = np.zeros((N_THING + 1, X, Y), np.float32)
    best = np.zeros((X, Y), np.float32)      # strongest peak seen, for the height channel

    rad = 4.0 * sigma
    two_s2 = 2.0 * sigma * sigma
    for (cx, cy), cls, h in zip(centers, classes, heights):
        if int(cls) not in THING_IDS:
            continue
        lo = np.maximum(np.ceil(np.array([cx, cy]) - rad).astype(np.int64), 0)
        hi = np.minimum(np.floor(np.array([cx, cy]) + rad).astype(np.int64) + 1, (X, Y))
        if np.any(lo >= hi):
            continue
        gx = np.arange(lo[0], hi[0], dtype=np.float32)[:, None]
        gy = np.arange(lo[1], hi[1], dtype=np.float32)[None, :]
        g = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / two_s2)

        ch = THING_IDS.index(int(cls))
        win = out[ch, lo[0]:hi[0], lo[1]:hi[1]]
        np.maximum(win, g, out=win)

        # Height of whichever centre dominates this cell. Within `out` the
        # height channel is the last one, after the N_THING class maps.
        bwin = best[lo[0]:hi[0], lo[1]:hi[1]]
        hwin = out[N_THING, lo[0]:hi[0], lo[1]:hi[1]]
        take = g > bwin
        hwin[take] = h
        bwin[take] = g[take]
    return out


def build_layout(sem, ins, moving=None, sigma=SIGMA):
    """Layout condition for one scene, from its labels.

    Returns [LAYOUT_CHANNELS, X, Y] float32 in [0, 1].
    """
    X, Y, Z = sem.shape
    layout = np.zeros((LAYOUT_CHANNELS, X, Y), np.float32)

    for i, c in enumerate(GROUND_IDS):
        layout[i] = (sem == c).any(axis=2).astype(np.float32)

    st = instance_stats(ins, sem, moving)
    keep = st['keep']
    if keep.any():
        cen = st['centroid'][keep]
        obj = rasterise_centers(cen[:, :2], st['cls'][keep],
                                cen[:, 2] / max(Z - 1, 1), (X, Y), sigma)
        layout[N_GROUND:] = obj
    return layout


def quantise(layout):
    """float [0,1] -> uint8, for storage. ~0.85 MB per frame."""
    return np.clip(layout * 255.0 + 0.5, 0, 255).astype(np.uint8)


def dequantise(layout_u8):
    return layout_u8.astype(np.float32) / 255.0


def augment_layout(layout, transpose=False, flip_x=False, flip_y=False):
    """Apply the same geometric transform the triplane received.

    The triplane augmentation also does crop-resize and gaussian noise. Noise is
    latent-space only and needs no counterpart here; crop-resize is skipped
    entirely for conditional training, because a condition that does not match
    its target is worse than weaker augmentation.
    """
    if transpose:
        layout = np.swapaxes(layout, 1, 2)
    if flip_x:
        layout = layout[:, ::-1]
    if flip_y:
        layout = layout[:, :, ::-1]
    return np.ascontiguousarray(layout)


def centers_from_layout(layout, threshold=0.3, nms_size=3):
    """Recover the requested (position, class) set from a layout tensor.

    Used by the adherence metric and to author layouts by hand: whatever is
    read back here is exactly what the generator will be scored against.
    """
    from scipy.ndimage import label, maximum_filter

    req = []
    for ch, cls in enumerate(THING_IDS):
        hm = layout[N_GROUND + ch]
        peak = (hm >= maximum_filter(hm, size=nms_size)) & (hm > threshold)
        lab, k = label(peak, structure=np.ones((3, 3)))
        if k == 0:
            continue
        idx = np.argwhere(peak).astype(np.float32)
        grp = lab[peak] - 1
        n = np.bincount(grp, minlength=k).astype(np.float32)
        pos = np.stack([np.bincount(grp, weights=idx[:, d], minlength=k) / n
                        for d in range(2)], axis=1)
        for p in pos:
            req.append((float(p[0]), float(p[1]), int(cls)))
    return req


# --------------------------------------------------------------------------
# Authoring: build or edit a layout directly, rather than deriving one from a
# scene. This is the application path -- specify the objects you want and the
# road they sit on, and the generator realises them.
# --------------------------------------------------------------------------

def layout_from_objects(objects, road_mask=None, shape=(256, 256), z=0.2,
                        sigma=SIGMA):
    """Compose a layout condition from an explicit object list.

    Args:
        objects: iterable of (x, y, cls) or (x, y, cls, z_norm). Positions are
            in voxels on the BEV grid; cls is a thing-class train id.
        road_mask: [4, X, Y] ground channels, or [X, Y] treated as road, or
            None for no road constraint. Taking the road from a real scene and
            repopulating it is the usual way to author a controlled experiment.
        z: default centroid height, normalised, when not given per object.

    Returns [LAYOUT_CHANNELS, X, Y] float32.
    """
    X, Y = shape
    layout = np.zeros((LAYOUT_CHANNELS, X, Y), np.float32)
    if road_mask is not None:
        road_mask = np.asarray(road_mask, np.float32)
        if road_mask.ndim == 2:
            layout[0] = road_mask
        else:
            layout[:N_GROUND] = road_mask[:N_GROUND]

    objects = list(objects)
    if objects:
        pos = np.array([[o[0], o[1]] for o in objects], np.float32)
        cls = np.array([o[2] for o in objects])
        hts = np.array([o[3] if len(o) > 3 else z for o in objects], np.float32)
        layout[N_GROUND:] = rasterise_centers(pos, cls, hts, (X, Y), sigma)
    return layout


def edit_layout(layout, add=(), remove_near=(), radius=4.0, sigma=SIGMA):
    """Add or remove objects, keeping the road channels untouched.

    `remove_near` drops any requested object within `radius` voxels of the given
    positions. Rebuilding from the surviving set is deliberate: erasing a
    gaussian in place would leave a residual shoulder that `centers_from_layout`
    could still read back as a request.
    """
    objs = [(x, y, c) for x, y, c in centers_from_layout(layout)]
    for rx, ry in remove_near:
        objs = [o for o in objs if np.hypot(o[0] - rx, o[1] - ry) > radius]
    objs += [tuple(o) for o in add]

    out = layout.copy()
    out[N_GROUND:] = 0.0
    if objs:
        pos = np.array([[o[0], o[1]] for o in objs], np.float32)
        cls = np.array([o[2] for o in objs])
        hts = np.array([o[3] if len(o) > 3 else 0.2 for o in objs], np.float32)
        out[N_GROUND:] = rasterise_centers(pos, cls, hts, layout.shape[1:], sigma)
    return out


#: Ground classes each thing class actually sits on, measured over 100 cleaned
#: scenes. Requesting a bicycle in the middle of the carriageway is an
#: out-of-distribution ask, and scoring the answer as a class-control failure
#: confuses the model with the question.
#:   car 44% road / 25% sidewalk / 25% parking   truck 75% road / 25% parking
#:   bicycle 91% sidewalk   motorcycle 88% sidewalk   person 57% sidewalk, 4% road
CLASS_SURFACES = {
    1: (9, 10, 11),   # car           road, parking, sidewalk
    2: (11,),         # bicycle       sidewalk
    3: (11,),         # motorcycle    sidewalk
    4: (9, 10),       # truck         road, parking
    5: (9, 11, 12),   # other-vehicle sidewalk, road, other-ground
    6: (11,),         # person        sidewalk
    7: (9, 11),       # bicyclist
    8: (9, 11),       # motorcyclist
}


def sample_positions_on_road(layout, n, cls, rng, min_sep=8.0, max_tries=2000,
                             surface_ids=None):
    """Draw `n` object positions on ground surface, spaced at least `min_sep`.

    min_sep defaults to 8 voxels (1.6 m): the 5th-percentile real same-class
    centroid separation is 4.3 voxels, and packing requests tighter than the
    matching gate would make adherence unmeasurable by construction.

    surface_ids restricts placement to those ground classes -- pass
    CLASS_SURFACES[cls] to make the request in-distribution for that class.
    Default is every ground surface, which is the historical behaviour.
    """
    if surface_ids:
        idx = [GROUND_IDS.index(g) for g in surface_ids if g in GROUND_IDS]
        road = layout[idx].max(0) > 0.5 if idx else np.zeros(layout.shape[1:], bool)
        if not road.any():                       # requested surface absent
            road = layout[:N_GROUND].max(0) > 0.5
    else:
        road = layout[:N_GROUND].max(0) > 0.5
    cand = np.argwhere(road)
    if len(cand) == 0:
        return []
    chosen = []
    for _ in range(max_tries):
        if len(chosen) >= n:
            break
        p = cand[rng.integers(len(cand))].astype(np.float32)
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) >= min_sep for q in chosen):
            chosen.append(p)
    return [(float(p[0]), float(p[1]), cls) for p in chosen]
