"""
Turn an image into a rotatable binary obstacle mask for the wind tunnel.

Best results: a PNG with a transparent background around the object (a
silhouette/logo/CAD export). The alpha channel is used directly as the mask.
Also works with a JPG/PNG of an object on a plain light or dark background;
in that case brightness is thresholded to guess the silhouette.
"""
import numpy as np
from PIL import Image, ImageOps


def _pad_into_canvas(obj_mask_img, canvas_size):
    """Center obj_mask_img (already sized to fit the object) into an empty
    canvas_size x canvas_size buffer, leaving margin for safe rotation."""
    canvas = Image.new("L", (canvas_size, canvas_size), 0)
    ox = (canvas_size - obj_mask_img.width) // 2
    oy = (canvas_size - obj_mask_img.height) // 2
    canvas.paste(obj_mask_img, (ox, oy))
    return canvas


def load_mask_from_image(path, obj_size, canvas_size):
    """
    Load an image and return a (canvas_size x canvas_size) PIL 'L' image mask.
    The object itself is scaled to fit within obj_size x obj_size, then
    centered in the larger canvas_size buffer so nothing clips when rotated.
    """
    img = Image.open(path).convert("RGBA")
    alpha = np.array(img.getchannel("A"))

    if alpha.min() < 250:
        # real transparency present - use it directly
        mask = alpha > 128
    else:
        # no transparency - threshold on brightness, object assumed to be
        # the minority / higher-contrast region against its background
        gray = np.array(img.convert("L")).astype(np.float64)
        thresh = gray.mean()
        mask = gray < thresh
        if mask.mean() > 0.5:
            mask = ~mask

    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img.thumbnail((obj_size, obj_size), Image.LANCZOS)  # fit the OBJECT, not the canvas
    return _pad_into_canvas(mask_img, canvas_size)


def default_shape_mask(obj_size, canvas_size):
    """Simple built-in ellipse, used when no image is supplied."""
    y, x = np.ogrid[:obj_size, :obj_size]
    cx = cy = obj_size / 2
    rx, ry = obj_size * 0.46, obj_size * 0.16
    mask = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0
    obj_img = Image.fromarray((mask * 255).astype(np.uint8))
    return _pad_into_canvas(obj_img, canvas_size)


def mirror_mask(mask_img):
    """Flip the mask horizontally (image left-right) - use when an imported
    object's nose/front is pointing the wrong way."""
    return ImageOps.mirror(mask_img)


def rotate_mask(base_mask_img, angle_degrees):
    """
    Rotate the base mask and return a boolean numpy array indexed [x, y] to
    match the simulation domain's convention (axis0 = flow direction,
    axis1 = cross-stream). PIL/numpy images are natively [row, col] =
    [height, width] = [y, x], so this transposes after rotating.
    """
    rotated = base_mask_img.rotate(
        -angle_degrees, resample=Image.BILINEAR, expand=False, fillcolor=0
    )
    return np.array(rotated).T > 128


def place_mask(domain_shape, small_mask_bool, anchor_x, anchor_y):
    """Place a small square mask (already [x, y]-oriented) into the full
    (nx, ny) domain, centered at anchor."""
    nx, ny = domain_shape
    ts = small_mask_bool.shape[0]
    full = np.zeros((nx, ny), dtype=np.bool_)

    x0, y0 = anchor_x - ts // 2, anchor_y - ts // 2
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x0 + ts, nx), min(y0 + ts, ny)
    sx0, sy0 = x0c - x0, y0c - y0
    sx1, sy1 = sx0 + (x1c - x0c), sy0 + (y1c - y0c)

    full[x0c:x1c, y0c:y1c] = small_mask_bool[sx0:sx1, sy0:sy1]
    return full
