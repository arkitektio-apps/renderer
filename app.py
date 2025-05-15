from typing import AsyncGenerator, Optional
from arkitekt_next import startup, register
import time
from mikro_next.api.schema import (
    File,
    Stage,
    Snapshot,
    Image,
    from_array_like,
    PartialAffineTransformationViewInput,
    PartialRGBViewInput,
    create_snapshot,
    ColorMap,
    awatch_rois,
    awatch_files,
    awatch_images,
)
from matplotlib import cm
import xarray as xr
import numpy as np
import uuid
import os
from mikro_next.api.schema import create_stage
from PIL import Image as PILImage

def red_map(x):
    # Normalize to [0,1] and return red-only RGB image.
    x_norm = x.astype(np.float32) / 255.0
    red = x_norm
    green = np.zeros_like(x_norm)
    blue = np.zeros_like(x_norm)
    return np.dstack((red, green, blue))

def green_map(x):
    # Normalize to [0,1] and return green-only RGB image.
    x_norm = x.astype(np.float32) / 255.0
    red = np.zeros_like(x_norm)
    green = x_norm
    blue = np.zeros_like(x_norm)
    return np.dstack((red, green, blue))

def blue_map(x):
    # Normalize to [0,1] and return blue-only RGB image.
    x_norm = x.astype(np.float32) / 255.0
    red = np.zeros_like(x_norm)
    green = np.zeros_like(x_norm)
    blue = x_norm
    return np.dstack((red, green, blue))

cm_colormaps = {
    ColorMap.INFERNO: lambda data: cm.inferno(data.astype(np.float32) / 255.0),
    ColorMap.VIRIDIS: lambda data: cm.viridis(data.astype(np.float32) / 255.0),
    ColorMap.MAGMA: lambda data: cm.magma(data.astype(np.float32) / 255.0),
    ColorMap.PLASMA: lambda data: cm.plasma(data.astype(np.float32) / 255.0),
    ColorMap.RED: red_map,
    ColorMap.GREEN: green_map,
    ColorMap.BLUE: blue_map,
}

@register
def render(image: Image) -> Snapshot:
    """Render

    This function renders the image

    Parameters
    ----------
    image : Image
        The image

    Returns
    -------
    Thumbnail
        The thumbnail
    """

    context = image.rgb_contexts[0]

    t_sel = image.data.sel(t=0)
    max_proj = t_sel.max(dim="z")

    r_g_b = np.zeros((max_proj.y.size, max_proj.x.size, 3), dtype=np.uint8)

    for view in context.views:
        view_data = (
            max_proj.isel(c=[view.c_min, view.c_min]).sum(dim="c").data.compute()
        )

        print(view_data.shape)

        if view.contrast_limit_max or view.contrast_limit_min:
            view_data = np.clip(
                view_data, view.contrast_limit_min, view.contrast_limit_max
            )

        if view.rescale is True:
            vmin, vmax = view_data.min(), view_data.max()
            new_data = np.interp(view_data, (vmin, vmax), (0, 255)).astype(np.uint8)
        else:
            # Check if view_data is an integer or float type and scale accordingly
            if np.issubdtype(view_data.dtype, np.integer):
                max_value = np.iinfo(view_data.dtype).max
            else:
                max_value = view_data.max()  # For float types, use the actual max value

            view_data = (view_data / max_value) * 255
            new_data = view_data.astype(np.uint8)

        new_data = new_data.astype(np.uint8)  # Ensure dtype conversion
        rgb = np.array(view.base_color[:3]).reshape(1, 1, 3) / 255.0

        if view.color_map == ColorMap.INTENSITY:
            # Expand dims so that new_data shape becomes (M, N, 1)
            r_g_b += (new_data[:, :, np.newaxis] * rgb).astype(np.uint8)
        else:
            cmap = cm_colormaps.get(view.color_map)
            if cmap is not None:
                # For matplotlib colormaps and our own, ensure the input is normalized.
                cmap_output = (cmap(new_data) * 255).astype(np.uint8)
                r_g_b += cmap_output[:, :, :3]  # Drop the alpha channel if present
            else:
                raise NotImplementedError(f"Color Map not implemented: {view.color_map}")

    print(r_g_b.shape)
    print(r_g_b.flatten().max())

    assert r_g_b.dtype == np.uint8
    assert r_g_b.min() >= 0 and r_g_b.max() <= 255

    print(r_g_b.shape)

    temp_file = uuid.uuid4().hex + ".jpg"

    aspect = r_g_b.shape[0] / r_g_b.shape[1]

    img = PILImage.fromarray(r_g_b)
    img = img.convert("RGB")

    if r_g_b.shape[0] > 512:
        img = img.resize((512, int(512 * aspect)), PILImage.Resampling.BILINEAR)
    img.save(temp_file, quality=80)

    th = create_snapshot(file=open(temp_file, "rb"), image=image)
    print("Done")
    os.remove(temp_file)
    return th