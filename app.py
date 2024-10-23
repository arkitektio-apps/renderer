from typing import AsyncGenerator, Optional
from arkitekt_next import startup, register
import time
from mikro_next.api import (
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


cm_colormaps = {
    ColorMap.INFERNO: cm.inferno,
    ColorMap.VIRIDIS: cm.viridis,
    ColorMap.MAGMA: cm.magma,
    ColorMap.PLASMA: cm.plasma,
    ColorMap.RED: cm.inferno,
    ColorMap.GREEN: cm.viridis,
    ColorMap.BLUE: cm.plasma,
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
            min, max = view_data.min(), view_data.max()
            new_data = np.interp(view_data, (min, max), (0, 255)).astype(np.uint8)
        else:
            # Check if view_data is an integer or float type and scale accordingly
            if np.issubdtype(view_data.dtype, np.integer):
                max_value = np.iinfo(view_data.dtype).max
            else:
                max_value = view_data.max()  # For float types, use the actual max value

            view_data = (view_data / max_value) * 255
            new_data = view_data.astype(np.uint8)

        new_data = new_data.astype(np.uint8)  # Ensure dtype conversion
        rgb = np.array(view.base_color[:3]).reshape(1, 1, 3) / 255

        if view.color_map == ColorMap.INTENSITY:
            r_g_b += (new_data[:, :, np.newaxis] * rgb).astype(np.uint8)
        else:
            cmap = cm_colormaps.get(view.color_map)
            if cmap is not None:
                cmap_output = (cmap(new_data) * 255).astype(np.uint8)
                r_g_b += cmap_output[:, :, :3]  # Drop the alpha channel (4th dimension)
            else:
                raise NotImplementedError(f"Color Map not implemented: {view.color_map}")

    print(r_g_b.shape)

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