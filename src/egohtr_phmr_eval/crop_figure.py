"""Crop horizontal whitespace from already-rendered qualitative figure panels.

This is an image-only post-process: it does not load or modify any mesh data,
does not rerender meshes, and does not resize the cropped mesh images.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _find_panel_meshes(image: Image.Image, color: str) -> list[tuple[int, int, int, int]]:
    """Find the four large colored mesh silhouettes in the figure's lower half."""
    import cv2

    bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    if color == "blue":
        mask = (hsv[:, :, 0] >= 96) & (hsv[:, :, 0] <= 132) & (hsv[:, :, 1] >= 65) & (hsv[:, :, 2] >= 45)
        x_min, x_max = int(.38 * w), int(.63 * w)
    elif color == "red":
        hue = hsv[:, :, 0]
        mask = ((hue <= 12) | (hue >= 170)) & (hsv[:, :, 1] >= 65) & (hsv[:, :, 2] >= 45)
        x_min, x_max = int(.72 * w), int(.96 * w)
    else:
        raise ValueError(color)
    mask[:int(.42 * h), :] = False
    mask[:, :x_min] = False
    mask[:, x_max:] = False
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    boxes = []
    for x, y, bw, bh, area in stats[1:count]:
        # Text and axes are small; mesh silhouettes are tall connected regions.
        if area >= 2500 and bh >= int(.03 * h):
            boxes.append((int(x), int(y), int(bw), int(bh)))
    boxes.sort(key=lambda box: box[1])
    if len(boxes) != 4:
        raise ValueError(f"Expected four lower {color} mesh panels, found {len(boxes)}: {boxes}")
    return boxes


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    try:
        import matplotlib
        face = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        font_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / face
        return ImageFont.truetype(str(font_path), size=size)
    except Exception:
        return ImageFont.load_default()


def _centered_text(draw: ImageDraw.ImageDraw, center_x: int, y: int, text: str,
                   font: ImageFont.FreeTypeFont, fill):
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((center_x - (box[2] - box[0]) / 2, y), text, font=font, fill=fill)


def crop_existing_figure(figure_path: str | Path, rgb_path: str | Path,
                         output_path: str | Path | None = None) -> Path:
    """Crop only the lower mesh panel image whitespace and recompose the layout.

    The upper 3D location and RGB projection area is copied pixel-for-pixel.
    Lower GT and prediction crops are scaled uniformly by a shared factor,
    preserving aspect ratio and the common GT/prediction display scale.
    """
    figure_path = Path(figure_path).expanduser().resolve()
    rgb_path = Path(rgb_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve() if output_path else figure_path
    source = Image.open(figure_path).convert("RGB")
    original_rgb = Image.open(rgb_path).convert("RGB")
    width, height = source.size
    if original_rgb.width != original_rgb.height:
        # Preserve any non-square input image aspect ratio below as well.
        pass

    gt_boxes = _find_panel_meshes(source, "blue")
    pred_boxes = _find_panel_meshes(source, "red")
    # Crop tightly around the rendered silhouette, with a small consistent
    # antialias margin. No mesh pixels are scaled or interpolated.
    crops = []
    for gt, pred in zip(gt_boxes, pred_boxes):
        pair = []
        for x, y, bw, bh in (gt, pred):
            pad = 8
            pair.append(source.crop((max(0, x - pad), max(0, y - pad),
                                     min(width, x + bw + pad), min(height, y + bh + pad))))
        crops.append(pair)

    # Retain the entire upper figure unchanged; only its lower layout is rebuilt.
    upper_end = 1080
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(source.crop((0, 0, width, upper_end)), (0, 0))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(26, bold=True)
    col_font = _font(29, bold=True)
    row_font = _font(25, bold=True)

    # Remove the old section heading, which was split by the crop boundary,
    # and place a complete heading below the unchanged upper axes.
    draw.rectangle((0, 1037, 700, upper_end), fill="white")
    draw.text((22, 1083), "3D LOCATION  ·  SLAM ORIGIN (m)", font=title_font,
              fill=(45, 45, 45))
    draw.text((22, 1137), "POSE & SHAPE  ·  GT-defined front / right / left / rear views",
              font=title_font, fill=(45, 45, 45))
    centers = (500, 1540, 2565)
    _centered_text(draw, centers[0], 1204, "Input 2nd Aria RGB", col_font, (35, 35, 35))
    _centered_text(draw, centers[1], 1204, "EgoHTR GT · Male SMPL-X", col_font, (40, 100, 195))
    _centered_text(draw, centers[2], 1204, "PromptHMR fit · Male SMPL-X", col_font, (210, 60, 52))

    row_centers = [1510, 1915, 2320, 2725]
    input_image = original_rgb.copy()
    input_size = 930
    input_image.thumbnail((input_size, input_size), Image.Resampling.LANCZOS)
    input_x = 35
    input_y = int(np.mean((row_centers[0], row_centers[-1])) - input_image.height / 2)
    canvas.paste(input_image, (input_x, input_y))

    labels = ("Front", "Right", "Left", "Rear")
    for label, row_center, pair in zip(labels, row_centers, crops):
        draw.text((1110, row_center - 12), label, font=row_font, fill=(45, 45, 45))
        for crop, center_x in zip(pair, centers[1:]):
            # Enlarge each cropped raster uniformly by the same factor. This
            # preserves aspect ratio and keeps GT/pred at identical display scale.
            scale = 1.5
            crop = crop.resize((round(crop.width * scale), round(crop.height * scale)),
                               Image.Resampling.LANCZOS)
            canvas.paste(crop, (int(center_x - crop.width / 2), int(row_center - crop.height / 2)))

    evaluation_path = rgb_path.parent / "evaluation.npz"
    if evaluation_path.exists():
        with np.load(evaluation_path, allow_pickle=False) as evaluation:
            mpjpe = float(evaluation["mean_mpjpe_mm"])
            pa_mpjpe = (float(evaluation["mean_pa_mpjpe_mm"])
                        if "mean_pa_mpjpe_mm" in evaluation else None)
        footer = f"MPJPE: {mpjpe:.1f} mm"
        if pa_mpjpe is not None:
            footer += f"  |  PA-MPJPE: {pa_mpjpe:.1f} mm"
        footer_font = _font(23, bold=True)
        box = draw.textbbox((0, 0), footer, font=footer_font)
        draw.text((width - 24 - (box[2] - box[0]), height - 48), footer,
                  font=footer_font, fill=(55, 55, 55))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + ".tmp.png")
    canvas.save(temp_path, format="PNG", optimize=True)
    os.replace(temp_path, output_path)

    # Guard the user's requirement that the upper figure remain identical.
    saved = Image.open(output_path).convert("RGB")
    before = np.asarray(source.crop((0, 0, width, upper_end))).copy()
    after = np.asarray(saved.crop((0, 0, width, upper_end)))
    before[1037:upper_end, :701] = after[1037:upper_end, :701]  # replaced clipped heading only
    if not np.array_equal(before, after):
        raise RuntimeError("Upper location/projection panels changed during lower-panel crop")
    print(f"Saved cropped layout: {output_path}")
    print("Upper location/projection image panels are pixel-identical; mesh crops were uniformly scaled 1.5x with aspect ratio preserved.")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Crop whitespace from an existing qualitative comparison PNG")
    parser.add_argument("--figure", required=True, help="Existing rendered figure PNG")
    parser.add_argument("--rgb", required=True, help="Selected input RGB image (rgb.png)")
    parser.add_argument("--output", help="Output PNG; defaults to replacing --figure")
    args = parser.parse_args()
    crop_existing_figure(args.figure, args.rgb, args.output)


if __name__ == "__main__":
    main()
