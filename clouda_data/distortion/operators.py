from __future__ import annotations

import io
import math
import random
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from .base import Distortion, DistortionSpec
from .metadata import DistortionMetadata

_SEVERITY = {
    "none": 0.0,
    "minimal": 0.12,
    "light": 0.25,
    "medium": 0.5,
    "heavy": 0.75,
    "extreme": 1.0,
}


def _number(spec: DistortionSpec, key: str, default: float) -> float:
    value = float(spec.parameters.get(key, default))
    if not math.isfinite(value):
        raise ValueError(f"{spec.name}.{key} must be finite")
    return value


def _rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, "white")
        background.alpha_composite(image)
        return background.convert("RGB")
    return image.convert("RGB")


def _gradient(size: tuple[int, int], horizontal: bool = False) -> Image.Image:
    width, height = size
    length = width if horizontal else height
    band = Image.linear_gradient("L").resize((length, 1) if horizontal else (1, length))
    return band.resize(size)


def _noise(size: tuple[int, int], rng: random.Random) -> Image.Image:
    width, height = size
    return Image.frombytes("L", size, rng.randbytes(width * height))


def _region_boxes(
    context: dict[str, Any] | None, size: tuple[int, int]
) -> list[tuple[int, int, int, int]]:
    if not context:
        return []
    boxes: list[tuple[int, int, int, int]] = []
    for item in context.get("regions", []):
        bbox = (
            item.get("bbox") if isinstance(item, dict) else getattr(item, "bbox", None)
        )
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (int(float(value)) for value in bbox)
        boxes.append((max(0, x1), max(0, y1), min(size[0], x2), min(size[1], y2)))
    return [box for box in boxes if box[2] > box[0] and box[3] > box[1]]


class RealImageDistortion(Distortion):
    """Deterministic Pillow implementation shared by registered operators."""

    version = "1.0.0"

    def apply(
        self, image: Any, seed: int, context: dict[str, Any] | None = None
    ) -> tuple[Image.Image, DistortionMetadata]:
        self.validate_input(image)
        if not isinstance(image, Image.Image):
            raise TypeError("Real distortions require a Pillow image")
        rng = random.Random(seed)
        source = _rgb(ImageOps.exif_transpose(image))
        output, affected, details = self._apply(source, rng, context)
        if output.size[0] < 1 or output.size[1] < 1:
            raise ValueError("Distortion produced invalid dimensions")
        metadata = DistortionMetadata(
            name=self.spec.name,
            version=self.version,
            probability=self.spec.probability,
            severity=self.spec.severity,
            parameters=dict(self.spec.parameters),
            random_seed=seed,
            input_requirements=list(self.spec.input_requirements),
            output_metadata={
                "mode": "real_pixels",
                "text_reference_changed": False,
                "affected_regions": affected,
                "source_dimensions": list(source.size),
                "output_dimensions": list(output.size),
                **details,
            },
        )
        return output, metadata

    def _apply(
        self,
        image: Image.Image,
        rng: random.Random,
        context: dict[str, Any] | None,
    ) -> tuple[Image.Image, list[list[int]], dict[str, Any]]:
        name = self.spec.name
        strength = _SEVERITY[self.spec.severity]
        width, height = image.size
        regions = _region_boxes(context, image.size)
        affected = [list(box) for box in regions] or [[0, 0, width, height]]

        if strength == 0:
            return image.copy(), affected, {"layout_mode": "whole_page_fallback"}

        if name in {"rotation", "small_rotation", "skew", "page_misalignment"}:
            degrees = _number(self.spec, "degrees", 0.5 + 3.5 * strength)
            angle = rng.uniform(-degrees, degrees)
            return (
                image.rotate(
                    angle, Image.Resampling.BICUBIC, expand=False, fillcolor="white"
                ),
                affected,
                {"angle": angle},
            )

        if name in {
            "perspective",
            "page_curvature",
            "warping",
            "book_gutter_deformation",
        }:
            shift = max(1, int(min(width, height) * 0.025 * strength))
            coeffs = (
                1,
                rng.uniform(-0.02, 0.02) * strength,
                rng.randint(-shift, shift),
                rng.uniform(-0.02, 0.02) * strength,
                1,
                rng.randint(-shift, shift),
                0,
                0,
            )
            return (
                image.transform(
                    image.size,
                    Image.Transform.PERSPECTIVE,
                    coeffs,
                    Image.Resampling.BICUBIC,
                    fillcolor="white",
                ),
                affected,
                {"shift_pixels": shift},
            )

        if name in {"stretching", "non_uniform_scaling"}:
            sx = 1 + rng.uniform(-0.12, 0.12) * strength
            sy = 1 + rng.uniform(-0.12, 0.12) * strength
            resized = image.resize(
                (max(1, int(width * sx)), max(1, int(height * sy))),
                Image.Resampling.BICUBIC,
            )
            return (
                ImageOps.fit(
                    resized,
                    image.size,
                    method=Image.Resampling.BICUBIC,
                    centering=(0.5, 0.5),
                ),
                affected,
                {"scale": [sx, sy]},
            )

        if name in {"cropping", "edge_clipping", "partial_clipping"}:
            margin = max(1, int(min(width, height) * 0.04 * strength))
            side = rng.choice(("left", "right", "top", "bottom"))
            box = {
                "left": (margin, 0, width, height),
                "right": (0, 0, width - margin, height),
                "top": (0, margin, width, height),
                "bottom": (0, 0, width, height - margin),
            }[side]
            cropped = image.crop(box)
            return (
                ImageOps.pad(cropped, image.size, color="white"),
                affected,
                {"clipped_side": side},
            )

        if name == "padding":
            margin = max(1, int(min(width, height) * 0.04 * strength))
            padded = ImageOps.expand(image, border=margin, fill="white")
            return (
                padded.resize(image.size, Image.Resampling.LANCZOS),
                affected,
                {"padding": margin},
            )

        if name == "translation":
            dx = rng.randint(
                -max(1, int(width * 0.03 * strength)),
                max(1, int(width * 0.03 * strength)),
            )
            dy = rng.randint(
                -max(1, int(height * 0.03 * strength)),
                max(1, int(height * 0.03 * strength)),
            )
            return ImageChops.offset(image, dx, dy), affected, {"offset": [dx, dy]}

        if name in {
            "dpi_reduction",
            "downscale_upscale",
            "small_font_degradation",
            "footnote_degradation",
            "margin_note_degradation",
        }:
            scale = max(0.2, 1.0 - 0.65 * strength)
            small = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.Resampling.BILINEAR,
            )
            degraded = small.resize(image.size, Image.Resampling.BILINEAR)
            return self._regional(image, degraded, regions), affected, {"scale": scale}

        if name in {
            "blur",
            "gaussian_blur",
            "defocus_blur",
            "anisotropic_blur",
            "edge_blur",
            "local_blur",
            "text_region_blur",
        }:
            radius = _number(self.spec, "radius", 0.4 + 2.6 * strength)
            blurred = image.filter(ImageFilter.GaussianBlur(radius))
            if name == "edge_blur":
                mask = Image.new("L", image.size, 0)
                draw = ImageDraw.Draw(mask)
                margin = max(4, int(min(width, height) * 0.08))
                draw.rectangle((0, 0, width, height), outline=255, width=margin)
                return (
                    Image.composite(blurred, image, mask),
                    affected,
                    {"radius": radius},
                )
            return (
                self._regional(
                    image, blurred, regions if "local" in name or "text" in name else []
                ),
                affected,
                {"radius": radius},
            )

        if name == "motion_blur":
            offset = max(1, int(5 * strength))
            layers = [
                ImageChops.offset(image, i, 0) for i in range(-offset, offset + 1)
            ]
            result = layers[0]
            for index, layer in enumerate(layers[1:], start=2):
                result = Image.blend(result, layer, 1 / index)
            return result, affected, {"radius": offset}

        if name in {
            "gaussian_noise",
            "poisson_noise",
            "speckle_noise",
            "scanner_noise",
            "paper_texture",
            "paper_fiber",
            "old_paper_texture",
            "toner_scatter",
        }:
            noise = _noise(image.size, rng)
            noise_rgb = Image.merge("RGB", (noise, noise, noise))
            alpha = 0.025 + 0.13 * strength
            result = Image.blend(image, noise_rgb, alpha)
            return result, affected, {"noise_alpha": alpha}

        if name in {
            "salt_pepper_noise",
            "dust",
            "foxing",
            "stains",
            "isolated_dark_points",
            "isolated_light_points",
        }:
            result = image.copy()
            draw = ImageDraw.Draw(result, "RGBA")
            count = max(1, int(width * height * (0.00002 + 0.00015 * strength)))
            for _ in range(count):
                x, y = rng.randrange(width), rng.randrange(height)
                radius = rng.randint(1, max(1, int(3 * strength)))
                color = rng.choice(
                    ((20, 15, 10, 120), (120, 75, 30, 90), (255, 255, 255, 150))
                )
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius), fill=color
                )
            return result, affected, {"point_count": count}

        if name in {
            "row_noise",
            "column_noise",
            "banding",
            "streaks",
            "scanner_lines",
            "scratches",
            "fold_marks",
            "creases",
            "wrinkles",
        }:
            result = image.copy()
            draw = ImageDraw.Draw(result, "RGBA")
            count = max(1, int(2 + 12 * strength))
            horizontal = name not in {"column_noise", "scratches"}
            for _ in range(count):
                if horizontal:
                    y = rng.randrange(height)
                    draw.line(
                        (0, y, width, y + rng.randint(-1, 1)),
                        fill=(30, 30, 30, rng.randint(15, 70)),
                        width=rng.randint(1, 2),
                    )
                else:
                    x = rng.randrange(width)
                    draw.line(
                        (x, 0, x + rng.randint(-1, 1), height),
                        fill=(30, 30, 30, rng.randint(15, 70)),
                        width=rng.randint(1, 2),
                    )
            return result, affected, {"line_count": count}

        if name in {
            "jpeg_compression",
            "repeated_jpeg_compression",
            "webp_compression",
            "ringing_artifacts",
        }:
            quality = max(8, int(92 - 70 * strength))
            fmt = "WEBP" if name == "webp_compression" else "JPEG"
            result = image
            passes = 3 if name == "repeated_jpeg_compression" else 1
            for _ in range(passes):
                buffer = io.BytesIO()
                result.save(buffer, format=fmt, quality=quality)
                buffer.seek(0)
                result = Image.open(buffer).convert("RGB").copy()
            return (
                result,
                affected,
                {"format": fmt, "quality": quality, "passes": passes},
            )

        if name in {
            "contrast_loss",
            "low_contrast",
            "photocopy_degradation",
            "repeated_photocopy",
            "brightness_variation",
            "faded_text",
            "uneven_fading",
            "local_text_fading",
        }:
            contrast = max(0.15, 1 - 0.65 * strength)
            brightness = 1 + rng.uniform(-0.2, 0.25) * strength
            result = ImageEnhance.Contrast(image).enhance(contrast)
            result = ImageEnhance.Brightness(result).enhance(brightness)
            if name in {"repeated_photocopy"}:
                result = ImageEnhance.Contrast(result).enhance(1.35)
            return (
                self._regional(image, result, regions if "local" in name else []),
                affected,
                {"contrast": contrast, "brightness": brightness},
            )

        if name in {
            "uneven_illumination",
            "gradient_illumination",
            "vignetting",
            "glare",
            "local_overexposure",
            "local_underexposure",
            "shadow",
            "binding_shadow",
            "corner_shadow",
            "edge_shadow",
            "edge_darkening",
        }:
            mask = _gradient(
                image.size, horizontal=name in {"binding_shadow", "edge_shadow"}
            )
            if name in {"vignetting", "corner_shadow", "edge_darkening"}:
                mask = ImageOps.invert(mask)
            tint = Image.new(
                "RGB",
                image.size,
                "white" if name in {"glare", "local_overexposure"} else "black",
            )
            alpha = 0.08 + 0.35 * strength
            result = Image.composite(tint, image, mask.point(lambda p: int(p * alpha)))
            return result, affected, {"illumination_alpha": alpha}

        if name in {
            "yellow_paper",
            "browning",
            "gray_paper",
            "background_darkening",
            "background_color_variation",
        }:
            tint_color: tuple[int, int, int] = {
                "gray_paper": (190, 190, 190),
                "browning": (150, 95, 45),
            }.get(name, (230, 205, 130))
            alpha = 0.06 + 0.24 * strength
            return (
                Image.blend(image, Image.new("RGB", image.size, tint_color), alpha),
                affected,
                {"tint": tint_color, "alpha": alpha},
            )

        if name in {
            "ink_bleed",
            "ink_spread",
            "dot_gain",
            "character_dilation",
            "merged_strokes",
        }:
            return (
                image.filter(ImageFilter.MinFilter(3 if strength < 0.8 else 5)),
                affected,
                {"morphology": "dilate_dark"},
            )

        if name in {
            "ink_erosion",
            "broken_strokes",
            "broken_characters",
            "character_erosion",
            "disconnected_strokes",
            "weak_dots",
            "missing_dots",
            "faint_diacritics",
            "diacritic_erosion",
            "weak_punctuation",
        }:
            eroded = image.filter(ImageFilter.MaxFilter(3))
            return (
                Image.blend(image, eroded, 0.3 + 0.55 * strength),
                affected,
                {"morphology": "erode_dark"},
            )

        if name in {"thresholding_artifacts", "halftone_pattern"}:
            threshold = int(128 + rng.uniform(-35, 35) * strength)
            return (
                image.convert("L")
                .point(lambda p: 255 if p > threshold else 0)
                .convert("RGB"),
                affected,
                {"threshold": threshold},
            )

        if name in {
            "border_artifacts",
            "page_border",
            "black_edge",
            "white_edge",
            "punched_holes",
            "torn_edges",
            "synthetic_stamp",
            "synthetic_handwritten_marks",
            "synthetic_page_number",
            "synthetic_marginal_noise",
            "double_page_scan",
            "baseline_jitter",
            "water_marks",
            "bleed_through",
            "show_through",
        }:
            result = image.copy()
            draw = ImageDraw.Draw(result, "RGBA")
            edge = max(1, int(min(width, height) * 0.015 * strength))
            if name == "white_edge":
                draw.rectangle(
                    (0, 0, width - 1, height - 1), outline="white", width=edge
                )
            elif name == "punched_holes":
                for i in range(3):
                    y = int(height * (i + 1) / 4)
                    draw.ellipse(
                        (0, y - edge * 2, edge * 4, y + edge * 2),
                        fill=(20, 20, 20, 220),
                    )
            elif name == "synthetic_page_number":
                draw.text(
                    (width // 2, height - max(20, edge * 3)),
                    str(rng.randint(1, 999)),
                    fill=(40, 40, 40, 180),
                    anchor="mm",
                )
            elif name in {"water_marks", "synthetic_stamp"}:
                draw.ellipse(
                    (width * 0.3, height * 0.4, width * 0.7, height * 0.6),
                    outline=(140, 50, 50, 80),
                    width=max(1, edge),
                )
            elif name in {"bleed_through", "show_through"}:
                ghost = ImageOps.mirror(image).filter(ImageFilter.GaussianBlur(1.2))
                result = Image.blend(result, ghost, 0.03 + 0.12 * strength)
            else:
                draw.rectangle(
                    (0, 0, width - 1, height - 1), outline=(20, 20, 20, 180), width=edge
                )
            return result, affected, {"synthetic_overlay": name, "edge": edge}

        raise KeyError(f"Operator has no implementation: {name}")

    @staticmethod
    def _regional(
        original: Image.Image,
        transformed: Image.Image,
        boxes: list[tuple[int, int, int, int]],
    ) -> Image.Image:
        if not boxes:
            return transformed
        result = original.copy()
        for box in boxes:
            result.paste(transformed.crop(box), box)
        return result
