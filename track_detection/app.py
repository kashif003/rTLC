from pathlib import Path
import tempfile
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # no GUI backend needed
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np
from PIL import Image as PILImage

from shiny import App, reactive, render, ui

from ultralytics import FastSAM
from segment import Mask_builder
from mark import TrackMarker


app_ui = ui.page_fluid(
    ui.h2("Track detection viewer"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_file(
                "image_file",
                "Upload an image",
                accept=[".png", ".jpg", ".jpeg", ".tif", ".tiff"],
                multiple=True,
            ),
            ui.input_select("selected_image", "Preview image", choices={}),
            ui.hr(),
            ui.input_checkbox("manual_mode", "Manual override markings", value=False),
            ui.input_numeric("num_bands", "num_bands", value=0, min=0, step=1),
            ui.input_numeric("first_app_position", "first_app_position", value=0.0, step=1.0),
            ui.input_numeric("edge_cut", "edge_cut", value=0.0, step=1.0),
            ui.input_numeric("migration_front", "migration_front", value=0.0, step=1.0),
            ui.input_numeric("band_spacing", "band_spacing (px)", value=30.0, min=1.0, step=1.0),
            ui.input_numeric("band_width", "band_width (px)", value=10.0, min=1.0, step=1.0),
            ui.p("Tip: Upload first, then tweak values in manual mode."),
            ui.hr(),
            ui.input_action_button("extract_btn", "Extract Bands", class_="btn-primary"),
            ui.output_text_verbatim("extract_status"),
        ),
        ui.output_plot("marked_plot"),
        ui.output_text_verbatim("summary"),
    ),
)


model = FastSAM("FastSAM-s.pt")
marker = TrackMarker(
    area_tolerance=150,
    split_x_distance_px=30,
    edge_margin_px=20,
    vertical_align_x_tolerance_px=20,
    singleton_line_len_px=30,
    debug=False,
)


def draw_markings_on_image(image_path: str, analysis_result: dict):
    image = PILImage.open(image_path).convert("RGB")
    canvas = np.asarray(image)

    mask = analysis_result["mask"]
    rectangle_centers_x = analysis_result.get("rectangle_centers_x", [])
    global_top_y = float(analysis_result.get("global_top_y", 0.0))
    global_bottom_y = float(analysis_result.get("global_bottom_y", 0.0))
    rectangle_width = float(analysis_result.get("rectangle_width", 1.0))

    mask_h, mask_w = mask.shape[:2]
    canvas_h, canvas_w = canvas.shape[:2]

    scale_x = float(canvas_w) / float(mask_w) if mask_w > 0 else 1.0
    scale_y = float(canvas_h) / float(mask_h) if mask_h > 0 else 1.0

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(canvas)

    half_len = marker.singleton_line_len_px / 2.0
    y0 = global_top_y * scale_y
    y1 = global_bottom_y * scale_y
    if abs(y1 - y0) < 1.0:
        y0 = max(0.0, y0 - half_len)
        y1 = min(float(canvas_h - 1), y1 + half_len)

    for x in rectangle_centers_x:
        x = float(x) * scale_x
        rect_width = max(1.0, rectangle_width * scale_x)
        rect = Rectangle(
            (x - rect_width / 2.0, y0),
            rect_width,
            max(1.0, y1 - y0),
            linewidth=1.5,
            edgecolor="cyan",
            facecolor="none",
            alpha=0.9,
        )
        ax.add_patch(rect)

    ax.set_title("Image + Markings")
    ax.axis("off")
    plt.tight_layout()
    return fig


def build_auto_result(image_path: str):
    mask = Mask_builder([image_path], model).get_mask()[0]
    return marker.analyze_mask(mask)


def build_auto_results(image_paths):
    if not image_paths:
        return []

    masks = Mask_builder(image_paths, model).get_mask()
    return [marker.analyze_mask(mask) for mask in masks]


def get_default_spacing(result: dict):
    spacing = result.get("consistent_group_spacing", None)
    if spacing is not None and spacing > 0:
        return float(spacing)

    xs = sorted(float(x) for x in result.get("rectangle_centers_x", []))
    if len(xs) >= 2:
        diffs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        diffs = [d for d in diffs if d > 0]
        if diffs:
            return float(np.median(diffs))

    return 30.0


def get_default_image_settings(result: dict):
    bands = marker.get_band_summary(result)
    return {
        "manual_mode": False,
        "num_bands": int(bands.get("num_bands", 0)),
        "first_app_position": float(bands.get("first_app_position") or 0.0),
        "edge_cut": float(bands.get("edge_cut") or 0.0),
        "migration_front": float(bands.get("migration_front") or 0.0),
        "band_spacing": float(get_default_spacing(result)),
        "band_width": _safe_float(result.get("rectangle_width", 1.0), 1.0),
    }


def get_image_settings(image_state_value, image_name: str, result: dict):
    settings = dict(get_default_image_settings(result))
    saved = image_state_value.get(image_name)
    if saved is not None:
        settings.update(saved)
    return settings


def _safe_float(value, default=0.0):
    return default if value is None else float(value)


def _safe_int(value, default=0):
    return default if value is None else int(value)


def apply_manual_override(result: dict, num_bands: int, first_x: float, edge_cut: float, migration_front: float, spacing: float, band_width: float):
    updated = dict(result)

    mask = result["mask"]
    h = int(mask.shape[0])
    image_bottom_y = float(max(0, h - 1))

    nb = max(0, _safe_int(num_bands, 0))
    first_x = _safe_float(first_x, 0.0)
    spacing = max(1.0, _safe_float(spacing, 30.0))
    band_width = max(1.0, _safe_float(band_width, _safe_float(result.get("rectangle_width", 1.0), 1.0)))

    centers = [float(first_x) + i * spacing for i in range(nb)]

    top_y = image_bottom_y - _safe_float(migration_front, 0.0)
    bottom_y = image_bottom_y - _safe_float(edge_cut, 0.0)

    top_y = float(np.clip(top_y, 0, image_bottom_y))
    bottom_y = float(np.clip(bottom_y, 0, image_bottom_y))
    if bottom_y < top_y:
        top_y, bottom_y = bottom_y, top_y

    updated["rectangle_centers_x"] = centers
    updated["global_top_y"] = top_y
    updated["global_bottom_y"] = bottom_y
    updated["rectangle_width"] = band_width

    return updated


def make_unique_output_folder(base_output_dir: str, image_name: str):
    base_path = Path(base_output_dir)
    candidate = base_path / image_name
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    suffix = 1
    while True:
        candidate = base_path / f"{image_name}_{suffix}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        suffix += 1


def extract_and_save_bands(image_path: str, analysis_result: dict, original_filename: str, base_output_dir: str = "extracted_bands"):
    """Extract each band from the image and save to a folder."""
    try:
        image = PILImage.open(image_path).convert("RGB")
        image_array = np.asarray(image)
        canvas_h, canvas_w = image_array.shape[:2]

        mask = analysis_result["mask"]
        rectangle_centers_x = analysis_result.get("rectangle_centers_x", [])
        global_top_y = float(analysis_result.get("global_top_y", 0.0))
        global_bottom_y = float(analysis_result.get("global_bottom_y", 0.0))
        rectangle_width = float(analysis_result.get("rectangle_width", 1.0))

        mask_h, mask_w = mask.shape[:2]

        scale_x = float(canvas_w) / float(mask_w) if mask_w > 0 else 1.0
        scale_y = float(canvas_h) / float(mask_h) if mask_h > 0 else 1.0

        y0 = int(global_top_y * scale_y)
        y1 = int(global_bottom_y * scale_y)
        if abs(y1 - y0) < 1:
            half_len = 15
            y0 = max(0, y0 - half_len)
            y1 = min(canvas_h - 1, y1 + half_len)

        y0 = max(0, y0)
        y1 = min(canvas_h, y1)

        image_name = Path(original_filename).stem
        output_folder = make_unique_output_folder(base_output_dir, image_name)

        saved_paths = []
        for idx, x_center in enumerate(rectangle_centers_x):
            x_center = float(x_center) * scale_x
            rect_width = max(1.0, rectangle_width * scale_x)

            x0 = int(x_center - rect_width / 2.0)
            x1 = int(x_center + rect_width / 2.0)

            x0 = max(0, x0)
            x1 = min(canvas_w, x1)

            if x1 <= x0 or y1 <= y0:
                continue

            cropped = image_array[y0:y1, x0:x1]
            cropped_img = PILImage.fromarray(cropped.astype(np.uint8))

            band_filename = f"{idx + 1:03d}.png"
            band_path = output_folder / band_filename
            cropped_img.save(str(band_path))
            saved_paths.append(str(band_path))

        return {
            "success": True,
            "output_dir": str(output_folder),
            "num_bands_extracted": len(saved_paths),
            "band_paths": saved_paths,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def server(input, output, session):
    image_state = reactive.Value({})
    loading_state = reactive.Value(False)
    output_base_dir = str(Path(__file__).resolve().parent.parent / "extracted_bands")

    @reactive.calc
    def uploaded_files():
        uploaded = input.image_file()
        if not uploaded:
            return []
        return uploaded

    @reactive.calc
    def selected_uploaded_file():
        files = uploaded_files()
        if not files:
            return None
        selected_name = input.selected_image()
        for file_info in files:
            if file_info.get("name") == selected_name:
                return file_info
        return files[0]

    @reactive.calc
    def uploaded_path():
        selected_file = selected_uploaded_file()
        if selected_file is None:
            return None
        return selected_file["datapath"]

    @reactive.calc
    def selected_image_name():
        selected_file = selected_uploaded_file()
        if selected_file is None:
            return None
        return selected_file.get("name")

    @reactive.calc
    def uploaded_analysis_results():
        files = uploaded_files()
        if not files:
            return {}

        image_paths = [file_info["datapath"] for file_info in files]
        batch_results = build_auto_results(image_paths)

        analysis_results = {}
        for idx, file_info in enumerate(files):
            image_name = file_info.get("name", "image")
            if idx < len(batch_results):
                analysis_results[image_name] = batch_results[idx]

        return analysis_results

    @reactive.calc
    def selected_analysis_result():
        image_name = selected_image_name()
        if image_name is None:
            return None
        return uploaded_analysis_results().get(image_name)

    @reactive.effect
    @reactive.event(uploaded_files)
    def _sync_image_choices():
        files = uploaded_files()
        choices = {file_info.get("name", f"image_{idx + 1}"): file_info.get("name", f"image_{idx + 1}") for idx, file_info in enumerate(files)}

        if not choices:
            ui.update_select("selected_image", choices={}, selected=None)
            return

        selected_name = input.selected_image()
        if selected_name not in choices:
            selected_name = next(iter(choices.keys()))

        ui.update_select("selected_image", choices=choices, selected=selected_name)

    @reactive.effect
    @reactive.event(selected_image_name, uploaded_analysis_results)
    def _load_saved_or_auto_values():
        image_name = selected_image_name()
        result = selected_analysis_result()
        if image_name is None or result is None:
            return

        saved = image_state().get(image_name)

        loading_state.set(True)
        try:
            if saved is not None:
                ui.update_checkbox("manual_mode", value=bool(saved.get("manual_mode", False)))
                ui.update_numeric("num_bands", value=_safe_int(saved.get("num_bands", 0), 0))
                ui.update_numeric("first_app_position", value=_safe_float(saved.get("first_app_position", 0.0), 0.0))
                ui.update_numeric("edge_cut", value=_safe_float(saved.get("edge_cut", 0.0), 0.0))
                ui.update_numeric("migration_front", value=_safe_float(saved.get("migration_front", 0.0), 0.0))
                ui.update_numeric("band_spacing", value=_safe_float(saved.get("band_spacing", get_default_spacing(result)), get_default_spacing(result)))
                ui.update_numeric("band_width", value=_safe_float(saved.get("band_width", result.get("rectangle_width", 1.0)), _safe_float(result.get("rectangle_width", 1.0), 1.0)))
            else:
                bands = marker.get_band_summary(result)
                ui.update_checkbox("manual_mode", value=False)
                ui.update_numeric("num_bands", value=int(bands.get("num_bands", 0)))
                ui.update_numeric("first_app_position", value=float(bands.get("first_app_position") or 0.0))
                ui.update_numeric("edge_cut", value=float(bands.get("edge_cut") or 0.0))
                ui.update_numeric("migration_front", value=float(bands.get("migration_front") or 0.0))
                ui.update_numeric("band_spacing", value=float(get_default_spacing(result)))
                ui.update_numeric("band_width", value=_safe_float(result.get("rectangle_width", 1.0), 1.0))
        finally:
            loading_state.set(False)

    @reactive.effect
    @reactive.event(input.manual_mode, input.num_bands, input.first_app_position, input.edge_cut, input.migration_front, input.band_spacing, input.band_width)
    def _persist_edits_for_selected_image():
        if loading_state():
            return

        image_name = selected_image_name()
        if image_name is None:
            return

        state = dict(image_state())
        state[image_name] = {
            "manual_mode": bool(input.manual_mode() or False),
            "num_bands": _safe_int(input.num_bands(), 0),
            "first_app_position": _safe_float(input.first_app_position(), 0.0),
            "edge_cut": _safe_float(input.edge_cut(), 0.0),
            "migration_front": _safe_float(input.migration_front(), 0.0),
            "band_spacing": _safe_float(input.band_spacing(), 30.0),
            "band_width": _safe_float(input.band_width(), _safe_float(selected_analysis_result().get("rectangle_width", 1.0) if selected_analysis_result() else 1.0, 1.0)),
        }
        image_state.set(state)

    @reactive.calc
    def current_result():
        result = selected_analysis_result()
        if result is None:
            return None

        image_name = selected_image_name()
        if image_name is None:
            return result

        settings = get_image_settings(image_state(), image_name, result)

        if not settings.get("manual_mode", False):
            return result

        return apply_manual_override(
            result=result,
            num_bands=_safe_int(settings.get("num_bands", 0), 0),
            first_x=_safe_float(settings.get("first_app_position", 0.0), 0.0),
            edge_cut=_safe_float(settings.get("edge_cut", 0.0), 0.0),
            migration_front=_safe_float(settings.get("migration_front", 0.0), 0.0),
            spacing=_safe_float(settings.get("band_spacing", 30.0), 30.0),
            band_width=_safe_float(settings.get("band_width", result.get("rectangle_width", 1.0)), _safe_float(result.get("rectangle_width", 1.0), 1.0)),
        )

    @render.plot
    def marked_plot():
        path = uploaded_path()
        result = current_result()

        if path is None or result is None:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "Upload an image to see the markings", ha="center", va="center")
            ax.axis("off")
            return fig

        return draw_markings_on_image(path, result)

    @render.text
    def summary():
        result = current_result()
        if result is None:
            return "No image uploaded yet."

        bands = marker.get_band_summary(result)
        lines = [f"{k}: {v}" for k, v in bands.items()]
        mode = "manual" if input.manual_mode() else "auto"
        selected_name = input.selected_image() or ""
        return "selected image: " + selected_name + "\nmode: " + mode + "\n" + "\n".join(lines)

    @reactive.effect
    @reactive.event(input.extract_btn)
    def _handle_extract():
        files = uploaded_files()
        analysis_results = uploaded_analysis_results()
        if not files:
            return

        extracted = []
        errors = []

        for file_info in files:
            path = file_info["datapath"]
            original_name = file_info.get("name", "image")

            try:
                result = analysis_results.get(original_name)
                if result is None:
                    result = build_auto_result(path)
                settings = get_image_settings(image_state(), original_name, result)
                if settings.get("manual_mode", False):
                    result = apply_manual_override(
                        result=result,
                        num_bands=settings.get("num_bands", 0),
                        first_x=settings.get("first_app_position", 0.0),
                        edge_cut=settings.get("edge_cut", 0.0),
                        migration_front=settings.get("migration_front", 0.0),
                        spacing=settings.get("band_spacing", 30.0),
                        band_width=settings.get("band_width", result.get("rectangle_width", 1.0)),
                    )

                extraction_result = extract_and_save_bands(
                    path,
                    result,
                    original_name,
                    base_output_dir=output_base_dir,
                )

                if extraction_result.get("success"):
                    extracted.append(
                        f"{original_name}: {extraction_result['num_bands_extracted']} bands -> {extraction_result['output_dir']}"
                    )
                else:
                    errors.append(f"{original_name}: {extraction_result.get('error', 'Unknown error')}")
            except Exception as exc:
                errors.append(f"{original_name}: {exc}")

        extract_status_value.set(
            {
                "success": len(errors) == 0,
                "extracted": extracted,
                "errors": errors,
            }
        )

    extract_status_value = reactive.Value({})

    @render.text
    def extract_status():
        status = extract_status_value()
        if not status:
            return ""

        lines = []
        if status.get("extracted"):
            lines.append("Extracted:")
            lines.extend(status["extracted"])
        if status.get("errors"):
            lines.append("Errors:")
            lines.extend(status["errors"])
        return "\n".join(lines)


app = App(app_ui, server)