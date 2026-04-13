import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path


class TrackMarker:
    def __init__(
        self,
        min_blob_area=50,
        area_tolerance=None,
        split_x_distance_px=20,
        edge_margin_px=20,
        vertical_align_x_tolerance_px=35,
        singleton_line_len_px=30,
        group_gap_tol_ratio=0.25,
        group_width_tol_ratio=0.25,
        min_consistent_group_size=2,
        debug=False,
    ):
        """Configure thresholds used for blob-center extraction.

        Args:
            min_blob_area: Minimum connected-component area in pixels to keep as a blob.
            area_tolerance: Alias for min_blob_area; blobs below this area are ignored.
            split_x_distance_px: Split a connected blob into multiple blobs only when
                sub-cluster centers differ by more than this value on the x-axis.
            edge_margin_px: Ignore blobs whose bounding box is too close to image edges.
            vertical_align_x_tolerance_px: Maximum x-gap allowed to treat two centers
                as vertically aligned.
            singleton_line_len_px: Length of the short vertical line for unpaired centers.
            debug: Enables verbose diagnostic printing when True.
        """
        if area_tolerance is None:
            area_tolerance = min_blob_area
        self.min_blob_area = int(area_tolerance)
        self.split_x_distance_px = float(split_x_distance_px)
        self.edge_margin_px = int(edge_margin_px)
        self.vertical_align_x_tolerance_px = float(vertical_align_x_tolerance_px)
        self.singleton_line_len_px = float(singleton_line_len_px)
        self.group_gap_tol_ratio = float(group_gap_tol_ratio)
        self.group_width_tol_ratio = float(group_width_tol_ratio)
        self.min_consistent_group_size = max(2, int(min_consistent_group_size))
        self.debug = debug

    def _build_track_widths(self, vertical_lines, blobs):
        """Compute one representative width per vertical track from member blobs."""
        blob_widths = [float(blob["bbox"][2]) for blob in blobs]
        track_widths = []
        for line in vertical_lines:
            idxs = line.get("indices", [])
            widths = [blob_widths[i] for i in idxs if 0 <= i < len(blob_widths)]
            if widths:
                track_widths.append(float(np.median(widths)))
            else:
                track_widths.append(1.0)
        return track_widths

    def _find_consistent_adjacent_group(self, x_sorted_lines, x_sorted_widths):
        """Return indices of the most similar adjacent group (by gap and width)."""
        n = len(x_sorted_lines)
        if n == 0:
            return []
        if n <= self.min_consistent_group_size:
            return list(range(n))

        best_group = [0]
        best_score = (1, float("inf"), float("inf"))

        for start in range(n):
            group = [start]
            gaps = []
            widths = [float(x_sorted_widths[start])]

            for j in range(start + 1, n):
                prev = j - 1
                gap = float(x_sorted_lines[j]["x"] - x_sorted_lines[prev]["x"])
                if gap <= 0.0:
                    break

                candidate_width = float(x_sorted_widths[j])

                if gaps:
                    ref_gap = float(np.median(gaps))
                    allowed_gap_dev = max(2.0, self.group_gap_tol_ratio * ref_gap)
                    if abs(gap - ref_gap) > allowed_gap_dev:
                        break

                ref_width = float(np.median(widths))
                allowed_width_dev = max(2.0, self.group_width_tol_ratio * ref_width)
                if abs(candidate_width - ref_width) > allowed_width_dev:
                    break

                group.append(j)
                gaps.append(gap)
                widths.append(candidate_width)

            group_len = len(group)
            if group_len < self.min_consistent_group_size:
                continue

            gap_std = float(np.std(gaps)) if gaps else 0.0
            width_std = float(np.std(widths)) if widths else 0.0
            score = (-group_len, gap_std, width_std)
            if score < best_score:
                best_score = score
                best_group = group

        if len(best_group) < self.min_consistent_group_size:
            return list(range(n))
        return best_group

    def _regularize_lines_from_group(self, x_sorted_lines, group_indices, ideal_spacing):
        """Project all line x positions from a consistent group spacing."""
        if not x_sorted_lines:
            return []
        if ideal_spacing is None or ideal_spacing <= 0:
            return [dict(line) for line in x_sorted_lines]

        anchor_pos = int(group_indices[len(group_indices) // 2]) if group_indices else (len(x_sorted_lines) // 2)
        anchor_x = float(x_sorted_lines[anchor_pos]["x"])

        regularized = []
        for i, line in enumerate(x_sorted_lines):
            new_line = dict(line)
            new_line["x"] = float(anchor_x + (i - anchor_pos) * ideal_spacing)
            regularized.append(new_line)

        return regularized

    def _build_tiled_rectangle_centers(self, regularized_lines, blobs, rect_width, spacing):
        """Extend equal-width/equal-spacing rectangle centers to cover blob extremes."""
        if not blobs:
            return []

        base_centers = sorted(float(line["x"]) for line in regularized_lines)
        if not base_centers:
            base_centers = sorted(float(blob["center"][0]) for blob in blobs)
            if not base_centers:
                return []
            base_centers = [base_centers[len(base_centers) // 2]]

        rect_width = max(1.0, float(rect_width))
        if spacing is None or spacing <= 0:
            if len(base_centers) > 1:
                diffs = [base_centers[i + 1] - base_centers[i] for i in range(len(base_centers) - 1)]
                diffs = [d for d in diffs if d > 0]
                spacing = float(np.median(diffs)) if diffs else rect_width
            else:
                spacing = rect_width
        spacing = max(1.0, float(spacing))

        target_left = min(float(blob["center"][0]) for blob in blobs)
        target_right = max(float(blob["center"][0]) for blob in blobs)
        half_w = rect_width / 2.0

        centers = list(base_centers)
        while target_left < (centers[0] - half_w):
            centers.insert(0, centers[0] - spacing)
        while target_right > (centers[-1] + half_w):
            centers.append(centers[-1] + spacing)

        return centers

    def _fit_width_without_intersection(self, centers_x, rect_width):
        """Shrink shared width so neighboring rectangles do not overlap."""
        rect_width = float(rect_width)
        if len(centers_x) < 2:
            return max(0.1, rect_width)

        sorted_x = sorted(float(x) for x in centers_x)
        gaps = [sorted_x[i + 1] - sorted_x[i] for i in range(len(sorted_x) - 1)]
        gaps = [g for g in gaps if g > 0.0]
        if not gaps:
            return 0.1

        max_non_overlap_width = max(0.1, min(gaps) - 1e-3)
        return max(0.1, min(rect_width, max_non_overlap_width))

    def _build_vertical_tracks(self, centers):
        """Group centers into vertical tracks by x alignment and return one line per track."""
        n = len(centers)
        if n == 0:
            return []

        centers_arr = np.asarray(centers, dtype=np.float32)
        x_order = np.argsort(centers_arr[:, 0])

        tracks = []
        current = [int(x_order[0])]
        current_x_mean = float(centers_arr[current[0], 0])

        for idx in x_order[1:]:
            idx = int(idx)
            x_val = float(centers_arr[idx, 0])
            if abs(x_val - current_x_mean) <= self.vertical_align_x_tolerance_px:
                current.append(idx)
                current_x_mean = float(np.mean(centers_arr[current, 0]))
            else:
                tracks.append(current)
                current = [idx]
                current_x_mean = x_val


        tracks.append(current)

        line_specs = []
        for track in tracks:
            points = centers_arr[track]
            x_line = float(np.median(points[:, 0]))
            y_min = float(np.min(points[:, 1]))
            y_max = float(np.max(points[:, 1]))
            line_specs.append(
                {
                    "indices": track,
                    "x": x_line,
                    "y_min": y_min,
                    "y_max": y_max,
                }
            )

        return line_specs

    def _split_component_points(self, points_xy):
        """Recursively split one connected component using k-means on x coordinates."""
        if not self._is_joined_component(points_xy):
            return [points_xy]

        pending = [points_xy]
        final_clusters = []

        while pending:
            pts = pending.pop()

            if not self._is_joined_component(pts):
                final_clusters.append(pts)
                continue

            if pts.shape[0] < 2 * self.min_blob_area:
                final_clusters.append(pts)
                continue

            x_values = pts[:, 0].astype(np.float32).reshape(-1, 1)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.2)
            _, labels, centers = cv2.kmeans(
                x_values,
                2,
                None,
                criteria,
                5,
                cv2.KMEANS_PP_CENTERS,
            )

            labels = labels.ravel()
            cluster_a = pts[labels == 0]
            cluster_b = pts[labels == 1]

            if cluster_a.size == 0 or cluster_b.size == 0:
                final_clusters.append(pts)
                continue

            center_dx = abs(float(centers[0, 0]) - float(centers[1, 0]))

            if (
                center_dx > self.split_x_distance_px
                and cluster_a.shape[0] >= self.min_blob_area
                and cluster_b.shape[0] >= self.min_blob_area
            ):
                pending.append(cluster_a)
                pending.append(cluster_b)
            else:
                final_clusters.append(pts)

        return final_clusters

    def _is_joined_component(self, points_xy):
        """Return True only when a component looks like multiple blobs joined along x."""
        if points_xy.shape[0] < 2 * self.min_blob_area:
            return False

        x_vals = points_xy[:, 0].astype(np.int32)
        x_min = int(x_vals.min())
        x_max = int(x_vals.max())

        if x_max <= x_min:
            return False

        x_hist = np.bincount(x_vals - x_min, minlength=(x_max - x_min + 1)).astype(np.float32)
        if x_hist.size < 3:
            return False

        kernel = np.ones(5, dtype=np.float32) / 5.0
        smooth = np.convolve(x_hist, kernel, mode="same")

        peak_floor = 0.30 * float(np.max(smooth))
        if peak_floor <= 0:
            return False

        peaks = []
        for i in range(1, smooth.size - 1):
            if smooth[i] >= smooth[i - 1] and smooth[i] > smooth[i + 1] and smooth[i] >= peak_floor:
                peaks.append(i)

        if len(peaks) < 2:
            return False

        peaks.sort()
        for left, right in zip(peaks[:-1], peaks[1:]):
            if (right - left) > self.split_x_distance_px:
                return True

        return False

    def _extract_blob_centers(self, mask):
        nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        image_h, image_w = mask.shape[:2]

        blobs = []

        for i in range(1, nlabels):
            _, _, _, _, area = stats[i]

            if area < self.min_blob_area:
                continue

            component_yx = np.column_stack(np.where(labels == i))
            if component_yx.size == 0:
                continue

            component_xy = np.column_stack((component_yx[:, 1], component_yx[:, 0])).astype(np.float32)
            sub_components = self._split_component_points(component_xy)

            for sub_idx, sub_component in enumerate(sub_components):
                sub_area = int(sub_component.shape[0])
                if sub_area < self.min_blob_area:
                    continue

                x_min = int(np.floor(sub_component[:, 0].min()))
                x_max = int(np.ceil(sub_component[:, 0].max()))
                y_min = int(np.floor(sub_component[:, 1].min()))
                y_max = int(np.ceil(sub_component[:, 1].max()))

                if (
                    x_min <= self.edge_margin_px
                    or y_min <= self.edge_margin_px
                    or x_max >= (image_w - 1 - self.edge_margin_px)
                    or y_max >= (image_h - 1 - self.edge_margin_px)
                ):
                    continue

                cx = float(np.mean(sub_component[:, 0]))
                cy = float(np.mean(sub_component[:, 1]))

                blobs.append(
                    {
                        "id": f"{i}_{sub_idx}",
                        "bbox": (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1),
                        "area": sub_area,
                        "center": (cx, cy),
                    }
                )

        return blobs

    def analyze_mask(self, mask):
        mask = np.asarray(mask)
        if mask.ndim > 2:
            mask = mask.squeeze()
        mask = (mask > 0).astype(np.uint8)

        blobs = self._extract_blob_centers(mask)
        centers = [blob["center"] for blob in blobs]
        vertical_lines = self._build_vertical_tracks(centers)

        lines_sorted = sorted(vertical_lines, key=lambda line: float(line["x"]))
        track_widths = self._build_track_widths(lines_sorted, blobs)
        group_indices = self._find_consistent_adjacent_group(lines_sorted, track_widths)

        group_widths = [track_widths[i] for i in group_indices] if group_indices else []
        consistent_width = float(np.median(group_widths)) if group_widths else 1.0

        group_x = [float(lines_sorted[i]["x"]) for i in group_indices] if group_indices else []
        group_gaps = [group_x[i + 1] - group_x[i] for i in range(len(group_x) - 1)]
        group_gaps = [gap for gap in group_gaps if gap > 0.0]
        consistent_spacing = float(np.median(group_gaps)) if group_gaps else None

        regularized_lines = self._regularize_lines_from_group(
            lines_sorted,
            group_indices,
            consistent_spacing,
        )

        if blobs:
            global_median_width = float(np.median([blob["bbox"][2] for blob in blobs]))
        else:
            global_median_width = 1.0

        if consistent_spacing is not None:
            non_intersect_width = max(1.0, float(consistent_spacing) - 1.0)
            rectangle_width = min(consistent_width, non_intersect_width)
        else:
            rectangle_width = max(1.0, consistent_width)

        rectangle_centers_x = self._build_tiled_rectangle_centers(
            regularized_lines=regularized_lines,
            blobs=blobs,
            rect_width=rectangle_width,
            spacing=consistent_spacing,
        )
        rectangle_width = self._fit_width_without_intersection(
            centers_x=rectangle_centers_x,
            rect_width=rectangle_width,
        )

        if blobs:
            global_top_y = float(min(blob["bbox"][1] for blob in blobs))
            global_bottom_y = float(max(blob["bbox"][1] + blob["bbox"][3] - 1 for blob in blobs))
        else:
            global_top_y = 0.0
            global_bottom_y = 0.0

        result = {
            "mask": mask,
            "blobs": blobs,
            "centers": centers,
            "vertical_lines": vertical_lines,
            "regularized_vertical_lines": regularized_lines,
            "consistent_group_indices": group_indices,
            "consistent_group_width": consistent_width,
            "consistent_group_spacing": consistent_spacing,
            "rectangle_centers_x": rectangle_centers_x,
            "global_top_y": global_top_y,
            "global_bottom_y": global_bottom_y,
            "global_median_width": global_median_width,
            "rectangle_width": rectangle_width,
        }

        if self.debug:
            print("\n[INFO]")
            print("Area tolerance:", self.min_blob_area)
            print("Detected blobs:", len(blobs))
            print("Blob centers (x, y):", centers)
            print("Vertical tracks:", [line["indices"] for line in vertical_lines])
            print("Consistent group (sorted-line indices):", group_indices)
            print("Consistent group width:", consistent_width)
            print("Consistent group spacing:", consistent_spacing)
            print("Rectangle centers used:", rectangle_centers_x)
            print("Global top/bottom y:", (global_top_y, global_bottom_y))
            print("Global median width:", global_median_width)
            print("Final no-overlap width:", rectangle_width)

        return result

    def get_band_summary(self, analysis_result):
        """Return extracted band geometry summary from an analysis result.

        Returns:
            dict with:
                - band_count: number of plotted bands/rectangles
                - first_band_x: x-position of first (left-most) band center
                - rectangle_height: global rectangle height
                - rectangle_width: shared rectangle width
                - migration_front: highest point of rectangle from image bottom
                - edge_cut: lowest point of rectangle from image bottom
                - distances_between_rectangles: neighbor center-to-center distances
                - mean_distance_between_rectangles: average neighbor distance
        """
        centers_x = sorted(float(x) for x in analysis_result.get("rectangle_centers_x", []))
        band_count = len(centers_x)

        first_band_x = centers_x[0] if centers_x else None

        global_top_y = float(analysis_result.get("global_top_y", 0.0))
        global_bottom_y = float(analysis_result.get("global_bottom_y", 0.0))
        rectangle_height = max(1.0, global_bottom_y - global_top_y)

        mask = np.asarray(analysis_result.get("mask", np.zeros((1, 1), dtype=np.uint8)))
        mask_h = int(mask.shape[0]) if mask.ndim >= 2 else 1
        image_bottom_y = float(max(0, mask_h - 1))

        rectangle_width = float(analysis_result.get("rectangle_width", 1.0))
        edge_cut = max(0.0, image_bottom_y - global_bottom_y)
        lowest_blob_bottom_y = float(analysis_result.get("global_bottom_y", 0.0))
        migration_front = max(0.0, image_bottom_y - global_top_y)

        distances = []
        for i in range(band_count - 1):
            distances.append(float(centers_x[i + 1] - centers_x[i]))

        mean_distance = float(np.mean(distances)) if distances else None

        return {
            "num_bands": band_count,
            "first_app_position": first_band_x,
            "edge_cut": edge_cut,
            "migration_front": migration_front,
            "band_width": rectangle_width,
        #    "distances_between_tracks": mean_distance-rectangle_width,
        }

    def plot(self, analysis_result, image=None):
        mask = analysis_result["mask"]
        rectangle_centers_x = analysis_result.get("rectangle_centers_x", [])
        global_top_y = float(analysis_result.get("global_top_y", 0.0))
        global_bottom_y = float(analysis_result.get("global_bottom_y", 0.0))
        rectangle_width = float(analysis_result.get("rectangle_width", 1.0))

        mask_h, mask_w = mask.shape[:2]

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        if image is not None:
            if hasattr(image, "detach"):
                canvas = image.detach().cpu().numpy()
            else:
                canvas = np.asarray(image)

            if canvas.ndim == 3 and canvas.shape[0] in (1, 3, 4) and canvas.shape[-1] not in (1, 3, 4):
                canvas = np.transpose(canvas, (1, 2, 0))

            if canvas.ndim == 3 and canvas.shape[-1] == 1:
                canvas = canvas[..., 0]

            ax.imshow(canvas)
            canvas_h = canvas.shape[0]
            canvas_w = canvas.shape[1]
            title = "Image + Markings"
        else:
            ax.imshow(mask, cmap="gray")
            canvas_h = mask.shape[0]
            canvas_w = mask.shape[1]
            title = "Mask + Markings"

        if mask_w > 0 and mask_h > 0:
            scale_x = float(canvas_w) / float(mask_w)
            scale_y = float(canvas_h) / float(mask_h)
        else:
            scale_x = 1.0
            scale_y = 1.0

        half_len = self.singleton_line_len_px / 2.0
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

        ax.set_title(title)
        ax.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    from ultralytics import FastSAM
    from segment import Mask_builder

    paths = [
        "images/AH/4_19_2_5_2_53_3_1_1_148.tif", 
        "images/AH/3_33_2_5_2_40_4_1_1_149.tif" ,
        "images/AH/3_33_2_5_1_41_1_1_1_152.tif",
        "images/AH/3_33_2_5_2_1_1_1_1_164.tif"
    ]

    #model = FastSAM("FastSAM-s.pt")
    import time
    start = time.time()
    #model = FastSAM("FastSAM-s.pt")
    model =  FastSAM("FastSAM-x.pt")
    masks = Mask_builder(paths, model).get_mask()

    marker = TrackMarker(
        area_tolerance=150,
        split_x_distance_px=30,   # impt
        edge_margin_px=20,
        vertical_align_x_tolerance_px=20,
        singleton_line_len_px=30,
        debug=False,
    )
    end = time.time()
    print("time:", end-start)
    from utils import load_image
    image_tensors =  load_image(paths, plot=False, return_gray_scale=False, image_size=(1024, 1024))
    for idx,(path,mask) in enumerate(zip(paths,masks )):
        print(f"\n=== Image {idx} ===")
        print("path:",path)
        result = marker.analyze_mask(mask)
        band_results = marker.get_band_summary(result)
        for key, value in band_results.items():
            print(f"{key}: {value}")
        marker.plot(result, image=image_tensors[idx])

        







