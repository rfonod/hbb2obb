import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import hbb2obb.converter as converter
from hbb2obb.converter import (
    aggregate_masks_by_majority_vote,
    clear_model_cache,
    create_obb_annotations_multi_model,
    load_sam_model,
    pack_results,
    resolve_confidences,
    save_confidence_annotations,
    save_obb_annotations,
    save_polygon_annotations,
    scale_bounding_boxes,
    unpack_results,
)
from hbb2obb.evaluator import parse_obb_file


class MockAnnotations:
    """Mock Annotations class for testing"""

    def __init__(self, hbb_xyxy, img_shape):
        self.hbb_xyxy = hbb_xyxy
        self.img_shape = img_shape


class TestConverter(unittest.TestCase):
    def setUp(self):
        # Create test image dimensions
        self.img_width = 640
        self.img_height = 480

        # Create sample HBB boxes
        self.hbb_boxes = np.array(
            [
                [0, 100, 100, 300, 200],  # [label, x1, y1, x2, y2]
                [1, 400, 300, 500, 400],
            ]
        )

        # Create sample image
        self.img = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

    def test_create_obb_annotations_single_mask_inside_hbb(self):
        """Test create_obb_annotations with a single mask inside the HBB"""
        # Create a mask inside the first HBB
        mask = np.zeros((self.img_height, self.img_width), dtype=bool)
        mask[120:180, 120:280] = True  # Mask smaller than the HBB

        all_models_masks = [[mask]]

        result, masks, contours, confidences = create_obb_annotations_multi_model(
            self.hbb_boxes[0:1], all_models_masks, opening_kernel_percentage=0.0
        )

        self.assertEqual(len(result), 1, "Should return 1 OBB annotation")
        self.assertEqual(result[0][0], 0, "Label should be preserved")
        self.assertEqual(len(result[0]), 9, "OBB should have label + 8 coordinates")

        # The result should be an OBB derived from the mask
        # Since our mask is rectangular and aligned with axes, the OBB should be similar to the mask bounds
        x_coords = result[0][1::2]  # x coordinates are at odd indices
        y_coords = result[0][2::2]  # y coordinates are at even indices

        self.assertTrue(np.min(x_coords) >= 120, "OBB should be inside or match mask boundary")
        self.assertTrue(np.max(x_coords) <= 280, "OBB should be inside or match mask boundary")
        self.assertTrue(np.min(y_coords) >= 120, "OBB should be inside or match mask boundary")
        self.assertTrue(np.max(y_coords) <= 180, "OBB should be inside or match mask boundary")

    def test_create_obb_annotations_mask_larger_than_hbb(self):
        """Test create_obb_annotations with a mask larger than the HBB"""
        # Create a mask larger than the first HBB
        mask = np.zeros((self.img_height, self.img_width), dtype=bool)
        mask[80:220, 80:320] = True  # Mask larger than the HBB

        all_models_masks = [[mask]]

        result, masks, contours, confidences = create_obb_annotations_multi_model(
            self.hbb_boxes[0:1], all_models_masks, opening_kernel_percentage=0.0
        )

        self.assertEqual(len(result), 1, "Should return 1 OBB annotation")

        # The result should be an OBB derived from the mask but constrained by the HBB
        # Verify the OBB is constrained to the HBB
        label, x1, y1, x2, y2, x3, y3, x4, y4 = result[0]
        self.assertEqual(label, 0, "Label should be preserved")

        # Get the bounding rectangle of the resulting OBB
        min_x = min(x1, x2, x3, x4)
        max_x = max(x1, x2, x3, x4)
        min_y = min(y1, y2, y3, y4)
        max_y = max(y1, y2, y3, y4)

        # The mask is cropped to the HBB, so the OBB should not exceed the HBB
        self.assertTrue(min_x >= self.hbb_boxes[0, 1], "OBB should not exceed HBB boundary")
        self.assertTrue(min_y >= self.hbb_boxes[0, 2], "OBB should not exceed HBB boundary")
        self.assertTrue(max_x <= self.hbb_boxes[0, 3], "OBB should not exceed HBB boundary")
        self.assertTrue(max_y <= self.hbb_boxes[0, 4], "OBB should not exceed HBB boundary")

    def test_create_obb_annotations_multiple_models(self):
        """Test create_obb_annotations with multiple models' masks"""
        # Create masks from different models
        mask1 = np.zeros((self.img_height, self.img_width), dtype=bool)
        mask1[120:180, 120:280] = True

        mask2 = np.zeros((self.img_height, self.img_width), dtype=bool)
        mask2[110:190, 110:290] = True

        all_models_masks = [[mask1], [mask2]]

        result, masks, contours, confidences = create_obb_annotations_multi_model(
            self.hbb_boxes[0:1], all_models_masks, opening_kernel_percentage=0.0
        )

        self.assertEqual(len(result), 1, "Should return 1 OBB annotation")
        self.assertEqual(result[0][0], 0, "Label should be preserved")

    def test_create_obb_annotations_no_valid_masks(self):
        """Test create_obb_annotations with no valid masks"""
        # Create an empty mask
        mask = np.zeros((self.img_height, self.img_width), dtype=bool)
        all_models_masks = [[mask]]

        result, masks, contours, confidences = create_obb_annotations_multi_model(
            self.hbb_boxes[0:1], all_models_masks, opening_kernel_percentage=0.0
        )

        self.assertEqual(len(result), 1, "Should return 1 OBB annotation")
        self.assertEqual(result[0][0], 0, "Label should be preserved")

        # With no valid mask, it should fall back to using the HBB
        expected_box_points = [
            100,
            100,  # x1, y1
            300,
            100,  # x2, y2
            300,
            200,  # x3, y3
            100,
            200,  # x4, y4
        ]

        for i, val in enumerate(expected_box_points):
            self.assertEqual(result[0][i + 1], val, f"OBB coordinate {i + 1} should match HBB")

    def test_create_obb_annotations_non_rectangular_mask(self):
        """Test create_obb_annotations with a non-rectangular mask"""
        # Create a triangular mask
        mask = np.zeros((self.img_height, self.img_width), dtype=bool)
        for i in range(100, 200):
            width = int((i - 100) * 2)
            start = 100 + (200 - width) // 2
            mask[i, start : start + width] = True

        all_models_masks = [[mask]]

        result, masks, contours, confidences = create_obb_annotations_multi_model(
            self.hbb_boxes[0:1], all_models_masks, opening_kernel_percentage=0.0
        )

        self.assertEqual(len(result), 1, "Should return 1 OBB annotation")
        self.assertEqual(result[0][0], 0, "Label should be preserved")

        # For a triangular mask, the OBB should be oriented

    def test_aggregate_masks_by_majority_vote_single_mask(self):
        """Test aggregating a single mask"""
        mask = np.zeros((100, 100), dtype=bool)
        mask[25:75, 25:75] = True

        result = aggregate_masks_by_majority_vote([mask])

        # Result should be identical to the input mask
        np.testing.assert_array_equal(result, mask)

    def test_aggregate_masks_by_majority_vote_identical_masks(self):
        """Test aggregating multiple identical masks"""
        mask = np.zeros((100, 100), dtype=bool)
        mask[25:75, 25:75] = True

        result = aggregate_masks_by_majority_vote([mask, mask, mask])

        # Result should be identical to any of the input masks
        np.testing.assert_array_equal(result, mask)

    def test_aggregate_masks_by_majority_vote_different_masks(self):
        """Test majority voting with different masks"""
        base_mask = np.zeros((100, 100), dtype=bool)

        mask1 = base_mask.copy()
        mask1[20:60, 20:60] = True

        mask2 = base_mask.copy()
        mask2[25:65, 25:65] = True

        mask3 = base_mask.copy()
        mask3[30:70, 30:70] = True

        result = aggregate_masks_by_majority_vote([mask1, mask2, mask3])

        # Check that pixels voted by majority (2/3) are included
        self.assertTrue(result[30, 30], "Pixel voted by majority should be True")
        self.assertTrue(result[55, 55], "Pixel voted by majority should be True")

        # Check that pixels voted by minority (1/3) are excluded
        self.assertFalse(result[20, 20], "Pixel voted by minority should be False")
        self.assertFalse(result[65, 65], "Pixel voted by minority should be False")

    def test_scale_boxes_single_factor(self):
        """Test scale boxes with a single factor"""
        factor = 0.1
        annotations = MockAnnotations(self.hbb_boxes, (self.img_width, self.img_height))

        result = scale_bounding_boxes(annotations, factor)

        # Check dimensions of result
        self.assertEqual(result.shape, self.hbb_boxes.shape)

        # First box: [0, 100, 100, 300, 200]
        orig_width = self.hbb_boxes[0, 3] - self.hbb_boxes[0, 1]  # 200
        orig_height = self.hbb_boxes[0, 4] - self.hbb_boxes[0, 2]  # 100

        expected_x1 = max(0, self.hbb_boxes[0, 1] - orig_width * factor)  # 100 - 200*0.1 = 80
        expected_y1 = max(0, self.hbb_boxes[0, 2] - orig_height * factor)  # 100 - 100*0.1 = 90
        expected_x2 = min(self.img_width - 1, self.hbb_boxes[0, 3] + orig_width * factor)  # 300 + 200*0.1 = 320
        expected_y2 = min(self.img_height - 1, self.hbb_boxes[0, 4] + orig_height * factor)  # 200 + 100*0.1 = 210

        self.assertAlmostEqual(result[0, 1], expected_x1, delta=1)
        self.assertAlmostEqual(result[0, 2], expected_y1, delta=1)
        self.assertAlmostEqual(result[0, 3], expected_x2, delta=1)
        self.assertAlmostEqual(result[0, 4], expected_y2, delta=1)

    def test_scale_boxes_dual_factors(self):
        """Test scale boxes with different factors for short and long sides"""
        factors = (0.05, 0.15)  # (short_factor, long_factor)
        annotations = MockAnnotations(self.hbb_boxes, (self.img_width, self.img_height))

        result = scale_bounding_boxes(annotations, factors)

        # For the first box (wider than tall), width is the long dimension
        orig_width = self.hbb_boxes[0, 3] - self.hbb_boxes[0, 1]  # 200
        orig_height = self.hbb_boxes[0, 4] - self.hbb_boxes[0, 2]  # 100

        # Apply the appropriate factors
        expected_x1 = max(0, self.hbb_boxes[0, 1] - orig_width * factors[1])  # long factor for width
        expected_y1 = max(0, self.hbb_boxes[0, 2] - orig_height * factors[0])  # short factor for height
        expected_x2 = min(self.img_width - 1, self.hbb_boxes[0, 3] + orig_width * factors[1])
        expected_y2 = min(self.img_height - 1, self.hbb_boxes[0, 4] + orig_height * factors[0])

        self.assertAlmostEqual(result[0, 1], expected_x1, delta=1)
        self.assertAlmostEqual(result[0, 2], expected_y1, delta=1)
        self.assertAlmostEqual(result[0, 3], expected_x2, delta=1)
        self.assertAlmostEqual(result[0, 4], expected_y2, delta=1)

    def test_scale_boxes_boundary_constraints(self):
        """Test that scaled boxes don't exceed image boundaries"""
        factor = 0.2
        # Create a box near the boundary
        boundary_box = np.array([[0, 5, 5, 20, 15]])  # Very close to left and top edges

        annotations = MockAnnotations(boundary_box, (self.img_width, self.img_height))
        result = scale_bounding_boxes(annotations, factor)

        # Check that we don't go below 0
        self.assertGreaterEqual(result[0, 1], 0)
        self.assertGreaterEqual(result[0, 2], 0)

        # Create a box near the right and bottom boundary
        boundary_box = np.array(
            [[0, self.img_width - 20, self.img_height - 15, self.img_width - 5, self.img_height - 5]]
        )

        annotations = MockAnnotations(boundary_box, (self.img_width, self.img_height))
        result = scale_bounding_boxes(annotations, factor)

        # Check that we don't exceed image dimensions
        self.assertLessEqual(result[0, 3], self.img_width - 1)
        self.assertLessEqual(result[0, 4], self.img_height - 1)


class TestModelCache(unittest.TestCase):
    """Tests for the SAM/FastSAM model cache (load-once behavior)."""

    def setUp(self):
        # Count constructor calls and stub out the real ultralytics models
        self.sam_calls = []
        self.fastsam_calls = []

        def fake_sam(path):
            self.sam_calls.append(str(path))
            return ("SAM", str(path))

        def fake_fastsam(path):
            self.fastsam_calls.append(str(path))
            return ("FastSAM", str(path))

        self._orig_sam = converter.SAM
        self._orig_fastsam = converter.FastSAM
        converter.SAM = fake_sam
        converter.FastSAM = fake_fastsam
        clear_model_cache()

    def tearDown(self):
        converter.SAM = self._orig_sam
        converter.FastSAM = self._orig_fastsam
        clear_model_cache()

    def test_model_loaded_once_and_reused(self):
        first = load_sam_model("sam_b")
        second = load_sam_model("sam_b")
        # Same instance returned, constructor called only once
        self.assertIs(first, second)
        self.assertEqual(len(self.sam_calls), 1)

    def test_name_and_pt_suffix_share_cache_entry(self):
        load_sam_model("sam_b")
        load_sam_model("sam_b.pt")  # resolves to the same weights path
        self.assertEqual(len(self.sam_calls), 1)

    def test_fastsam_routed_to_fastsam_loader(self):
        load_sam_model("FastSAM-s")
        self.assertEqual(len(self.fastsam_calls), 1)
        self.assertEqual(len(self.sam_calls), 0)

    def test_clear_model_cache_forces_reload(self):
        load_sam_model("sam_b")
        clear_model_cache()
        load_sam_model("sam_b")
        self.assertEqual(len(self.sam_calls), 2)


class TestConfidence(unittest.TestCase):
    """Tests for the per-OBB confidence score."""

    def setUp(self):
        self.img_width = 640
        self.img_height = 480
        # Single HBB covering the region the test masks live in
        self.hbb_boxes = np.array([[0, 100, 100, 300, 200]])

    def test_rectangular_mask_high_confidence(self):
        """A well-aligned rectangular mask fills its min-area rect -> confidence ~1.0."""
        mask = np.zeros((self.img_height, self.img_width), dtype=bool)
        mask[120:180, 120:280] = True
        _, _, _, confidences = create_obb_annotations_multi_model(
            [self.hbb_boxes[0]], [[mask]], opening_kernel_percentage=0.0
        )

        self.assertEqual(len(confidences), 1)
        self.assertGreater(confidences[0], 0.95, "Rectangular mask should score near 1.0")
        self.assertLessEqual(confidences[0], 1.0)

    def test_fallback_zero_confidence(self):
        """No usable mask -> HBB fallback with confidence 0.0."""
        mask = np.zeros((self.img_height, self.img_width), dtype=bool)
        _, _, _, confidences = create_obb_annotations_multi_model(
            [self.hbb_boxes[0]], [[mask]], opening_kernel_percentage=0.0
        )

        self.assertEqual(confidences, [0.0])

    def test_disagreeing_models_lower_consensus(self):
        """Two disagreeing masks yield lower confidence than two identical masks."""
        mask_a = np.zeros((self.img_height, self.img_width), dtype=bool)
        mask_a[120:180, 120:280] = True
        mask_b = np.zeros((self.img_height, self.img_width), dtype=bool)
        mask_b[130:170, 150:250] = True  # overlaps but differs from mask_a

        _, _, _, conf_agree = create_obb_annotations_multi_model(
            [self.hbb_boxes[0]], [[mask_a], [mask_a]], opening_kernel_percentage=0.0
        )
        _, _, _, conf_disagree = create_obb_annotations_multi_model(
            [self.hbb_boxes[0]], [[mask_a], [mask_b]], opening_kernel_percentage=0.0
        )

        self.assertLess(conf_disagree[0], conf_agree[0], "Disagreeing models should lower confidence")

    def test_all_confidences_within_unit_range(self):
        """Confidence scores are always within [0, 1]."""
        mask = np.zeros((self.img_height, self.img_width), dtype=bool)
        mask[120:180, 120:280] = True
        empty = np.zeros((self.img_height, self.img_width), dtype=bool)
        boxes = np.array([[0, 100, 100, 300, 200], [1, 400, 300, 500, 400]])
        _, _, _, confidences = create_obb_annotations_multi_model(boxes, [[mask, empty]], opening_kernel_percentage=0.0)

        self.assertEqual(len(confidences), 2)
        for c in confidences:
            self.assertGreaterEqual(c, 0.0)
            self.assertLessEqual(c, 1.0)


class TestSaveConfidenceRoundTrip(unittest.TestCase):
    """save_obb_annotations with confidences -> parse_obb_file still reads the boxes."""

    def test_confidence_column_written_and_parsed(self):
        obb_annotations = np.array([[0, 100, 100, 300, 100, 300, 200, 100, 200]])
        confidences = [0.7321]

        with tempfile.TemporaryDirectory() as tmp:
            obb_dir = Path(tmp)
            img_path = obb_dir / "sample.jpg"
            save_obb_annotations(obb_annotations, obb_dir, img_path, confidences=confidences)

            saved = obb_dir / "sample.txt"
            parts = saved.read_text().strip().split()
            self.assertEqual(len(parts), 10, "Line should have label + 8 coords + confidence")
            self.assertAlmostEqual(float(parts[9]), 0.7321, places=4)

            # The evaluator's parser tolerates (and ignores) the 10th column
            boxes = parse_obb_file(saved)
            self.assertEqual(len(boxes), 1, "Parser must not skip the 10-field line")
            self.assertEqual(boxes[0]["label"], 0)


class TestPackUnpackResults(unittest.TestCase):
    """unpack_results() must invert pack_results() for every combination of the two flags,
    exactly as the CLI relies on, without either side hard-coding the other's append order."""

    def setUp(self):
        self.obb_annotations = np.array([[0, 100, 100, 300, 100, 300, 200, 100, 200]])
        self.confidences = [0.5]
        self.contours = [np.array([[100, 100]])]

    def _round_trip(self, return_confidence, return_contours):
        packed = pack_results(self.obb_annotations, self.confidences, self.contours, return_confidence, return_contours)
        return unpack_results(packed, return_confidence, return_contours)

    def test_neither_flag(self):
        obb, confidences, contours = self._round_trip(False, False)
        np.testing.assert_array_equal(obb, self.obb_annotations)
        self.assertIsNone(confidences)
        self.assertIsNone(contours)

    def test_confidence_only(self):
        obb, confidences, contours = self._round_trip(True, False)
        np.testing.assert_array_equal(obb, self.obb_annotations)
        self.assertEqual(confidences, self.confidences)
        self.assertIsNone(contours)

    def test_contours_only(self):
        obb, confidences, contours = self._round_trip(False, True)
        np.testing.assert_array_equal(obb, self.obb_annotations)
        self.assertIsNone(confidences)
        self.assertEqual(contours, self.contours)

    def test_both_flags(self):
        obb, confidences, contours = self._round_trip(True, True)
        np.testing.assert_array_equal(obb, self.obb_annotations)
        self.assertEqual(confidences, self.confidences)
        self.assertEqual(contours, self.contours)


class TestResolveConfidences(unittest.TestCase):
    """Tests for picking which confidence score gets reported."""

    def setUp(self):
        self.conversion = [0.9, 0.5]
        self.detector = np.array([0.8, 0.4])

    def test_conversion_source_is_unchanged(self):
        """The default leaves the heuristic conversion scores untouched."""
        result = resolve_confidences(self.conversion, self.detector, "conversion")
        self.assertEqual(result, self.conversion)

    def test_detector_source(self):
        """'detector' reports the score parsed from the HBB input file."""
        result = resolve_confidences(self.conversion, self.detector, "detector")
        np.testing.assert_allclose(result, [0.8, 0.4])

    def test_combined_source(self):
        """'combined' multiplies the detector and conversion scores."""
        result = resolve_confidences(self.conversion, self.detector, "combined")
        np.testing.assert_allclose(result, [0.72, 0.2])

    def test_missing_detector_score_falls_back(self):
        """A nan detector score (no confidence column) falls back to the conversion score."""
        detector = np.array([0.8, float("nan")])

        np.testing.assert_allclose(resolve_confidences(self.conversion, detector, "detector"), [0.8, 0.5])
        np.testing.assert_allclose(resolve_confidences(self.conversion, detector, "combined"), [0.72, 0.5])

    def test_no_detector_scores_at_all(self):
        """With no scores available, every box falls back and nothing is nan."""
        detector = np.array([float("nan"), float("nan")])
        result = resolve_confidences(self.conversion, detector, "detector")

        np.testing.assert_allclose(result, self.conversion)
        self.assertFalse(np.isnan(result).any())

    def test_empty_input(self):
        """Empty input yields an empty result for every source."""
        for source in ("conversion", "detector", "combined"):
            self.assertEqual(list(resolve_confidences([], np.empty((0,)), source)), [])

    def test_unsupported_source_raises(self):
        """An unknown confidence source is rejected rather than silently ignored."""
        with self.assertRaises(ValueError):
            resolve_confidences(self.conversion, self.detector, "detector_only")

    def test_length_mismatch_raises(self):
        """A detector_scores/conversion_scores length mismatch is rejected, not silently masked."""
        with self.assertRaises(ValueError):
            resolve_confidences(self.conversion, np.array([0.8]), "detector")


class TestSavePolygonAnnotations(unittest.TestCase):
    """Tests for writing the segmentation polygons alongside the OBBs."""

    def setUp(self):
        img_height, img_width = 480, 640
        self.hbb_boxes = np.array(
            [
                [0, 100, 100, 300, 200],  # [label, x1, y1, x2, y2]
                [1, 400, 300, 500, 400],
            ]
        )

        # A rotated ellipse inside the first HBB only, so the second object falls back to its HBB
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        cv2.ellipse(mask, (200, 150), (80, 40), 20, 0, 360, 1, -1)

        self.obb_annotations, _, self.contours, self.confidences = create_obb_annotations_multi_model(
            self.hbb_boxes, [[mask.astype(bool)]], opening_kernel_percentage=0.0
        )

    def write_both(self, tmp, confidences=None, simplify_epsilon=0.0):
        """Write the OBB and polygon files for a fake image and return their lines."""
        img_path = Path(tmp) / "images" / "sample.jpg"
        obb_dir = Path(tmp) / "labels_obb"
        polygon_dir = Path(tmp) / "labels_polygon"

        save_obb_annotations(self.obb_annotations, obb_dir, img_path, confidences=confidences)
        save_polygon_annotations(
            self.contours, self.obb_annotations, polygon_dir, img_path, confidences, simplify_epsilon
        )

        return (
            (obb_dir / "sample.txt").read_text().splitlines(),
            (polygon_dir / "sample.txt").read_text().splitlines(),
        )

    def test_setup_has_one_contour_and_one_fallback(self):
        """The fixture must exercise both the segmented and the fallback path"""
        self.assertIsNotNone(self.contours[0], "First object should have a contour")
        self.assertIsNone(self.contours[1], "Second object should be a fallback")

    def test_fallback_writes_four_point_line(self):
        """A fallback object is written as a four-point rectangle, not skipped"""
        with tempfile.TemporaryDirectory() as tmp:
            obb_lines, polygon_lines = self.write_both(tmp)

        parts = polygon_lines[1].split()
        self.assertEqual(len(parts), 9, "Fallback line should be a label plus four points")
        self.assertEqual(parts[1:], obb_lines[1].split()[1:], "Fallback polygon should equal its OBB row")

    def test_row_aligned_with_obb_file(self):
        """The polygon file has one line per OBB, in the same order"""
        with tempfile.TemporaryDirectory() as tmp:
            obb_lines, polygon_lines = self.write_both(tmp)

        self.assertEqual(len(polygon_lines), len(obb_lines), "Both files should have the same number of lines")
        self.assertEqual(
            [line.split()[0] for line in polygon_lines],
            [line.split()[0] for line in obb_lines],
            "Class labels should match line by line",
        )

    def test_segmented_object_written_as_polygon(self):
        """A segmented object keeps its full contour, not just four corners"""
        with tempfile.TemporaryDirectory() as tmp:
            _, polygon_lines = self.write_both(tmp)

        parts = polygon_lines[0].split()
        self.assertGreater(len(parts), 9, "A segmented object should have more than four points")
        self.assertEqual((len(parts) - 1) % 2, 0, "Coordinates should come in x y pairs")

    def test_confidence_column_mirrors_obb_file(self):
        """The trailing confidence field is identical in both files"""
        with tempfile.TemporaryDirectory() as tmp:
            obb_lines, polygon_lines = self.write_both(tmp, confidences=self.confidences)

        for polygon_line, obb_line in zip(polygon_lines, obb_lines):
            self.assertEqual(polygon_line.split()[-1], obb_line.split()[-1])
        self.assertEqual(float(polygon_lines[1].split()[-1]), 0.0, "Fallback should keep confidence 0.0")

    def test_simplification_reduces_vertices(self):
        """A positive epsilon drops vertices but never below a triangle"""
        with tempfile.TemporaryDirectory() as tmp:
            _, raw_lines = self.write_both(tmp)
        with tempfile.TemporaryDirectory() as tmp:
            _, simplified_lines = self.write_both(tmp, simplify_epsilon=0.02)

        raw_points = (len(raw_lines[0].split()) - 1) // 2
        simplified_points = (len(simplified_lines[0].split()) - 1) // 2
        self.assertLess(simplified_points, raw_points, "Simplification should drop vertices")
        self.assertGreaterEqual(simplified_points, 3, "Simplification should never leave fewer than 3 points")
        self.assertEqual(len(simplified_lines[1].split()), 9, "Fallback rectangles are never simplified")

    def test_empty_annotations_write_empty_file(self):
        """An image with no objects produces an empty polygon file"""
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "images" / "sample.jpg"
            polygon_dir = Path(tmp) / "labels_polygon"
            save_polygon_annotations([], np.array([]), polygon_dir, img_path)

            self.assertEqual((polygon_dir / "sample.txt").read_text(), "")

    def test_contour_count_mismatch_raises(self):
        """Row alignment is enforced, not assumed"""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                save_polygon_annotations(self.contours[:1], self.obb_annotations, Path(tmp), Path(tmp) / "sample.jpg")


class TestDeviceForwarding(unittest.TestCase):
    """`hbb2obb(device=...)` must reach the ultralytics inference call."""

    class _FakeResult:
        masks = None  # no mask -> the box falls back to its HBB, which is all this test needs

    class _FakeModel:
        def __init__(self):
            self.calls = []

        def __call__(self, _img, **kwargs):
            self.calls.append(kwargs)
            return [TestDeviceForwarding._FakeResult()]

    def setUp(self):
        self.model = self._FakeModel()
        self._orig_loader = converter.load_sam_model
        converter.load_sam_model = lambda _name: self.model

        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.img_path = root / "images" / "sample.jpg"
        self.img_path.parent.mkdir()
        cv2.imwrite(str(self.img_path), np.zeros((200, 200, 3), np.uint8))
        self.hbb_dir = root / "labels_hbb"
        self.hbb_dir.mkdir()
        (self.hbb_dir / "sample.txt").write_text("0 100 100 40 40\n")

    def tearDown(self):
        converter.load_sam_model = self._orig_loader
        self.tmp.cleanup()

    def _run(self, **kwargs):
        converter.hbb2obb(img_path=self.img_path, hbb_dir=self.hbb_dir, sam_models=["sam_b"], **kwargs)
        return self.model.calls[0]

    def test_device_is_passed_through(self):
        self.assertEqual(self._run(device="cpu")["device"], "cpu")

    def test_no_device_leaves_it_unset(self):
        self.assertNotIn("device", self._run())

    def test_explicit_model_kwargs_device_wins(self):
        self.assertEqual(self._run(device="cpu", model_kwargs={"device": "0"})["device"], "0")

    def test_a_shared_model_kwargs_dict_is_not_mutated(self):
        shared = {}
        self._run(device="cpu", model_kwargs=shared)
        self.assertEqual(shared, {})


if __name__ == "__main__":
    unittest.main()


class TestConfidenceSidecar(unittest.TestCase):
    """save_confidence_annotations writes the scores beside the labels, not inside them."""

    def test_scores_are_written_one_per_line_and_row_aligned(self):
        obb_annotations = np.array(
            [
                [0, 100, 100, 300, 100, 300, 200, 100, 200],
                [1, 400, 400, 600, 400, 600, 500, 400, 500],
            ]
        )
        confidences = [0.7321, 0.0]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_path = root / "images" / "sample.jpg"
            img_path.parent.mkdir()

            save_obb_annotations(obb_annotations, root / "labels_obb", img_path)
            save_confidence_annotations(confidences, root / "labels_confidence", img_path)

            labels = (root / "labels_obb" / "sample.txt").read_text().splitlines()
            scores = (root / "labels_confidence" / "sample.txt").read_text().splitlines()

            # The label file keeps its nine standard fields; the scores live next door, row-aligned.
            self.assertEqual([len(line.split()) for line in labels], [9, 9])
            self.assertEqual(len(scores), len(labels))
            self.assertAlmostEqual(float(scores[0]), 0.7321, places=4)
            self.assertAlmostEqual(float(scores[1]), 0.0, places=4)

    def test_the_directory_defaults_beside_the_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "images" / "sample.jpg"
            img_path.parent.mkdir()
            save_confidence_annotations([0.5], None, img_path)
            self.assertTrue((Path(tmp) / "labels_confidence" / "sample.txt").is_file())
