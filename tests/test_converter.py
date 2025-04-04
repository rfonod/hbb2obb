import unittest

import numpy as np

from hbb2obb.converter import aggregate_masks_by_majority_vote, create_obb_annotations_multi_model, scale_bounding_boxes


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

        result, masks, contours = create_obb_annotations_multi_model(self.hbb_boxes[0:1], all_models_masks, opening_kernel_percentage=0.0)

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

        result, masks, contours = create_obb_annotations_multi_model(self.hbb_boxes[0:1], all_models_masks, opening_kernel_percentage=0.0)

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

        result, masks, contours = create_obb_annotations_multi_model(self.hbb_boxes[0:1], all_models_masks, opening_kernel_percentage=0.0)

        self.assertEqual(len(result), 1, "Should return 1 OBB annotation")
        self.assertEqual(result[0][0], 0, "Label should be preserved")

    def test_create_obb_annotations_no_valid_masks(self):
        """Test create_obb_annotations with no valid masks"""
        # Create an empty mask
        mask = np.zeros((self.img_height, self.img_width), dtype=bool)
        all_models_masks = [[mask]]

        result, masks, contours = create_obb_annotations_multi_model(self.hbb_boxes[0:1], all_models_masks, opening_kernel_percentage=0.0)

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

        result, masks, contours = create_obb_annotations_multi_model(self.hbb_boxes[0:1], all_models_masks, opening_kernel_percentage=0.0)

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


if __name__ == "__main__":
    unittest.main()
