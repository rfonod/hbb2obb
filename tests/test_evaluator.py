import tempfile
from pathlib import Path

from shapely.geometry import Polygon

from hbb2obb.evaluator import (
    calculate_obb_iou,
    evaluate_obb,
    format_bbox,
    is_edge_box,
    match_boxes,
    parse_obb_file,
    print_results,
)


class TestEvaluatorUtils:
    def test_parse_obb_file(self, gt_dir):
        """Test parsing an OBB file."""
        sample_file = list(gt_dir.glob('*.txt'))[0]
        boxes = parse_obb_file(sample_file)

        # Check that boxes are parsed correctly
        assert isinstance(boxes, list)
        assert len(boxes) > 0
        assert 'label' in boxes[0]
        assert 'points' in boxes[0]
        assert 'polygon' in boxes[0]
        assert isinstance(boxes[0]['points'], list)
        assert isinstance(boxes[0]['polygon'], Polygon)

    def test_calculate_obb_iou(self, sample_obb_boxes):
        """Test IoU calculation between oriented bounding boxes."""
        # Test same polygon (IoU = 1.0)
        box1 = sample_obb_boxes[0]
        iou = calculate_obb_iou(box1['polygon'], box1['polygon'])
        assert iou == 1.0

        # Test disjoint polygons (IoU = 0.0)
        box2 = sample_obb_boxes[1]
        iou = calculate_obb_iou(box1['polygon'], box2['polygon'])
        assert iou == 0.0

        # Test overlapping polygons
        overlap_poly = Polygon([[150, 150], [150, 250], [250, 250], [250, 150]])
        iou = calculate_obb_iou(box1['polygon'], overlap_poly)
        assert 0.0 < iou < 1.0

        # Check the value is correct (area of intersection / area of union)
        box1_area = box1['polygon'].area
        overlap_area = overlap_poly.area
        intersection_area = box1['polygon'].intersection(overlap_poly).area
        expected_iou = intersection_area / (box1_area + overlap_area - intersection_area)
        assert abs(iou - expected_iou) < 1e-10

    def test_match_boxes(self, sample_gt_boxes, sample_pred_boxes):
        """Test matching ground truth boxes to prediction boxes."""
        # Create simplified test boxes with string identifiers
        test_gt = [{'id': '1', 'label': 'Car', 'polygon': sample_gt_boxes[0]['polygon']}]
        test_pred = [{'id': '2', 'label': 'Car', 'polygon': sample_pred_boxes[0]['polygon']}]

        matches, unmatched_gt, unmatched_pred = match_boxes(test_gt, test_pred, iou_threshold=0.1, class_agnostic=False)

        # Basic checks on the outputs
        assert isinstance(matches, list)
        assert isinstance(unmatched_gt, list)
        assert isinstance(unmatched_pred, list)

        # Should match since IoU > threshold
        assert len(matches) > 0

    def test_is_edge_box(self, sample_obb_boxes):
        """Test detection of edge boxes."""
        # Non-edge box
        box = sample_obb_boxes[0]
        assert not is_edge_box(box, 1000, 1000, tolerance=1)

        # Edge box (touching left edge)
        edge_box = {
            'label': 0,
            'points': [[0, 100], [0, 200], [50, 200], [50, 100]],
            'polygon': Polygon([[0, 100], [0, 200], [50, 200], [50, 100]]),
        }
        assert is_edge_box(edge_box, 1000, 1000, tolerance=1)

        # Test with tolerance
        near_edge_box = {
            'label': 0,
            'points': [[5, 100], [5, 200], [55, 200], [55, 100]],
            'polygon': Polygon([[5, 100], [5, 200], [55, 200], [55, 100]]),
        }
        assert is_edge_box(near_edge_box, 1000, 1000, tolerance=5)
        assert not is_edge_box(near_edge_box, 1000, 1000, tolerance=4)

    def test_format_bbox(self, sample_gt_boxes):
        """Test bbox formatting for display."""
        box = sample_gt_boxes[0]
        formatted = format_bbox(box)
        assert isinstance(formatted, str)
        assert "Car" in formatted  # Label should be in the formatted string
        assert "100.0,100.0" in formatted  # Check for coordinate formatting instead
        assert "200.0,200.0" in formatted  # Check for another coordinate


class TestEvaluator:
    def test_evaluate_obb_with_real_data(self, gt_dir, pred_dir):
        """Test the OBB evaluation function with real data."""
        results = evaluate_obb(gt_dir=gt_dir, pred_dir=pred_dir, iou_threshold=0.1)

        # Check that results contain expected fields
        assert 'total_gt' in results
        assert 'total_pred' in results
        assert 'total_matches' in results
        assert 'avg_iou' in results
        assert 'std_iou' in results
        assert 'class_stats' in results

        # Basic sanity checks
        assert results['total_gt'] > 0
        assert results['total_pred'] > 0
        assert 0.0 <= results['avg_iou'] <= 1.0

    def test_evaluate_obb_with_excluded_classes(self, gt_dir, pred_dir):
        """Test OBB evaluation with excluded classes."""
        # First run without exclusions
        results_all = evaluate_obb(gt_dir=gt_dir, pred_dir=pred_dir, iou_threshold=0.1)

        # Run with exclusion of class 3 (Motorcycle from classes.yaml)
        results_excluded = evaluate_obb(gt_dir=gt_dir, pred_dir=pred_dir, excluded_classes=[3], iou_threshold=0.1)

        # Should have fewer GT and pred boxes when excluding a class
        assert results_excluded['total_gt'] <= results_all['total_gt']
        assert results_excluded['total_pred'] <= results_all['total_pred']
        assert 3 not in results_excluded['class_stats'].keys()

    def test_evaluate_obb_class_agnostic(self, gt_dir, pred_dir):
        """Test OBB evaluation with class-agnostic setting."""
        # Run with normal class-specific matching
        results_specific = evaluate_obb(gt_dir=gt_dir, pred_dir=pred_dir, iou_threshold=0.1, class_agnostic=False)

        # Run with class-agnostic matching
        results_agnostic = evaluate_obb(gt_dir=gt_dir, pred_dir=pred_dir, iou_threshold=0.1, class_agnostic=True)

        # Should have more matches in class-agnostic mode
        assert results_agnostic['total_matches'] >= results_specific['total_matches']
        assert results_agnostic['cross_class_matches'] >= 0

    def test_evaluate_obb_edge_cases(self, gt_dir, pred_dir):
        """Test evaluation with edge case exclusion."""
        # Run without edge case exclusion
        results_with_edges = evaluate_obb(gt_dir=gt_dir, pred_dir=pred_dir, iou_threshold=0.1)

        # Run with edge case exclusion
        results_no_edges = evaluate_obb(
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            iou_threshold=0.1,
            exclude_edge_cases=True,
            img_width=3840,  # Using common 4K resolution
            img_height=2160,
        )

        # Should have fewer or equal GT boxes when excluding edge cases
        assert results_no_edges['total_gt'] <= results_with_edges['total_gt']
        assert results_no_edges['total_edge_cases'] >= 0

    def test_print_results(self, gt_dir, pred_dir, capsys):
        """Test that results printing works without errors."""
        results = evaluate_obb(gt_dir=gt_dir, pred_dir=pred_dir, iou_threshold=0.1)

        # Test with plain results
        print_results(results)
        captured = capsys.readouterr()
        assert "Total Ground Truth Boxes" in captured.out
        assert "Total Predicted/Converted Boxes" in captured.out
        assert "Average IoU" in captured.out

        # Test with label map
        print_results(results, map_path=Path(__file__).parent.parent / 'data' / 'classes.yaml')
        captured = capsys.readouterr()
        assert "Car" in captured.out  # Class 0 should be printed as Car


class TestEndToEnd:
    def test_eval_workflow_with_temp_files(self, sample_gt_boxes, sample_pred_boxes):
        """Test the full evaluation workflow with temporary files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            gt_dir = Path(temp_dir) / 'gt'
            pred_dir = Path(temp_dir) / 'pred'
            gt_dir.mkdir()
            pred_dir.mkdir()

            # Create a test file
            gt_file = gt_dir / 'test.txt'
            pred_file = pred_dir / 'test.txt'

            # Write sample data to files
            with open(gt_file, 'w') as f:
                for box in sample_gt_boxes:
                    if box['label'] == 'Car':
                        label = 0
                    elif box['label'] == 'Bus':
                        label = 1
                    else:
                        label = 2
                    points = box['points'].flatten().tolist()
                    line = f"{label} " + " ".join(map(str, points))
                    f.write(line + '\n')

            with open(pred_file, 'w') as f:
                for box in sample_pred_boxes:
                    if box['label'] == 'Car':
                        label = 0
                    elif box['label'] == 'Bus':
                        label = 1
                    else:
                        label = 2
                    points = box['points'].flatten().tolist()
                    line = f"{label} " + " ".join(map(str, points))
                    f.write(line + '\n')

            # Run evaluation
            results = evaluate_obb(gt_dir=gt_dir, pred_dir=pred_dir, iou_threshold=0.1)

            # Verify results
            assert 'total_gt' in results
            assert 'total_pred' in results
            assert results['total_gt'] == len(sample_gt_boxes)
            assert results['total_pred'] == len(sample_pred_boxes)
