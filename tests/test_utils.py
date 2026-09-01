import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from hbb2obb.utils import Annotations, get_image_paths, load_label_map, process_ultralytics_kwargs


class TestUtils:
    def test_load_label_map_from_file(self, label_map_path):
        """Test loading label map from a file."""
        label_map = load_label_map(label_map_path)

        # Check that the map was loaded correctly
        assert isinstance(label_map, dict)
        assert 0 in label_map
        assert label_map[0] == "Car"
        assert 1 in label_map
        assert label_map[1] == "Bus"
        assert 2 in label_map
        assert label_map[2] == "Truck"
        assert 3 in label_map
        assert label_map[3] == "Motorcycle"

    def test_load_label_map_with_invalid_path(self):
        """Test behavior with invalid path."""
        # Should return None for non-existent path
        assert load_label_map(Path("/nonexistent/path.yaml")) is None

    def test_load_label_map_with_custom_file(self):
        """Test loading a custom label map file."""
        tmp_path = None
        try:
            # Create a temporary file
            fd, tmp_path = tempfile.mkstemp(suffix='.yaml')
            os.close(fd)

            # Write the YAML content to the file
            custom_map = {0: "Object1", 1: "Object2", 2: "Object3"}
            with open(tmp_path, 'w') as f:
                yaml.safe_dump(custom_map, f)

            # Load the custom map
            label_map = load_label_map(Path(tmp_path))

            # Verify contents
            assert isinstance(label_map, dict)
            assert len(label_map) == 3
            assert label_map[0] == "Object1"
            assert label_map[1] == "Object2"
            assert label_map[2] == "Object3"
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_load_label_map_with_reversed_format(self):
        """Test loading a label map with reversed key-value pairs."""
        tmp_path = None
        try:
            # Create a temporary file
            fd, tmp_path = tempfile.mkstemp(suffix='.yaml')
            os.close(fd)

            # Write the YAML content to the file
            custom_map = {"Object1": 0, "Object2": 1, "Object3": 2}
            with open(tmp_path, 'w') as f:
                yaml.safe_dump(custom_map, f)

            # Load the custom map with reverse=True
            label_map = load_label_map(Path(tmp_path), reverse=True)

            # Verify contents
            assert isinstance(label_map, dict)
            assert len(label_map) == 3
            assert label_map[0] == "Object1"
            assert label_map[1] == "Object2"
            assert label_map[2] == "Object3"
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_get_image_paths(self, tmp_path):
        """Test getting image paths from a directory."""
        # Create test image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        created_files = []

        for ext in image_extensions:
            file_path = tmp_path / f"test{ext}"
            file_path.touch()
            created_files.append(file_path)

        # Add a non-image file
        non_image = tmp_path / "test.txt"
        non_image.touch()

        # Get image paths
        image_paths = get_image_paths(tmp_path)

        # Verify all image files are found
        assert len(image_paths) == len(image_extensions)
        for file in created_files:
            assert file in image_paths
        assert non_image not in image_paths

    def test_process_ultralytics_kwargs(self):
        """Test processing keyword arguments for Ultralytics models."""
        # Test with valid input
        kwargs_string = "conf=0.25,iou=0.45,max_det=300"
        result = process_ultralytics_kwargs(kwargs_string)

        assert isinstance(result, dict)
        assert result['conf'] == 0.25
        assert result['iou'] == 0.45
        assert result['max_det'] == 300

        # Test with mixed types
        kwargs_string = "conf=0.25,iou=0.45,max_det=300,verbose=True,agnostic=False"
        result = process_ultralytics_kwargs(kwargs_string)

        assert result['conf'] == 0.25
        assert result['iou'] == 0.45
        assert result['max_det'] == 300
        assert result['verbose'] is True
        assert result['agnostic'] is False

        # Test with empty string
        assert process_ultralytics_kwargs("") == {}

        # Test with invalid input
        result = process_ultralytics_kwargs("conf=0.25,iou")
        assert result == {}


@pytest.fixture
def img():
    """A small blank image (200 px wide, 100 px tall) to size the annotations against."""
    return np.zeros((100, 200, 3), np.uint8)


def _annotations(tmp_path, content, img, input_format="xywh"):
    """Write a label file and parse it."""
    label_file = tmp_path / "frame.txt"
    label_file.write_text(content, encoding="utf-8")
    return Annotations(label_file, img, input_format=input_format)


class TestAnnotations:
    def test_absolute_xywh(self, tmp_path, img):
        """Absolute-pixel xywh lines are parsed without rescaling."""
        ann = _annotations(tmp_path, "0 100 50 40 20\n", img)

        assert ann.normalized is False
        np.testing.assert_allclose(ann.hbb_xyxy, [[0, 80, 40, 120, 60]])
        np.testing.assert_allclose(ann.hbb_xywh, [[0, 100, 50, 40, 20]])
        assert np.isnan(ann.hbb_scores).all()

    def test_normalized_xywh(self, tmp_path, img):
        """Relative coordinates are scaled by the image dimensions."""
        ann = _annotations(tmp_path, "1 0.5 0.5 0.2 0.2\n", img)

        assert ann.normalized is True
        np.testing.assert_allclose(ann.hbb_xyxy, [[1, 80, 40, 120, 60]])

    def test_absolute_xyxy(self, tmp_path, img):
        """The xyxy input format is parsed as corner coordinates."""
        ann = _annotations(tmp_path, "2 80 40 120 60\n", img, input_format="xyxy")

        np.testing.assert_allclose(ann.hbb_xyxy, [[2, 80, 40, 120, 60]])

    def test_confidence_column_xywh(self, tmp_path, img):
        """A 6th confidence column (what detectors write) is parsed, not rejected."""
        ann = _annotations(tmp_path, "0 100 50 40 20 0.87\n1 20 20 10 10 0.42\n", img)

        np.testing.assert_allclose(ann.hbb_xyxy[0], [0, 80, 40, 120, 60])
        np.testing.assert_allclose(ann.hbb_scores, [0.87, 0.42])

    def test_confidence_column_xyxy(self, tmp_path, img):
        """The same holds for the xyxy input format."""
        ann = _annotations(tmp_path, "0 80 40 120 60 0.87\n", img, input_format="xyxy")

        np.testing.assert_allclose(ann.hbb_xyxy, [[0, 80, 40, 120, 60]])
        np.testing.assert_allclose(ann.hbb_scores, [0.87])

    def test_ragged_confidence_columns(self, tmp_path, img):
        """Mixed 5- and 6-field lines parse, with nan marking the missing scores."""
        ann = _annotations(tmp_path, "0 100 50 40 20 0.87\n1 20 20 10 10\n", img)

        assert len(ann.hbb_xyxy) == 2
        assert ann.hbb_scores[0] == pytest.approx(0.87)
        assert np.isnan(ann.hbb_scores[1])

    def test_extra_trailing_fields_raise(self, tmp_path, img):
        """A line with tokens past the optional confidence column is rejected, not silently truncated."""
        with pytest.raises(ValueError):
            _annotations(tmp_path, "0 100 50 40 20 0.87 garbage\n", img)

    def test_empty_file(self, tmp_path, img):
        """An empty label file (a frame with no objects) yields empty arrays, not an IndexError."""
        ann = _annotations(tmp_path, "", img)

        assert ann.hbb_xyxy.shape == (0, 5)
        assert ann.hbb_xywh.shape == (0, 5)
        assert ann.hbb_scores.shape == (0,)
        assert ann.normalized is False

    def test_blank_only_file(self, tmp_path, img):
        """A file holding nothing but blank lines is treated as empty."""
        ann = _annotations(tmp_path, "\n\n   \n", img)

        assert ann.hbb_xyxy.shape == (0, 5)
        assert ann.hbb_xywh.shape == (0, 5)

    def test_leading_blank_line(self, tmp_path, img):
        """Normalization is decided from the first non-blank line, and blanks are skipped."""
        ann = _annotations(tmp_path, "\n0 100 50 40 20\n\n1 20 20 10 10\n", img)

        assert ann.normalized is False
        assert len(ann.hbb_xyxy) == 2
        np.testing.assert_allclose(ann.hbb_xyxy[0], [0, 80, 40, 120, 60])

    def test_unsupported_format_raises(self, tmp_path, img):
        """An unknown input format is still rejected."""
        with pytest.raises(ValueError, match="Unsupported format"):
            _annotations(tmp_path, "0 100 50 40 20\n", img, input_format="cxcywha")
