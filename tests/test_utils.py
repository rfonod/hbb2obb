import os
import tempfile
from pathlib import Path

import yaml

from hbb2obb.utils import get_image_paths, load_label_map, process_ultralytics_kwargs


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
