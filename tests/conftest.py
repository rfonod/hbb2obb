from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon


@pytest.fixture
def data_dir():
    """Return the path to the data directory."""
    return Path(__file__).parent.parent / 'data'


@pytest.fixture
def gt_dir(data_dir):
    """Return the path to the ground truth labels directory."""
    return data_dir / 'labels_obb_gt'


@pytest.fixture
def pred_dir(data_dir):
    """Return the path to the prediction labels directory."""
    return data_dir / 'labels_obb'


@pytest.fixture
def hbb_dir(data_dir):
    """Return the path to the horizontal bounding box labels directory."""
    return data_dir / 'labels_hbb'


@pytest.fixture
def images_dir(data_dir):
    """Return the path to the images directory."""
    return data_dir / 'images'


@pytest.fixture
def label_map_path(data_dir):
    """Return the path to the label map file."""
    return data_dir / 'classes.yaml'


@pytest.fixture
def sample_obb_boxes():
    """Return sample OBB boxes for testing."""
    return [
        {
            'label': 0,
            'points': np.array([[100, 100], [100, 200], [200, 200], [200, 100]]),
            'polygon': Polygon([[100, 100], [100, 200], [200, 200], [200, 100]]),
        },
        {
            'label': 1,
            'points': np.array([[300, 300], [300, 400], [400, 400], [400, 300]]),
            'polygon': Polygon([[300, 300], [300, 400], [400, 400], [400, 300]]),
        },
    ]


@pytest.fixture
def sample_gt_boxes():
    """Return sample ground truth boxes for testing."""
    return [
        {
            'label': 'Car',
            'points': np.array([[100, 100], [100, 200], [200, 200], [200, 100]]),
            'polygon': Polygon([[100, 100], [100, 200], [200, 200], [200, 100]]),
        },
        {
            'label': 'Bus',
            'points': np.array([[300, 300], [300, 400], [400, 400], [400, 300]]),
            'polygon': Polygon([[300, 300], [300, 400], [400, 400], [400, 300]]),
        },
        {
            'label': 'Car',
            'points': np.array([[500, 500], [500, 600], [600, 600], [600, 500]]),
            'polygon': Polygon([[500, 500], [500, 600], [600, 600], [600, 500]]),
        },
    ]


@pytest.fixture
def sample_pred_boxes():
    """Return sample prediction boxes for testing."""
    return [
        {
            'label': 'Car',
            'points': np.array([[110, 110], [110, 210], [210, 210], [210, 110]]),
            'polygon': Polygon([[110, 110], [110, 210], [210, 210], [210, 110]]),
        },
        {
            'label': 'Car',
            'points': np.array([[320, 320], [320, 420], [420, 420], [420, 320]]),
            'polygon': Polygon([[320, 320], [320, 420], [420, 420], [420, 320]]),
        },
        {
            'label': 'Car',
            'points': np.array([[700, 700], [700, 800], [800, 800], [800, 700]]),
            'polygon': Polygon([[700, 700], [700, 800], [800, 800], [800, 700]]),
        },
    ]
