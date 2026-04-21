import importlib.util
from pathlib import Path


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coco_json_to_yolo_conversion(tmp_path):
    repo_root = Path(__file__).parent.parent
    json2yolo = load_module(
        "json2yolo_module",
        repo_root / "scripts" / "json2yolo.py",
    )

    coco_path = tmp_path / "annotations.json"
    coco_path.write_text(
        """
{
  "images": [{"id": 1, "file_name": "img1.jpg", "width": 100, "height": 200}],
  "categories": [{"id": 5, "name": "Car"}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 5, "bbox": [10, 20, 30, 40]}]
}
        """.strip(),
        encoding="utf-8",
    )

    out_dir = tmp_path / "labels"
    json2yolo.process_files(coco_path, map_path=None, txt_dir=out_dir, normalize=True, input_format="coco")

    out_file = out_dir / "img1.txt"
    assert out_file.exists()
    values = out_file.read_text(encoding="utf-8").strip().split()
    assert len(values) == 5
    assert int(values[0]) == 0
    assert abs(float(values[1]) - 0.25) < 1e-6
    assert abs(float(values[2]) - 0.2) < 1e-6
    assert abs(float(values[3]) - 0.3) < 1e-6
    assert abs(float(values[4]) - 0.2) < 1e-6


def test_pascal_voc_xml_to_yolo_conversion(tmp_path):
    repo_root = Path(__file__).parent.parent
    voc2yolo = load_module(
        "voc2yolo_module",
        repo_root / "scripts" / "voc2yolo.py",
    )

    xml_path = tmp_path / "img1.xml"
    xml_path.write_text(
        """
<annotation>
  <size>
    <width>100</width>
    <height>100</height>
  </size>
  <object>
    <name>Car</name>
    <bndbox>
      <xmin>10</xmin>
      <ymin>20</ymin>
      <xmax>40</xmax>
      <ymax>60</ymax>
    </bndbox>
  </object>
</annotation>
        """.strip(),
        encoding="utf-8",
    )

    out_dir = tmp_path / "labels"
    voc2yolo.process_files(tmp_path, txt_dir=out_dir, normalize=True)

    out_file = out_dir / "img1.txt"
    assert out_file.exists()
    values = out_file.read_text(encoding="utf-8").strip().split()
    assert len(values) == 5
    assert int(values[0]) == 0
    assert abs(float(values[1]) - 0.25) < 1e-6
    assert abs(float(values[2]) - 0.4) < 1e-6
    assert abs(float(values[3]) - 0.3) < 1e-6
    assert abs(float(values[4]) - 0.4) < 1e-6
