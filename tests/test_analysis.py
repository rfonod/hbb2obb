import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


from hbb2obb import analysis


def rect(xc, yc, w, h, angle_deg=0.0):
    """The four corners of a rotated rectangle, in absolute pixels."""
    a = math.radians(angle_deg)
    cos, sin = math.cos(a), math.sin(a)
    corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    return [(xc + x * cos - y * sin, yc + x * sin + y * cos) for x, y in corners]


def write_obb(path: Path, boxes):
    path.write_text(
        "\n".join(f"{cls} " + " ".join(f"{v:.2f}" for pt in pts for v in pt) for cls, pts in boxes) + "\n",
        encoding="utf-8",
    )


class TestGeometry(unittest.TestCase):
    def test_box_sides_are_short_then_long(self):
        short, long_ = analysis.box_sides(rect(100, 100, 200, 50))
        self.assertAlmostEqual(short, 50, places=6)
        self.assertAlmostEqual(long_, 200, places=6)

    def test_box_sides_do_not_care_how_the_box_is_turned(self):
        turned = analysis.box_sides(rect(100, 100, 200, 50, 37.0))
        self.assertAlmostEqual(turned[0], 50, places=6)
        self.assertAlmostEqual(turned[1], 200, places=6)

    def test_off_axis_tops_out_at_45_degrees(self):
        self.assertAlmostEqual(analysis.off_axis_angle(rect(0, 0, 100, 20, 0)), 0.0, places=6)
        self.assertAlmostEqual(analysis.off_axis_angle(rect(0, 0, 100, 20, 90)), 0.0, places=6)
        self.assertAlmostEqual(analysis.off_axis_angle(rect(0, 0, 100, 20, 45)), 45.0, places=6)
        self.assertAlmostEqual(analysis.off_axis_angle(rect(0, 0, 100, 20, 100)), 10.0, places=6)


class TestCollectPairs(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        root = Path(self.tmp.name)
        self.gt, self.pred, self.hbb = root / "gt", root / "pred", root / "hbb"
        for d in (self.gt, self.pred, self.hbb):
            d.mkdir()
        # Two vehicles: one square to the image, one turned 30 degrees.
        self.gt_boxes = [(0, rect(200, 200, 100, 40, 0.0)), (0, rect(600, 400, 100, 40, 30.0))]
        write_obb(self.gt / "frame.txt", self.gt_boxes)
        # The conversion recovers both, the turned one a little wide.
        write_obb(
            self.pred / "frame.txt",
            [(0, rect(200, 200, 100, 40, 0.0)), (0, rect(600, 400, 104, 46, 30.0))],
        )
        # The prompt is the axis-aligned envelope of each.
        write_obb(
            self.hbb / "frame.txt",
            [(0, rect(200, 200, 100, 40, 0.0)), (0, rect(600, 400, 106.6, 84.6, 0.0))],
        )
        self.hbb_yolo = self.hbb / "yolo"
        self.hbb_yolo.mkdir()
        (self.hbb_yolo / "frame.txt").write_text("0 200 200 100 40\n0 600 400 106.6 84.6\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def pairs(self, **kwargs):
        return analysis.collect_pairs(self.gt, self.pred, no_bar=True, **kwargs)

    def test_every_ground_truth_box_is_matched_once(self):
        pairs = self.pairs()
        self.assertEqual(len(pairs), 2)
        self.assertEqual({round(p.off_axis) for p in pairs}, {0, 30})

    def test_a_perfect_box_scores_one(self):
        aligned = next(p for p in self.pairs() if p.off_axis < analysis.AXIS_TOLERANCE)
        self.assertAlmostEqual(aligned.iou, 1.0, places=6)
        self.assertAlmostEqual(aligned.angle, 0.0, places=6)
        self.assertAlmostEqual(aligned.pred_short / aligned.gt_short, 1.0, places=4)

    def test_a_wide_box_shows_up_in_the_side_ratios(self):
        turned = next(p for p in self.pairs() if p.off_axis > 1)
        self.assertGreater(turned.pred_short / turned.gt_short, 1.0)
        self.assertGreater(turned.area_ratio, 1.0)

    def test_the_identity_baseline_reads_the_prompt(self):
        pairs = self.pairs(hbb_dir=self.hbb_yolo)
        aligned = next(p for p in pairs if p.off_axis < analysis.AXIS_TOLERANCE)
        turned = next(p for p in pairs if p.off_axis > 1)
        self.assertAlmostEqual(aligned.identity_iou, 1.0, places=6)
        self.assertLess(turned.identity_iou, turned.iou, "the envelope of a turned box is worse than the fit")

    def test_without_a_prompt_directory_there_is_no_baseline(self):
        self.assertTrue(all(p.identity_iou is None for p in self.pairs()))

    def test_the_edge_cut_needs_image_sizes(self):
        self.assertTrue(all(p.edge is None for p in self.pairs()))
        sized = self.pairs(image_sizes={"frame": (700, 500)})
        self.assertEqual({p.edge for p in sized}, {False})

    def test_relative_prompts_without_a_frame_size_are_refused(self):
        relative = Path(self.tmp.name) / "hbb_rel"
        relative.mkdir()
        (relative / "frame.txt").write_text("0 0.5 0.5 0.1 0.05\n", encoding="utf-8")
        with self.assertRaises(SystemExit):
            self.pairs(hbb_dir=relative)

    def test_the_difficult_flag_is_read_from_a_dota_file_beside_the_labels(self):
        (self.gt / "frame.dota").write_text(
            "\n".join(
                " ".join(f"{v:.2f}" for pt in pts for v in pt) + f" car {flag}"
                for (_, pts), flag in zip(self.gt_boxes, (0, 1))
            )
            + "\n",
            encoding="utf-8",
        )
        flags = {p.difficult for p in self.pairs()}
        self.assertEqual(flags, {0, 1})

    def test_a_dota_file_of_the_wrong_length_is_ignored(self):
        (self.gt / "frame.dota").write_text("0 0 1 0 1 1 0 1 car 1\n", encoding="utf-8")
        self.assertTrue(all(p.difficult is None for p in self.pairs()))


class TestSummary(unittest.TestCase):
    def make(self, off_axis, iou, **kwargs):
        defaults = dict(
            frame="f",
            cls=0,
            angle=0.0,
            gt_short=40.0,
            gt_long=100.0,
            pred_short=40.0,
            pred_long=100.0,
            area_ratio=1.0,
            edge=False,
            difficult=0,
        )
        defaults.update(kwargs)
        return analysis.Pair(off_axis=off_axis, iou=iou, **defaults)

    def test_the_orientation_partition_puts_every_box_in_exactly_one_band(self):
        pairs = [self.make(0.0, 0.9), self.make(2.0, 0.8), self.make(20.0, 0.7), self.make(40.0, 0.6)]
        counts = [len(group) for _, group in analysis.orientation_partition(pairs)]
        self.assertEqual(sum(counts), len(pairs))

    def test_the_orientation_cut_reports_aligned_and_rotated_separately(self):
        pairs = [self.make(0.0, 1.0), self.make(0.0, 1.0), self.make(20.0, 0.5)]
        named = dict(analysis.summarise(pairs)["orientation"])
        self.assertEqual(named["rotated"]["boxes"], 1)
        self.assertAlmostEqual(named["rotated"]["mean_iou"], 0.5)
        self.assertEqual(named[f"axis-aligned (<{analysis.AXIS_TOLERANCE:g} deg)"]["boxes"], 2)

    def test_the_controlled_cut_drops_edge_and_difficult_boxes(self):
        pairs = [self.make(20.0, 0.9), self.make(20.0, 0.1, edge=True), self.make(20.0, 0.1, difficult=1)]
        summary = analysis.summarise(pairs)
        self.assertEqual(summary["controlled_boxes"], 1)
        rotated = dict(summary["orientation_controlled"])["rotated"]
        self.assertAlmostEqual(rotated["mean_iou"], 0.9)

    def test_the_standard_error_shrinks_with_the_count(self):
        few = analysis.group_stats([self.make(0.0, 0.8), self.make(0.0, 1.0)])
        many = analysis.group_stats([self.make(0.0, 0.8), self.make(0.0, 1.0)] * 50)
        self.assertAlmostEqual(few["std_iou"], many["std_iou"], places=9)
        self.assertLess(many["sem_iou"], few["sem_iou"] / 5)

    def test_the_side_ratio_table_separates_the_axis_aligned_spike(self):
        pairs = [self.make(0.0, 0.9, pred_short=38.0), self.make(2.0, 0.9, pred_short=42.0)]
        rows = {r["band"]: r for r in analysis.summarise(pairs)["side_ratios"]}
        self.assertAlmostEqual(rows["exactly aligned"]["short_ratio"], 38.0 / 40.0)
        self.assertAlmostEqual(rows["1 to 3 deg"]["short_ratio"], 42.0 / 40.0)

    def test_the_side_ratio_table_can_be_held_to_one_class(self):
        pairs = [self.make(0.0, 0.9, pred_short=38.0), self.make(0.0, 0.9, cls=3, pred_short=80.0)]
        rows = {r["band"]: r for r in analysis.summarise(pairs, boundary_classes=[0])["side_ratios"]}
        self.assertEqual(rows["exactly aligned"]["boxes"], 1)
        self.assertAlmostEqual(rows["exactly aligned"]["short_ratio"], 38.0 / 40.0)

    def test_a_baseline_appears_only_when_the_prompts_were_read(self):
        self.assertFalse(analysis.summarise([self.make(0.0, 0.9)])["has_identity"])
        with_prompt = analysis.summarise([self.make(0.0, 0.9, identity_iou=0.7, identity_angle=0.0)])
        self.assertTrue(with_prompt["has_identity"])
        self.assertAlmostEqual(with_prompt["overall"]["identity_iou"], 0.7)


class TestReport(unittest.TestCase):
    def pairs(self):
        return [
            analysis.Pair("f", 0, 0.95, 0.0, 0.0, 40, 100, 39, 99, 0.96, False, 0, 0.95, 0.0),
            analysis.Pair("f", 0, 0.80, 4.0, 20.0, 40, 100, 44, 101, 1.11, False, 0, 0.55, 25.0),
        ]

    def test_the_report_carries_the_headline_and_every_cut(self):
        text = analysis.render_markdown(analysis.summarise(self.pairs()))
        for heading in (
            "By ground-truth orientation",
            "By ground-truth short side",
            "By class",
            "Against doing nothing",
            "Converted-to-reference side ratios",
        ):
            self.assertIn(heading, text)
        self.assertIn("2 matched boxes", text)

    def test_the_baseline_section_is_left_out_without_prompts(self):
        bare = [analysis.Pair("f", 0, 0.9, 0.0, 0.0, 40, 100, 40, 100, 1.0)]
        self.assertNotIn("Against doing nothing", analysis.render_markdown(analysis.summarise(bare)))

    def test_the_yaml_keeps_box_counts_whole(self):
        safe = analysis.to_yaml_safe(analysis.summarise(self.pairs()))
        self.assertIsInstance(safe["orientation"][0]["boxes"], int)

    def test_the_figure_is_written(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "analysis.png"
            analysis.plot_analysis(analysis.summarise(self.pairs()), out)
            self.assertGreater(out.stat().st_size, 1000)

    def test_a_group_with_no_boxes_reports_a_count_and_nothing_else(self):
        self.assertEqual(analysis.group_stats([]), {"boxes": 0})


if __name__ == "__main__":
    unittest.main()
