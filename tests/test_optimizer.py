"""
Tests for hbb2obb.optimizer and the hbb2obb-optimize CLI.

Nothing here loads a checkpoint or runs SAM: the sweep itself is monkeypatched, so what is
tested is the configuration, the bookkeeping and the artifacts, which is where the mistakes
that waste an overnight run actually live.
"""

import sys

import pytest
import yaml

from hbb2obb import optimizer
from hbb2obb.cli import SUPPORTED_SAM_MODELS, main_hbb2obb_optimize


def grid_point(imgsz=1280, sf=0.05, ok=0.15, iou=0.9, angle=2.5):
    """One entry of all_results, with every key the writers read."""
    return {
        "imgsz": imgsz,
        "scale_factor": sf,
        "opening_kernel_percentage": ok,
        "avg_iou": iou,
        "std_iou": 0.07,
        "median_iou": iou + 0.01,
        "iou_fractions": {"0.50": 1.0, "0.75": 0.94, "0.85": 0.73, "0.90": 0.41},
        "median_angle_error": angle,
        "avg_angle_error": angle + 0.4,
        "std_angle_error": 1.2,
        "p90_angle_error": angle + 3.0,
        "total_matches": 200,
        "total_gt": 201,
        "total_pred": 201,
        "class_agnostic": False,
        "iou_threshold": 0.1,
        "excluded_classes": [],
        "execution_time": 12.5,
    }


def fake_outcome(points):
    return {
        "best_parameters": max(points, key=lambda p: p["avg_iou"]),
        "all_results": points,
        "model_load_seconds": 1.0,
        "sweep_seconds": 25.0,
    }


def write_config(tmp_path, path=None, **overrides):
    cfg = {
        "img_source": "data/images",
        "gt_dir": "data/labels_obb_gt",
        "output_folder": str(tmp_path / "bench"),
        "defaults": {"imgsz": [1280], "scale_factors": [0.04, 0.05], "opening_kernels": [0.15]},
        "runs": [{"sam_models": ["sam_b"]}, {"sam_models": ["sam_l", "sam_b"]}],
    }
    cfg.update(overrides)
    path = path or tmp_path / "bench.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    return path


# ------------------------------------------------------------------------------------- config
def test_load_config_rejects_an_unknown_top_level_key(tmp_path):
    path = write_config(tmp_path, output_dir="typo")
    with pytest.raises(SystemExit, match="unknown key"):
        optimizer.load_config(path)


def test_load_config_requires_the_inputs(tmp_path):
    path = tmp_path / "bench.yaml"
    path.write_text(yaml.dump({"img_source": "data/images", "runs": [{"sam_models": ["sam_b"]}]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="'gt_dir' is required"):
        optimizer.load_config(path)


def test_load_config_reports_a_missing_file(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        optimizer.load_config(tmp_path / "absent.yaml")


def test_expand_runs_merges_defaults_and_derives_names(tmp_path):
    cfg = optimizer.load_config(write_config(tmp_path))
    specs = optimizer.expand_runs(cfg, SUPPORTED_SAM_MODELS)

    assert [s.name for s in specs] == ["sam_b", "sam_l-sam_b"]
    assert all(s.imgsz == [1280] for s in specs)
    assert all(s.scale_factors == [0.04, 0.05] for s in specs)
    assert len(specs[0].grid) == 2


def test_a_run_overrides_the_defaults_it_names(tmp_path):
    cfg = optimizer.load_config(
        write_config(tmp_path, runs=[{"sam_models": ["sam_b"], "imgsz": [640, 960]}, {"sam_models": ["sam_l"]}])
    )
    specs = optimizer.expand_runs(cfg, SUPPORTED_SAM_MODELS)

    assert specs[0].imgsz == [640, 960]
    assert specs[1].imgsz == [1280]  # untouched by its neighbour
    assert specs[0].scale_factors == [0.04, 0.05]  # the default it did not override


def test_expand_runs_carries_a_device_from_the_defaults(tmp_path):
    cfg = optimizer.load_config(write_config(tmp_path, defaults={"imgsz": [1280], "device": "cpu"}))
    specs = optimizer.expand_runs(cfg, SUPPORTED_SAM_MODELS)

    assert all(s.device == "cpu" for s in specs)


def test_an_explicit_name_wins_over_the_derived_one(tmp_path):
    cfg = optimizer.load_config(write_config(tmp_path, runs=[{"sam_models": ["sam_b"], "name": "baseline"}]))
    assert optimizer.expand_runs(cfg, SUPPORTED_SAM_MODELS)[0].name == "baseline"


def test_expand_runs_rejects_a_typo_in_a_run(tmp_path):
    cfg = optimizer.load_config(write_config(tmp_path, runs=[{"sam_models": ["sam_b"], "image_size": [640]}]))
    with pytest.raises(SystemExit, match="unknown key"):
        optimizer.expand_runs(cfg, SUPPORTED_SAM_MODELS)


def test_expand_runs_rejects_an_unsupported_model(tmp_path):
    cfg = optimizer.load_config(write_config(tmp_path, runs=[{"sam_models": ["sam_xl"]}]))
    with pytest.raises(SystemExit, match="unsupported model"):
        optimizer.expand_runs(cfg, SUPPORTED_SAM_MODELS)


def test_expand_runs_rejects_duplicate_names(tmp_path):
    """Two runs writing into one folder would silently overwrite each other's results."""
    cfg = optimizer.load_config(write_config(tmp_path, runs=[{"sam_models": ["sam_b"]}, {"sam_models": ["sam_b"]}]))
    with pytest.raises(SystemExit, match="duplicate run name"):
        optimizer.expand_runs(cfg, SUPPORTED_SAM_MODELS)


def test_a_run_needs_models(tmp_path):
    cfg = optimizer.load_config(write_config(tmp_path, runs=[{"imgsz": [640]}]))
    with pytest.raises(SystemExit, match="names no 'sam_models'"):
        optimizer.expand_runs(cfg, SUPPORTED_SAM_MODELS)


def test_name_is_not_accepted_as_a_default(tmp_path):
    cfg = optimizer.load_config(write_config(tmp_path, defaults={"name": "everything"}))
    with pytest.raises(SystemExit, match="belongs to a run"):
        optimizer.expand_runs(cfg, SUPPORTED_SAM_MODELS)


def test_describe_grid_counts_the_full_product():
    spec = optimizer.RunSpec(name="x", sam_models=["sam_b"], imgsz=[640, 1280], scale_factors=[0.0, 0.05, 0.1])
    assert len(spec.grid) == 6
    assert "= 6 points" in spec.describe_grid()

    single = optimizer.RunSpec(
        name="x", sam_models=["sam_b"], imgsz=[640], scale_factors=[0.05], opening_kernels=[0.15]
    )
    assert "= 1 point" in single.describe_grid()


# -------------------------------------------------------------------------------------- state
def test_is_complete_needs_a_full_grid(tmp_path):
    spec = optimizer.RunSpec(name="x", sam_models=["sam_b"], imgsz=[640], scale_factors=[0.0, 0.05])
    folder = tmp_path / "x"
    folder.mkdir()
    assert not optimizer.is_complete(folder, spec)

    (folder / optimizer.RESULTS_NAME).write_text(
        yaml.dump({"all_results": [grid_point(imgsz=640, sf=0.0)]}), encoding="utf-8"
    )
    assert not optimizer.is_complete(folder, spec)  # one point of two: interrupted, must re-run

    (folder / optimizer.RESULTS_NAME).write_text(
        yaml.dump({"all_results": [grid_point(imgsz=640, sf=0.0), grid_point(imgsz=640, sf=0.05)]}), encoding="utf-8"
    )
    assert optimizer.is_complete(folder, spec)


def test_a_run_measured_on_a_different_grid_is_not_complete(tmp_path):
    """
    An edited axis must re-run, not resume.

    Counting grid points would call a 45-point sweep over [1024, 1280, 1536] complete for a
    45-point sweep over [768, 1024, 1280], and the benchmark summary would then report numbers
    from image sizes its own config never names.
    """
    measured = optimizer.RunSpec(name="x", sam_models=["sam_b"], imgsz=[1024, 1536], scale_factors=[0.0])
    folder = tmp_path / "x"
    folder.mkdir()
    (folder / optimizer.RESULTS_NAME).write_text(
        yaml.dump({"all_results": [grid_point(imgsz=1024, sf=0.0), grid_point(imgsz=1536, sf=0.0)]}),
        encoding="utf-8",
    )
    assert optimizer.is_complete(folder, measured)

    retuned = optimizer.RunSpec(name="x", sam_models=["sam_b"], imgsz=[768, 1024], scale_factors=[0.0])
    assert len(retuned.grid) == len(measured.grid)
    assert not optimizer.is_complete(folder, retuned)


# ---------------------------------------------------------------------------------- artifacts
def test_write_run_writes_the_four_files(tmp_path):
    spec = optimizer.RunSpec(name="sam_b", sam_models=["sam_b"], imgsz=[1280], scale_factors=[0.04, 0.05])
    outcome = fake_outcome([grid_point(sf=0.04, iou=0.88), grid_point(sf=0.05, iou=0.90)])
    config = optimizer.run_config_dict(spec, tmp_path / "images", tmp_path / "gt", tmp_path / "hbb")

    optimizer.write_run(tmp_path / "sam_b", spec, outcome, config, plot=True)

    folder = tmp_path / "sam_b"
    for name in (optimizer.RUN_CONFIG_NAME, optimizer.RESULTS_NAME, optimizer.SUMMARY_NAME, optimizer.PLOT_NAME):
        assert (folder / name).is_file(), name
    assert not (folder / "config.yaml").exists()  # the old name is gone

    recorded = yaml.safe_load((folder / optimizer.RUN_CONFIG_NAME).read_text())
    assert recorded["run_name"] == "sam_b"
    assert recorded["sam_models"] == ["sam_b"]
    assert "system_metadata" in recorded

    results = yaml.safe_load((folder / optimizer.RESULTS_NAME).read_text())
    assert results["best_parameters"]["avg_iou"] == 0.90
    assert results["model_load_seconds"] == 1.0  # timed apart from the grid points


def test_collect_rows_ranks_by_iou_and_skips_unfinished_runs(tmp_path):
    for name, iou in (("sam_b", 0.88), ("sam_l", 0.91)):
        folder = tmp_path / name
        folder.mkdir()
        (folder / optimizer.RESULTS_NAME).write_text(
            yaml.dump({"best_parameters": grid_point(iou=iou), "all_results": [grid_point(iou=iou)]}), encoding="utf-8"
        )
        (folder / optimizer.RUN_CONFIG_NAME).write_text(yaml.dump({"sam_models": [name]}), encoding="utf-8")
    (tmp_path / "started").mkdir()  # no results.yaml yet

    rows = optimizer.collect_rows(tmp_path, ["sam_b", "sam_l", "started"])
    assert [r["name"] for r in rows] == ["sam_l", "sam_b"]
    assert rows[0]["sam_models"] == ["sam_l"]


def test_collect_rows_reads_a_folder_written_before_the_config_rename(tmp_path):
    folder = tmp_path / "sam_l-sam_b"
    folder.mkdir()
    (folder / optimizer.RESULTS_NAME).write_text(
        yaml.dump({"best_parameters": grid_point(), "all_results": [grid_point()]}), encoding="utf-8"
    )
    (folder / "config.yaml").write_text(yaml.dump({"sam_models": ["sam_l", "sam_b"]}), encoding="utf-8")

    assert optimizer.collect_rows(tmp_path, ["sam_l-sam_b"])[0]["sam_models"] == ["sam_l", "sam_b"]


def test_write_summary_reports_the_winner_and_embeds_the_plot(tmp_path):
    rows = [
        {"name": "sam_l", "sam_models": ["sam_l"], "n_points": 36, "sweep_seconds": 900, **grid_point(iou=0.91)},
        {"name": "sam_b", "sam_models": ["sam_b"], "n_points": 36, "sweep_seconds": 500, **grid_point(iou=0.88)},
    ]
    out = optimizer.write_summary(
        tmp_path,
        rows,
        tmp_path / "images",
        tmp_path / "hbb",
        tmp_path / "gt",
        "hbb2obb-optimize -c b.yaml",
        elapsed_seconds=3600,
        plot=True,
    )

    text = out.read_text()
    assert "**Best overall:** `sam_l`" in text
    assert "0.9100" in text
    assert "36 grid points per run, 72 conversions" in text
    assert f"]({optimizer.BENCHMARK_PLOT_NAME})" in text
    assert (tmp_path / optimizer.BENCHMARK_PLOT_NAME).is_file()
    # No provenance was written, so the summary must not send the reader to a file that is absent
    assert optimizer.BENCHMARK_PROVENANCE_NAME not in text


def test_write_summary_survives_an_empty_folder(tmp_path):
    out = optimizer.write_summary(tmp_path, [], tmp_path, tmp_path, tmp_path, "cmd")
    assert "No finished runs" in out.read_text()


# -------------------------------------------------------------------------------------- CLI
@pytest.fixture
def no_sweep(monkeypatch):
    """Replace the SAM passes with a deterministic result, and record which runs were asked for."""
    called = []

    def fake_sweep(spec, img_source, gt_dir, hbb_dir=None, no_bar=True, quiet=False):
        called.append(spec.name)
        return fake_outcome([grid_point(sf=sf) for sf in spec.scale_factors])

    monkeypatch.setattr(optimizer, "sweep", fake_sweep)
    monkeypatch.setattr("hbb2obb.converter.clear_model_cache", lambda: None)
    return called


def run_cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["hbb2obb-optimize", *argv])
    main_hbb2obb_optimize()


def test_dry_run_runs_nothing(monkeypatch, tmp_path, capsys, no_sweep):
    config = write_config(tmp_path)
    run_cli(monkeypatch, "-c", str(config), "--dry_run")

    assert no_sweep == []
    out = capsys.readouterr().out
    assert "Total          : 4 conversions" in out
    assert not (tmp_path / "bench" / optimizer.BENCHMARK_SUMMARY_NAME).exists()
    # It reports what would happen and writes nothing, the output folder included: a typo there
    # should not leave a stray empty directory behind.
    assert not (tmp_path / "bench").exists()


def test_a_config_run_writes_every_artifact(monkeypatch, tmp_path, no_sweep):
    config = write_config(tmp_path)
    run_cli(monkeypatch, "-c", str(config), "--no_plot")

    assert no_sweep == ["sam_b", "sam_l-sam_b"]
    bench = tmp_path / "bench"
    assert (bench / "sam_b" / optimizer.RESULTS_NAME).is_file()
    assert (bench / "sam_l-sam_b" / optimizer.RESULTS_NAME).is_file()
    assert (bench / optimizer.BENCHMARK_SUMMARY_NAME).is_file()
    assert (bench / optimizer.BENCHMARK_PROVENANCE_NAME).is_file()

    provenance = (bench / optimizer.BENCHMARK_PROVENANCE_NAME).read_text()
    assert "Benchmark configuration, verbatim" in provenance
    assert "sam_l" in provenance
    # The summary may now point at the provenance, because it exists
    assert optimizer.BENCHMARK_PROVENANCE_NAME in (bench / optimizer.BENCHMARK_SUMMARY_NAME).read_text()

    # The config travels with the numbers, so the folder re-runs without the repository
    copy = bench / config.name
    assert copy.is_file() and copy.read_text() == config.read_text()
    assert f"hbb2obb-optimize -c {copy}" in provenance
    assert config.name in (bench / optimizer.BENCHMARK_SUMMARY_NAME).read_text()


def test_cli_device_overrides_every_run(monkeypatch, tmp_path):
    seen = []

    def fake_sweep(spec, *_args, **_kwargs):
        seen.append((spec.name, spec.device))
        return fake_outcome([grid_point(sf=sf) for sf in spec.scale_factors])

    monkeypatch.setattr(optimizer, "sweep", fake_sweep)
    monkeypatch.setattr("hbb2obb.converter.clear_model_cache", lambda: None)

    config = write_config(tmp_path, defaults={"imgsz": [1280], "scale_factors": [0.05], "device": "0"})
    run_cli(monkeypatch, "-c", str(config), "--no_plot", "--device", "cpu")

    assert seen == [("sam_b", "cpu"), ("sam_l-sam_b", "cpu")]  # the CLI wins over the config's "0"
    assert "--device cpu" in (tmp_path / "bench" / optimizer.BENCHMARK_PROVENANCE_NAME).read_text()


def test_a_config_inside_the_output_folder_is_not_copied_onto_itself(monkeypatch, tmp_path, no_sweep):
    """The copy is a convenience, not a way to truncate the file the run is reading."""
    bench = tmp_path / "bench"
    bench.mkdir()
    config = write_config(tmp_path, path=bench / "benchmark.yaml")
    run_cli(monkeypatch, "-c", str(config), "--no_plot")

    assert config.is_file() and "sam_models" in config.read_text()


def test_refresh_leaves_the_provenance_of_the_run_that_happened(monkeypatch, tmp_path, no_sweep):
    """
    A refresh redraws; it does not re-measure.

    Rewriting the provenance here would stamp the current code state onto numbers produced by
    whatever the code was when the sweep ran, which is precisely the claim the file exists to
    make honestly.
    """
    config = write_config(tmp_path)
    run_cli(monkeypatch, "-c", str(config), "--no_plot")
    provenance_file = tmp_path / "bench" / optimizer.BENCHMARK_PROVENANCE_NAME
    before = provenance_file.read_text()

    run_cli(monkeypatch, "-c", str(config), "--refresh", "--no_plot")
    assert provenance_file.read_text() == before
    # but the summary still points the reader at the config copy already on disk
    assert config.name in (tmp_path / "bench" / optimizer.BENCHMARK_SUMMARY_NAME).read_text()


def test_only_restricts_the_runs(monkeypatch, tmp_path, no_sweep):
    config = write_config(tmp_path)
    run_cli(monkeypatch, "-c", str(config), "--only", "sam_b", "--no_plot")
    assert no_sweep == ["sam_b"]


def test_only_rejects_a_name_the_config_does_not_have(monkeypatch, tmp_path, no_sweep):
    config = write_config(tmp_path)
    with pytest.raises(SystemExit, match="not configured"):
        run_cli(monkeypatch, "-c", str(config), "--only", "sam_h", "--no_plot")


def test_resume_skips_a_finished_run(monkeypatch, tmp_path, no_sweep):
    config = write_config(tmp_path)
    run_cli(monkeypatch, "-c", str(config), "--no_plot")
    provenance_file = tmp_path / "bench" / optimizer.BENCHMARK_PROVENANCE_NAME
    before = provenance_file.read_text()
    no_sweep.clear()

    run_cli(monkeypatch, "-c", str(config), "--resume", "--no_plot")
    assert no_sweep == []  # both were already complete
    # and having measured nothing, it must not restamp the record with today's code state
    assert provenance_file.read_text() == before


def test_a_partial_resume_says_which_runs_it_did_not_measure(monkeypatch, tmp_path, no_sweep):
    config = write_config(tmp_path)
    run_cli(monkeypatch, "-c", str(config), "--no_plot")

    results = tmp_path / "bench" / "sam_b" / optimizer.RESULTS_NAME
    data = yaml.safe_load(results.read_text())
    data["all_results"] = data["all_results"][:1]
    results.write_text(yaml.dump(data), encoding="utf-8")

    run_cli(monkeypatch, "-c", str(config), "--resume", "--no_plot")
    text = (tmp_path / "bench" / optimizer.BENCHMARK_PROVENANCE_NAME).read_text()
    assert "Resumed benchmark" in text
    assert "measured : sam_b" in text
    assert "kept     : sam_l-sam_b" in text


def test_resume_re_runs_an_interrupted_run(monkeypatch, tmp_path, no_sweep):
    config = write_config(tmp_path)
    run_cli(monkeypatch, "-c", str(config), "--no_plot")

    # Truncate one run's grid, as an interrupted sweep would leave it
    results = tmp_path / "bench" / "sam_b" / optimizer.RESULTS_NAME
    data = yaml.safe_load(results.read_text())
    data["all_results"] = data["all_results"][:1]
    results.write_text(yaml.dump(data), encoding="utf-8")

    no_sweep.clear()
    run_cli(monkeypatch, "-c", str(config), "--resume", "--no_plot")
    assert no_sweep == ["sam_b"]


def test_refresh_rebuilds_the_summary_without_sweeping(monkeypatch, tmp_path, no_sweep):
    config = write_config(tmp_path)
    run_cli(monkeypatch, "-c", str(config), "--no_plot")
    (tmp_path / "bench" / optimizer.BENCHMARK_SUMMARY_NAME).unlink()
    no_sweep.clear()

    run_cli(monkeypatch, "-c", str(config), "--refresh")
    assert no_sweep == []
    assert (tmp_path / "bench" / optimizer.BENCHMARK_SUMMARY_NAME).is_file()
    assert (tmp_path / "bench" / "sam_b" / optimizer.PLOT_NAME).is_file()  # rendered on the refresh


def test_a_single_sweep_needs_both_positionals(monkeypatch, no_sweep):
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, "data/images", "--no_plot")


def test_a_single_sweep_writes_one_folder(monkeypatch, tmp_path, no_sweep, images_dir, gt_dir):
    run_cli(
        monkeypatch,
        str(images_dir),
        str(gt_dir),
        "-sm",
        "sam_b",
        "-iz",
        "1280",
        "-sf",
        "0.05",
        "-o",
        str(tmp_path / "bench"),
        "--no_plot",
    )

    assert no_sweep == ["sam_b"]
    bench = tmp_path / "bench"
    assert (bench / "sam_b" / optimizer.RESULTS_NAME).is_file()
    assert (bench / optimizer.BENCHMARK_SUMMARY_NAME).is_file()
    # Every sweep records itself, config or not: a set of numbers with no record of the code
    # and checkpoints behind it is the thing this file exists to prevent
    provenance = (bench / optimizer.BENCHMARK_PROVENANCE_NAME).read_text()
    assert "--scale_factors 0.05" in provenance
    assert "source sha256" in provenance
    # but with no config there is nothing to copy beside the results
    assert not (bench / "bench.yaml").exists()

    summary = (bench / optimizer.BENCHMARK_SUMMARY_NAME).read_text()
    assert "--scale_factors 0.05" in summary  # the command it records reproduces the grid


def test_the_summary_covers_runs_this_invocation_did_not_touch(monkeypatch, tmp_path, no_sweep):
    """A resumed benchmark must still report the whole folder, not just what it re-ran."""
    config = write_config(tmp_path)
    run_cli(monkeypatch, "-c", str(config), "--no_plot")
    run_cli(monkeypatch, "-c", str(config), "--only", "sam_b", "--no_plot")

    summary = (tmp_path / "bench" / optimizer.BENCHMARK_SUMMARY_NAME).read_text()
    assert "sam_l sam_b" in summary
    assert "2 run(s)" in summary


# --------------------------------------------------------------------------- the plot metric
def test_an_unknown_plot_metric_stops_before_any_sweep_runs(monkeypatch, tmp_path):
    """
    A typo must not surface hours in, with the grid already measured and the plot the last step.
    """
    swept = []
    monkeypatch.setattr(optimizer, "sweep", lambda spec, *a, **k: swept.append(spec.name))
    monkeypatch.setattr("hbb2obb.converter.clear_model_cache", lambda: None)

    config = write_config(tmp_path)
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, "-c", str(config), "--plot_metric", "orientaton")

    assert swept == []
    assert not (tmp_path / "bench").exists()


def test_a_typo_in_the_plot_metric_names_the_real_ones(monkeypatch, tmp_path):
    config = write_config(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        run_cli(monkeypatch, "-c", str(config), "--plot_metric", "orientaton")

    message = str(excinfo.value)
    assert "unknown plot metric" in message
    assert "median_angle_error" in message


def test_refreshing_on_a_metric_a_run_never_recorded_says_so(monkeypatch, tmp_path, capsys):
    """
    A benchmark measured before a metric existed has nothing to draw for it. That is a fact about
    those runs, not a failure of the refresh: it must report them and still write the summary.
    """
    bench = tmp_path / "bench"
    (bench / "sam_b").mkdir(parents=True)
    points = [{k: v for k, v in grid_point(sf=sf).items() if "angle" not in k} for sf in (0.04, 0.05)]
    for point in points:
        point.pop("iou_fractions")
    with open(bench / "sam_b" / optimizer.RESULTS_NAME, "w", encoding="utf-8") as f:
        yaml.dump({"all_results": points, "best_parameters": points[-1]}, f)
    with open(bench / "sam_b" / optimizer.RUN_CONFIG_NAME, "w", encoding="utf-8") as f:
        yaml.dump({"run_name": "sam_b", "sam_models": ["sam_b"]}, f)

    config = write_config(tmp_path)
    run_cli(monkeypatch, "-c", str(config), "--refresh", "--plot_metric", "median_angle_error")

    assert "Skipping sam_b" in capsys.readouterr().out
    # A metric other than the ranking one writes its own summary, leaving summary.md alone
    assert not (bench / optimizer.BENCHMARK_SUMMARY_NAME).exists()
    summary = (bench / "summary_median_angle_error.md").read_text(encoding="utf-8")
    assert "No accuracy-against-compute figure" in summary
    # The table is still written, with the columns it cannot fill left empty rather than faked
    assert "| Models | Image size" in summary
    assert "| - | - |" in summary


def test_a_metric_never_overwrites_the_figures_already_there(monkeypatch, tmp_path, no_sweep):
    """
    Drawing a second metric must add to the folder, never replace what is in it.

    The whole reason to draw one is to compare it against the ranking metric, which is impossible
    if rendering it destroys the thing being compared against.
    """
    config = write_config(tmp_path)
    run_cli(monkeypatch, "-c", str(config))
    bench = tmp_path / "bench"
    default = (bench / "sam_b" / optimizer.PLOT_NAME).read_bytes()

    run_cli(monkeypatch, "-c", str(config), "--refresh", "--plot_metric", "median_angle_error")

    assert (bench / "sam_b" / optimizer.PLOT_NAME).read_bytes() == default
    assert (bench / "sam_b" / "plot_median_angle_error.png").is_file()
    assert (bench / optimizer.BENCHMARK_PLOT_NAME).is_file()
    assert (bench / "comparison_median_angle_error.png").is_file()
    assert (bench / optimizer.BENCHMARK_SUMMARY_NAME).is_file()
    assert (bench / "summary_median_angle_error.md").is_file()


def test_a_sweep_draws_the_other_metrics_without_being_asked(monkeypatch, tmp_path, no_sweep):
    """
    They cost no SAM time and are already recorded, so nobody should have to remember a second
    command to get them.
    """
    config = write_config(tmp_path)
    run_cli(monkeypatch, "-c", str(config))

    bench = tmp_path / "bench"
    for metric in optimizer.AUTO_PLOT_METRICS:
        assert (bench / "sam_b" / f"plot_{metric}.png").is_file(), metric
        assert (bench / f"comparison_{metric}.png").is_file(), metric
        assert (bench / f"summary_{metric}.md").is_file(), metric


def test_asking_for_one_metric_does_not_also_draw_the_rest(monkeypatch, tmp_path, no_sweep):
    """An explicit --plot_metric is a request for that one, not a request for everything."""
    config = write_config(tmp_path)
    run_cli(monkeypatch, "-c", str(config), "--plot_metric", "median_angle_error")

    bench = tmp_path / "bench"
    assert (bench / "sam_b" / "plot_median_angle_error.png").is_file()
    assert not (bench / "sam_b" / "plot_iou_at_90.png").exists()


def test_refresh_needs_no_images_labels_or_checkpoints(monkeypatch, tmp_path, capsys, no_sweep):
    """
    A finished results folder is usually read somewhere else: copied off the machine that measured
    it, without the imagery. Re-rendering reads results.yaml and nothing else, so it must work
    there rather than exiting over inputs it never opens.
    """
    bench = tmp_path / "bench"
    (bench / "sam_b").mkdir(parents=True)
    points = [grid_point(sf=sf) for sf in (0.04, 0.05)]
    with open(bench / "sam_b" / optimizer.RESULTS_NAME, "w", encoding="utf-8") as f:
        yaml.dump({"all_results": points, "best_parameters": points[-1]}, f)
    with open(bench / "sam_b" / optimizer.RUN_CONFIG_NAME, "w", encoding="utf-8") as f:
        yaml.dump({"run_name": "sam_b", "sam_models": ["sam_b"]}, f)

    # Nothing but the results folder exists: no images, no labels, no checkpoints
    config = write_config(tmp_path, img_source=str(tmp_path / "gone"), gt_dir=str(tmp_path / "gone_too"))
    capsys.readouterr()

    run_cli(monkeypatch, "-c", str(config), "--refresh")

    assert "Wrote" in capsys.readouterr().out
    assert (bench / optimizer.BENCHMARK_SUMMARY_NAME).is_file()
