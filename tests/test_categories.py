import pipeline.categories as cat


def test_counts():
    assert len(cat.track_categories()) == 12
    assert len(cat.image_categories()) == 18
    assert len(cat.fusion_categories()) == 18


def test_track_types_by_tier():
    by_name = {c["name"]: c for c in cat.track_categories()}
    assert by_name["aod_eb_all"]["track_types"] == "GenTrk"
    assert by_name["mini_eb_all"]["track_types"] == "Lost,PF,GSF"


def test_image_argv_has_flags():
    by_name = {c["name"]: c for c in cat.image_categories()}
    c = by_name["aod_ee_noes_eq1"]
    argv = c["argv"]
    assert "train_resnet_image_classifier.py" in argv[0]
    assert "--no-es" in argv
    assert "--track-types" in argv and "GenTrk" in argv
    assert "--detector" in argv and "ee" in argv
    i = argv.index("--weight-h5")
    assert argv[i + 1].endswith("AOD_eq1/category_weights.h5")
    assert "--no-es" not in by_name["aod_ee_es_eq1"]["argv"]


def test_fusion_points_to_shared_track():
    by_name = {c["name"]: c for c in cat.fusion_categories()}
    es = by_name["mini_ee_es_all"]
    noes = by_name["mini_ee_noes_all"]
    assert es["track_run"] == noes["track_run"]
    assert es["track_run"].endswith("track_mini_ee_all")
    assert es["image_run"].endswith("image_mini_ee_es_all")


import pipeline.make_submit as ms


def test_arglist_lines_roundtrip(tmp_path):
    cats = cat.track_categories()
    listfile = tmp_path / "track.txt"
    ms.write_arglist(listfile, cats)
    lines = [ln for ln in listfile.read_text().splitlines() if ln.strip()]
    assert len(lines) == 12
    assert any("train_point_transformer_track_classifier.py --detector eb" in ln for ln in lines)
    assert any("--track-types Lost,PF,GSF" in ln for ln in lines)


def test_submit_file_has_gpu_and_transfers(tmp_path):
    sub = tmp_path / "train_track.sub"
    ms.write_submit(sub, "train_track", tmp_path / "track.txt", request_gpus=1)
    text = sub.read_text()
    assert "request_gpus   = 1" in text
    assert "require_gpus   = (Capability >= 7.5)" in text
    assert "queue arguments from" in text
    for mod in ("train_resnet_image_classifier.py", "track_point_transformer.py",
                "train_point_transformer_track_classifier.py",
                "train_fusion_ensemble.py", "build_event_level_weights.py"):
        assert mod in text


import json
import pipeline.collect_results as cr


def test_collect_reads_fusion_metrics(tmp_path):
    runs = tmp_path / "runs"
    (runs / "fusion_aod_eb_all").mkdir(parents=True)
    (runs / "fusion_aod_eb_all" / "test_metrics.json").write_text(json.dumps({
        "test_auc": 0.91, "test_f1": 0.8,
        "component_metrics": {"image_test_auc": 0.85, "track_test_auc": 0.7,
                              "fusion_test_auc": 0.91},
    }))
    rows = cr.collect(runs_root=runs)
    assert rows and rows[0]["name"] == "aod_eb_all"
    assert rows[0]["fusion_test_auc"] == 0.91
    assert rows[0]["image_test_auc"] == 0.85
    assert rows[0]["track_test_auc"] == 0.7
