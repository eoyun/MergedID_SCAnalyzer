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
