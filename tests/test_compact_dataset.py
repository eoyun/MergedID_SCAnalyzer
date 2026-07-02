import numpy as np

import build_compact_dataset as bcd


def test_calo_key_and_channel_source_keys():
    assert bcd.calo_key("eb") == "SC_energy"
    assert bcd.calo_key("ee") == "EE_seed_energy"
    eb = bcd.raw_source_keys("eb", ("Lost", "PF", "GSF"))
    assert "SC_energy" in eb["calo"]
    assert eb["track"] == ["EB_track_pt_Lost", "EB_track_pt_PF", "EB_track_pt_GSF"]
    assert eb["es"] == []
    ee = bcd.raw_source_keys("ee", ("GenTrk",))
    assert ee["calo"] == ["EE_seed_energy"]
    assert ee["es"] == ["ES_seed_plane1_energy", "ES_seed_plane2_energy"]
    assert ee["track"] == ["EE_track_pt_GenTrk"]
