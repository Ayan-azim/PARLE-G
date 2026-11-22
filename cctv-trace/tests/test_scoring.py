from matching.fusion import fuse_plate
from matching.scoring import compare_plate, match
from tests.helpers import S, attrs, entry, frame


def fused(text, conf=0.9):
    return fuse_plate([frame(text, conf, idx=i, n_frames=3) for i in range(3)], S)


def test_confusable_beats_mismatch_but_not_exact():
    f = fused("CG04AB1234")
    exact, _ = compare_plate(f, "CG04AB1234")
    conf, cm = compare_plate(f, "CG04A81234")     # B <-> 8
    bad, _ = compare_plate(f, "CG04AC1234")
    assert exact > conf > bad
    assert cm[5].status == "confusable"


def test_clear_match():
    reg = [entry("CG04AB1234"), entry("MH12XY9999", "red", "SUV"), entry("DL01ZZ0001", "black", "bus")]
    v, cands, weak = match(fused("CG04AB1234"), attrs(), reg, S)
    assert v.status == "match" and cands[0].plate == "CG04AB1234" and not weak


def test_all_unknown_is_no_confident_match():
    reg = [entry("CG04AB1234")]
    v, cands, weak = match(fused("CG04AB1234", 0.1), attrs(), reg, S)
    assert v.status == "no_confident_match" and cands == [] and weak


def test_near_duplicates_give_low_margin():
    reg = [entry("CG04AB1234"), entry("CG04AB1284")]
    f = fuse_plate([frame("CG04AB1234", 0.9, idx=0, n_frames=2), frame("CG04AB1284", 0.9, idx=1, n_frames=2)], S)
    v, cands, weak = match(f, attrs(), reg, S)
    assert f.pattern == "CG04AB12?4"
    assert v.status == "no_confident_match"
    assert any(c.name == "margin over runner-up" and not c.passed for c in v.rule_checks)
