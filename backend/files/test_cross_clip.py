from matching.cross_clip import build_identities, compare, plate_agreement
from matching.fusion import fuse_plate
from matching.scoring import type_score
from tests.helpers import S, entry, frame, make_obs

REG = [entry("CG04AB1234"), entry("MH12XY9999", "red", "SUV")]
PLATE = "CG04AB1234"


def test_same_plate_in_two_clips_is_likely_same():
    l = compare(make_obs("c1.mp4", 1, PLATE), make_obs("c2.mp4", 1, PLATE))
    assert l.verdict == "likely_same" and l.overlap == 10


def test_contradicting_plates_are_different():
    l = compare(make_obs("c1.mp4", 1, PLATE), make_obs("c2.mp4", 1, "CG04AB9999"))
    assert l.verdict == "different"


def test_single_character_difference_is_only_possible_and_not_merged():
    a, b = make_obs("c1.mp4", 1, PLATE), make_obs("c2.mp4", 1, "CG04AB1284")
    assert compare(a, b).verdict == "possible_same"
    res = build_identities([a, b], REG, S)
    assert len(res.identities) == 2 and len(res.possible_links) == 1


def test_unreadable_plates_never_link_on_colour_and_type():
    a, b = make_obs("c1.mp4", 1, PLATE, conf=0.1), make_obs("c2.mp4", 1, PLATE, conf=0.1)
    assert compare(a, b).verdict == "insufficient_evidence"
    assert len(build_identities([a, b], REG, S).identities) == 2


def test_two_tracks_visible_together_in_one_clip_are_different():
    a = make_obs("c1.mp4", 1, PLATE, idx0=0, n=5)
    b = make_obs("c1.mp4", 2, PLATE, idx0=1, n=5)
    assert compare(a, b).verdict == "different"


def test_motorcycle_and_car_are_different():
    l = compare(make_obs("c1.mp4", 1, PLATE, typ="motorcycle"), make_obs("c2.mp4", 1, PLATE, typ="sedan"))
    assert l.verdict == "different"


def test_pooling_recovers_a_plate_neither_clip_read_alone():
    a = make_obs("c1.mp4", 1, PLATE, conf=[0.9] * 8 + [0.2] * 2)      # last two chars unreadable
    b = make_obs("c2.mp4", 1, PLATE, conf=[0.2] * 2 + [0.9] * 8)      # first two chars unreadable
    assert a.fusion.known_count == 8 and b.fusion.known_count == 8
    res = build_identities([a, b], REG, S)
    assert len(res.identities) == 1
    ident = res.identities[0]
    assert ident.fusion.pattern == PLATE and ident.fusion.known_count == 10
    assert ident.verdict.status == "match"


def test_contradiction_blocks_merge_through_a_third_sighting():
    a = make_obs("c1.mp4", 1, "CG04AB1234", conf=[0.9] * 7 + [0.2] * 3)    # knows CG04AB1, not the end
    c = make_obs("c3.mp4", 1, "CG04AB9999", conf=[0.2] * 4 + [0.9] * 6)    # knows AB9999 part
    b = make_obs("c2.mp4", 1, "CG04AB1234", conf=[0.9] * 10)               # links to a, contradicts c
    res = build_identities([a, c, b], REG, S)
    ids = [set(i.observation_ids) for i in res.identities]
    assert not any({a.obs_id, c.obs_id} <= s or {b.obs_id, c.obs_id} <= s for s in ids)


def test_plate_agreement_counts_confusable_pairs():
    fa = fuse_plate([frame("CG04AB1234", 0.9)], S)
    fb = fuse_plate([frame("CG04A81234", 0.9)], S)           # B <-> 8
    ag = plate_agreement(fa, fb)
    assert ag["confusable"] == 1 and ag["mismatch"] == 0


def test_registry_type_labels_two_wheeler_equals_motorcycle():
    assert type_score("motorcycle", "two-wheeler") == 1.0
    assert type_score("motorcycle", "sedan") == 0.0
