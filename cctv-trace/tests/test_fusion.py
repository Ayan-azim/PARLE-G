from matching.fusion import fuse_plate
from tests.helpers import S, frame


def test_all_frames_agree():
    f = fuse_plate([frame("CG04AB1234", 0.9, idx=i, n_frames=3) for i in range(3)], S)
    assert f.pattern == "CG04AB1234" and f.known_count == 10


def test_conflicting_frames_give_unknown():
    a = frame("CG04AB1234", 0.9, idx=0, n_frames=2)
    b = frame("CG04AB1284", 0.9, idx=1, n_frames=2)
    f = fuse_plate([a, b], S)
    assert f.pattern == "CG04AB12?4"
    assert f.chars[8].top_candidate in ("3", "8") and f.chars[8].status == "unknown"


def test_low_confidence_is_never_guessed():
    f = fuse_plate([frame("CG04AB1234", 0.1, idx=i, n_frames=3) for i in range(3)], S)
    assert f.pattern == "?" * 10 and f.known_count == 0


def test_dropped_char_is_aligned_not_lost():
    frames = [frame("CG04AB1234", 0.9, idx=0, weight=0.4), frame("CG04AB1234", 0.9, idx=1, weight=0.4),
              frame("CG04AB234", 0.9, idx=2, weight=0.2)]
    f = fuse_plate(frames, S)
    assert f.pattern == "CG04AB1234" and f.frames_used == [0, 1, 2]


def test_ind_prefix_is_trimmed():
    frames = [frame("INDCG04AB1234", 0.9, idx=0, weight=0.4), frame("CG04AB1234", 0.9, idx=1, weight=0.3),
              frame("INDCG04AB1234", 0.9, idx=2, weight=0.3)]
    f = fuse_plate(frames, S)
    assert f.pattern == "CG04AB1234" and f.length == 10


def test_flexible_format_dl1cs2343():
    f = fuse_plate([frame("DL1CS2343", 0.9, idx=i, n_frames=3) for i in range(3)], S)
    assert f.pattern == "DL1CS2343" and f.known_count == 9


def test_garbage_frame_rejected():
    frames = [frame("CG04AB1234", 0.9, idx=0, weight=0.4), frame("CG04AB1234", 0.9, idx=1, weight=0.4),
              frame("XXXXXXXXXX", 0.9, idx=2, weight=0.2)]
    f = fuse_plate(frames, S)
    assert f.frames_used == [0, 1] and f.pattern == "CG04AB1234"


def test_no_usable_frames():
    assert fuse_plate([], S).length == 0
