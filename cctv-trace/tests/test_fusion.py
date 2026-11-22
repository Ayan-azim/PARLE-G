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


def test_length_outlier_dropped():
    frames = [frame("CG04AB1234", 0.9, idx=0, weight=0.4), frame("CG04AB1234", 0.9, idx=1, weight=0.4),
              frame("CG04AB123", 0.9, idx=2, weight=0.2)]
    f = fuse_plate(frames, S)
    assert f.length == 10 and f.frames_used == [0, 1]


def test_no_usable_frames():
    assert fuse_plate([], S).length == 0
