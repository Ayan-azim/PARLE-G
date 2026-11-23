import cv2
import numpy as np
import pytest

from vision import colour as C

RNG = np.random.default_rng(7)
CARS = {
    "white": (238, 238, 240), "silver": (172, 174, 178), "grey": (105, 108, 112), "black": (28, 28, 30),
    "red": (190, 28, 36), "blue": (30, 62, 150), "green": (42, 120, 62), "yellow": (232, 196, 40),
}


def make_car(rgb, h=220, w=340, plate_rgb=(235, 235, 235)):
    img = np.full((h, w, 3), (115, 118, 120), np.float32)                          # road / background
    img[int(h * .10):int(h * .92), int(w * .06):int(w * .94)] = np.array(rgb[::-1], np.float32)
    img[int(h * .14):int(h * .40), int(w * .18):int(w * .82)] = (22, 26, 30)       # windows
    img[int(h * .86):int(h * .98), int(w * .10):int(w * .30)] = (15, 15, 15)       # tyres
    img[int(h * .86):int(h * .98), int(w * .70):int(w * .90)] = (15, 15, 15)
    box = (int(.40 * w), int(.64 * h), int(.60 * w), int(.74 * h))
    img[box[1]:box[3], box[0]:box[2]] = np.array(plate_rgb[::-1], np.float32)      # plate
    img += RNG.normal(0, 7, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), box


@pytest.mark.parametrize("name", list(CARS))
def test_plain_colours_are_recognised(name):
    img, box = make_car(CARS[name])
    e = C.estimate_colour(img, "car", box)
    assert e.value == name
    assert e.reliability > 0.5


@pytest.mark.parametrize("name", ["red", "blue", "green", "black", "white"])
def test_dim_and_warm_footage_with_plate_reference(name):
    img, box = make_car(CARS[name])
    dim = np.clip(img * 0.7, 0, 255).astype(np.uint8)
    warm = np.clip(img.astype(np.float32) * np.array([0.85, 1.0, 1.15]), 0, 255).astype(np.uint8)
    assert C.estimate_colour(dim, "car", box, use_plate_reference=True).value == name
    assert C.estimate_colour(warm, "car", box, use_plate_reference=True).value == name


def test_red_car_is_not_neutralised():
    img, box = make_car(CARS["red"])
    assert C.estimate_colour(img, "car", box).value == "red"


def test_yellow_plate_is_not_used_as_white_reference():
    img, box = make_car(CARS["white"], plate_rgb=(235, 200, 40))
    e = C.estimate_colour(img, "car", box, use_plate_reference=True)
    assert e.value == "white"
    assert "plate used as exposure/cast reference" not in e.notes


def test_small_or_empty_crops_give_unknown_not_a_guess():
    assert C.estimate_colour(np.zeros((20, 20, 3), np.uint8)).value == "unknown"


def test_two_wheeler_is_less_reliable_than_car():
    img, _ = make_car(CARS["black"])
    assert C.estimate_colour(img, "two-wheeler").reliability < C.estimate_colour(img, "car").reliability


def test_aggregate_and_attribute_dict():
    imgs = [make_car(CARS["blue"]) for _ in range(4)]
    ests = [C.estimate_colour(i, "car", b) for i, b in imgs]
    agg = C.aggregate_estimates([(e, 90000.0, 1.0) for e in ests])
    d = C.to_attribute_dict(agg)
    assert agg.value == "blue" and d["value"] == "blue" and 0.0 <= d["confidence"] <= 1.0


def test_similarity():
    assert C.colour_similarity("white", "white") == 1.0
    assert C.colour_similarity("white", "silver") == 0.5
    assert C.colour_similarity("white", "red") == 0.0