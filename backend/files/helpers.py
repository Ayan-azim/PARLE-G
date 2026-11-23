from schema import (Attribute, Attributes, CharReading, FrameReading, Settings)
from matching.registry import RegistryEntry


def frame(text, conf=0.9, weight=None, idx=0, n_frames=1):
    confs = conf if isinstance(conf, list) else [conf] * len(text)
    return FrameReading(frame_idx=idx, timestamp_s=idx * 0.2, sharpness=100.0,
                        quality_weight=weight if weight is not None else 1.0 / n_frames, used=True,
                        raw_text=text, chars=[CharReading(char=c, conf=p) for c, p in zip(text, confs)])


def attrs(colour="white", typ="sedan"):
    return Attributes(colour=Attribute(value=colour, confidence=0.8), body_type=Attribute(value=typ, confidence=0.8))


def entry(plate, colour="white", typ="sedan"):
    return RegistryEntry(plate, colour, typ, "Honda", "City")


S = Settings()


def make_obs(clip, track, text, conf=0.9, colour="white", typ="sedan", idx0=0, n=3):
    from schema import VisionOutput
    from matching.cross_clip import make_observation
    frames = [frame(text, conf, idx=idx0 + i, n_frames=n) for i in range(n)]
    v = VisionOutput(clip_name=clip, duration_s=5.0, fps=30.0, frame_size=(640, 360), sampled_fps=5.0,
                     track_id=track, target_selection_reason="test", vehicle_class="car", frames=frames)
    return make_observation(v, ".", S, attrs(colour, typ))
