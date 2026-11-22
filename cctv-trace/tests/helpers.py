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
