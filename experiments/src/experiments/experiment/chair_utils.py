from __future__ import annotations

import re

COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

SYNONYMS: dict[str, list[str]] = {
    "person": ["people", "man", "men", "woman", "women", "girl", "boy", "kid", "kids",
               "child", "children", "baby", "guy", "pedestrian", "skier", "rider", "surfer"],
    "bicycle": ["bike", "bicycles", "bicyclist"],
    "car": ["cars", "automobile", "vehicle", "vehicles", "suv", "taxi"],
    "motorcycle": ["motorbike", "motorcyclist"],
    "airplane": ["plane", "aeroplane"],
    "bus": ["buses"],
    "train": ["trains"],
    "truck": ["trucks", "lorry"],
    "boat": ["boats", "ship", "sailboat"],
    "traffic light": ["traffic lights"],
    "fire hydrant": ["hydrant"],
    "stop sign": ["stop signs"],
    "parking meter": ["parking meters"],
    "bench": ["benches"],
    "bird": ["birds"],
    "cat": ["cats"],
    "dog": ["dogs", "puppy", "puppies"],
    "horse": ["horses"],
    "sheep": ["sheep"],
    "cow": ["cows", "cattle"],
    "elephant": ["elephants"],
    "bear": ["bears"],
    "zebra": ["zebras"],
    "giraffe": ["giraffes"],
    "backpack": ["backpacks", "back pack"],
    "umbrella": ["umbrellas"],
    "handbag": ["handbags", "purse"],
    "tie": ["ties"],
    "suitcase": ["suitcases", "luggage"],
    "frisbee": ["frisbees"],
    "skis": ["ski"],
    "snowboard": ["snowboards"],
    "sports ball": ["ball", "balls", "baseball", "basketball", "soccer ball", "football",
                    "tennis ball", "volleyball"],
    "kite": ["kites"],
    "baseball bat": ["baseball bats"],
    "baseball glove": ["glove", "gloves"],
    "skateboard": ["skateboards"],
    "surfboard": ["surfboards"],
    "tennis racket": ["racket", "racquet"],
    "bottle": ["bottles"],
    "wine glass": ["wine glasses", "wineglass"],
    "cup": ["cups"],
    "fork": ["forks"],
    "knife": ["knives"],
    "spoon": ["spoons"],
    "bowl": ["bowls"],
    "banana": ["bananas"],
    "apple": ["apples"],
    "sandwich": ["sandwiches"],
    "orange": ["oranges"],
    "broccoli": ["broccoli"],
    "carrot": ["carrots"],
    "hot dog": ["hot dogs", "hotdog"],
    "pizza": ["pizzas"],
    "donut": ["donuts", "doughnut"],
    "cake": ["cakes"],
    "chair": ["chairs"],
    "couch": ["couches", "sofa", "sofas"],
    "potted plant": ["potted plants"],
    "bed": ["beds"],
    "dining table": ["dining tables", "table", "tables"],
    "toilet": ["toilets"],
    "tv": ["tvs", "television", "monitor", "monitors", "screen", "screens"],
    "laptop": ["laptops"],
    "mouse": ["mice"],
    "remote": ["remotes", "remote control"],
    "keyboard": ["keyboards"],
    "cell phone": ["cellphone", "phone", "phones", "mobile phone"],
    "microwave": ["microwave oven"],
    "oven": ["ovens"],
    "toaster": ["toasters"],
    "sink": ["sinks"],
    "refrigerator": ["fridge", "refrigerators", "fridges"],
    "book": ["books"],
    "clock": ["clocks", "watch"],
    "vase": ["vases"],
    "scissors": ["scissors"],
    "teddy bear": ["teddy bears"],
    "hair drier": ["hair dryer", "hairdryer"],
    "toothbrush": ["toothbrushes"],
}

_PUNCT = re.compile(r"[^\w\s]|_", re.UNICODE)


def _build_forms() -> dict[str, str]:
    forms: dict[str, str] = {}
    for cls in COCO_CLASSES:
        forms[cls] = cls
        for syn in SYNONYMS.get(cls, []):
            forms[syn] = cls
    return forms


FORMS = _build_forms()
_PATTERN = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(f) for f in sorted(FORMS, key=len, reverse=True)) + r")(?!\w)",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    return _PUNCT.sub(" ", text).lower()


def extract_mentions(text: str) -> set[str]:
    matches = {m.group(1).lower() for m in _PATTERN.finditer(text)}
    return {FORMS[m] for m in matches if m in FORMS}


def extract_mentions_with_spans(text: str) -> list[tuple[str, int, int]]:
    out = []
    for m in _PATTERN.finditer(text):
        form = m.group(1).lower()
        if form in FORMS:
            out.append((FORMS[form], m.start(), m.end()))
    return out