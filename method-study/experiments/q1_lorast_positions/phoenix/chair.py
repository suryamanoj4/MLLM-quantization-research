"""CHAIR (Caption Hallucination Assessment with Image Relevance), Rohrbach et al. 2018.

CHAIR_i = hallucinated object *mentions* / all object mentions
CHAIR_s = captions containing >= 1 hallucinated object / all captions

Ground truth for an image is the union of (a) the 80-way instance annotations and
(b) objects named in the five human reference captions -- (b) matters because the
segmentation annotations miss objects that are plainly visible and it is the
convention in the original implementation.

The synonym table below follows the canonical `synonyms.txt`. For camera-ready
numbers, drop the official file in and pass `--synonyms path/to/synonyms.txt`;
`load_synonyms` will use it instead.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

SYNONYMS_RAW = """
person, girl, boy, man, woman, kid, child, chef, baker, people, adult, rider, children, baby, worker, passenger, sister, biker, policeman, cop, officer, lady, cowboy, bride, groom, male, female, guy, traveler, mother, father, gentleman, pitcher, player, skier, snowboarder, skater, skateboarder, foreigner, caller, offender, coworker, trespasser, patient, politician, soldier, grandchild, serviceman, walker, drinker, doctor, lawyer, judge, teacher, student, camper, driver, hunter, shopper, villager
bicycle, bike, unicycle, minibike, trike
car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi
motorcycle, scooter, moped, motorbike
airplane, jetliner, plane, air plane, monoplane, aircraft, jet, airbus, biplane, seaplane
bus, minibus, trolley
train, locomotive, tramway, caboose
truck, pickup, lorry, hauler, firetruck
boat, ship, liner, sailboat, motorboat, dinghy, powerboat, speedboat, canoe, skiff, yacht, kayak, catamaran, pontoon, houseboat, vessel, rowboat, trawler, ferryboat, watercraft, tugboat, schooner, barge, ferry, sailboard, paddleboat, lifeboat, freighter, steamboat, riverboat, gondola, raft, dock, marina
traffic light, street light, traffic signal, stop light, streetlight, stoplight
fire hydrant, hydrant
stop sign
parking meter
bench, pew
bird, ostrich, owl, seagull, goose, duck, parakeet, falcon, robin, pelican, waterfowl, heron, hummingbird, mallard, finch, pigeon, sparrow, seabird, osprey, blackbird, fowl, shorebird, woodpecker, egret, chickadee, quail, bluebird, kingfisher, buzzard, willet, gull, swan, bluejay, flamingo, cormorant, parrot, loon, gosling, waterbird, pheasant, rooster, sandpiper, crow, raven, turkey, oriole, cowbird, warbler, magpie, peacock, cockatiel, lorikeet, puffin, vulture, condor, macaw, peafowl, eagle, bald eagle, lark, nightingale, hen, dove, albatross
cat, kitten, feline, tabby
dog, puppy, beagle, pup, chihuahua, schnauzer, dachshund, rottweiler, canine, pitbull, collie, pug, terrier, poodle, labrador, doggie, doberman, mutt, doggy, spaniel, bulldog, sheepdog, weimaraner, corgi, cocker spaniel, greyhound, retriever, brindle, hound, whippet, husky
horse, colt, pony, racehorse, stallion, equine, mare, foal, palomino, mustang, clydesdale, bronc, bronco
sheep, lamb, ram, goat, ewe
cow, cattle, oxen, ox, calf, holstein, heifer, buffalo, bull, zebu, bison
elephant
bear, panda
zebra
giraffe
backpack, knapsack
umbrella
handbag, wallet, purse, briefcase
tie, bow, bow tie
suitcase, suit case, luggage
frisbee
skis, ski
snowboard
sports ball, ball
kite
baseball bat
baseball glove
skateboard
surfboard, longboard, skimboard, shortboard, wakeboard
tennis racket, racket
bottle
wine glass
cup
fork
knife, pocketknife, knive
spoon
bowl, container
banana
apple
sandwich, burger, sub, cheeseburger, hamburger
orange
broccoli
carrot
hot dog
pizza
donut, doughnut, bagel
cake, cheesecake, cupcake, shortcake, coffeecake, pancake
chair, seat, stool
couch, sofa, recliner, futon, loveseat, settee, chesterfield
potted plant, houseplant
bed
dining table, table, desk
toilet, urinal, commode, lavatory, potty
tv, monitor, televison, television
laptop, computer, notebook, netbook, macbook, laptop computer
mouse
remote
keyboard
cell phone, mobile phone, phone, cellphone, telephone, phon, smartphone, iPhone
microwave
oven, stovetop, stove, stove top oven
toaster
sink
refrigerator, fridge, freezer
book
clock
vase
scissors
teddy bear, teddybear
hair drier, hairdryer
toothbrush
"""

COCO_80 = [line.split(",")[0].strip() for line in SYNONYMS_RAW.strip().splitlines()]

# Words that must be joined before matching, to avoid "hot dog" counting as "dog".
DOUBLE_WORDS = [
    "motor bike", "motor cycle", "air plane", "traffic light", "street light",
    "traffic signal", "stop light", "fire hydrant", "stop sign", "parking meter",
    "suit case", "sports ball", "baseball bat", "baseball glove", "tennis racket",
    "wine glass", "hot dog", "cell phone", "mobile phone", "teddy bear",
    "hair drier", "hair dryer", "potted plant", "bow tie", "laptop computer",
    "stove top oven", "hot dogs", "teddy bears", "dining table",
]

ANIMAL_WORDS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
                "zebra", "giraffe", "animal", "cub"}
VEHICLE_WORDS = {"jet", "train"}


def load_synonyms(path: str | Path | None = None) -> tuple[dict[str, str], list[str]]:
    """Return (word -> canonical COCO class, ordered class list)."""
    raw = Path(path).read_text() if path else SYNONYMS_RAW
    mapping: dict[str, str] = {}
    classes: list[str] = []
    for line in raw.strip().splitlines():
        parts = [p.strip().lower() for p in line.split(",") if p.strip()]
        if not parts:
            continue
        canon = parts[0]
        classes.append(canon)
        for p in parts:
            mapping[p] = canon
            if not p.endswith("s"):
                mapping[p + "s"] = canon
    return mapping, classes


_TOKEN_RE = re.compile(r"[a-zA-Z]+")


def _normalise(caption: str) -> list[str]:
    words = [w.lower() for w in _TOKEN_RE.findall(caption)]
    # join known double words
    out, i = [], 0
    while i < len(words):
        if i + 1 < len(words):
            pair = f"{words[i]} {words[i+1]}"
            if pair in DOUBLE_WORDS:
                out.append(pair.replace(" ", "_"))
                i += 2
                continue
        out.append(words[i])
        i += 1
    return out


class ChairScorer:
    def __init__(self, coco_root, subset: str = "val2014",
                 synonyms_path: str | None = None, use_captions: bool = True):
        from .data import load_captions, load_instances
        self.mapping, self.classes = load_synonyms(synonyms_path)
        # register underscore forms of the double words
        for dw in DOUBLE_WORDS:
            canon = self.mapping.get(dw)
            if canon:
                self.mapping[dw.replace(" ", "_")] = canon
        self.instances, _ = load_instances(Path(coco_root), subset)
        self.caption_gt: dict[int, set[str]] = defaultdict(set)
        if use_captions:
            for iid, caps in load_captions(Path(coco_root), subset).items():
                for c in caps:
                    self.caption_gt[iid] |= self.extract(c)

    def extract(self, caption: str) -> set[str]:
        found = set()
        for w in _normalise(caption):
            c = self.mapping.get(w) or self.mapping.get(w.replace("_", " "))
            if c:
                found.add(c)
        return found

    def gt_objects(self, image_id: int) -> set[str]:
        return set(self.instances.get(image_id, set())) | self.caption_gt.get(image_id, set())

    def score(self, records: Iterable[dict]) -> dict:
        """records: [{'image_id': int, 'caption': str}]"""
        n_cap = n_hall_cap = 0
        n_obj = n_hall_obj = 0
        lens, hallucinated, per_caption = [], [], []
        for r in records:
            gt = self.gt_objects(int(r["image_id"]))
            mentioned = self.extract(r["caption"])
            hall = mentioned - gt
            n_cap += 1
            n_obj += len(mentioned)
            n_hall_obj += len(hall)
            if hall:
                n_hall_cap += 1
            lens.append(len(r["caption"].split()))
            hallucinated.append(sorted(hall))
            per_caption.append((len(mentioned), len(hall)))
        return {
            "CHAIR_s": n_hall_cap / max(n_cap, 1),
            "CHAIR_i": n_hall_obj / max(n_obj, 1),
            "n_captions": n_cap,
            "objects_per_caption": n_obj / max(n_cap, 1),
            "avg_len": sum(lens) / max(len(lens), 1),
            "recall_objects": None,
            "hallucinated": hallucinated,
            "per_caption": per_caption,        # (mentioned, hallucinated) per caption
        }

    def per_caption_coverage(self, records: Iterable[dict]) -> list[tuple[int, int]]:
        """(GT objects mentioned, GT objects present) per caption, for paired bootstrap."""
        out = []
        for r in records:
            gt = self.gt_objects(int(r["image_id"]))
            out.append((len(gt & self.extract(r["caption"])), len(gt)))
        return out

    def coverage(self, records: Iterable[dict]) -> float:
        """Object recall: fraction of GT objects that get mentioned.

        Report it next to CHAIR. A model that says almost nothing trivially wins on
        CHAIR, so CHAIR without coverage is not evidence of anything.
        """
        hit = tot = 0
        for r in records:
            gt = self.gt_objects(int(r["image_id"]))
            if not gt:
                continue
            m = self.extract(r["caption"])
            hit += len(gt & m)
            tot += len(gt)
        return hit / max(tot, 1)
