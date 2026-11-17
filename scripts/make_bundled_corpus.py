"""Generate the small bundled sample corpus and the demo documents.

The 30 "bundled sample documents" exist so the plagiarism engine has something to
compare against when no external corpus is available.  They are **synthetic sample
texts written for this project** (not scraped articles), and every place that reports
the corpus size also says so.  The two demo documents behind the "Try a sample"
buttons are generated the same way.

Run from the repository root:  ``python scripts/make_bundled_corpus.py``
"""

from __future__ import annotations

import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLED = ROOT / "ml" / "bundled" / "corpus"
SAMPLES = ROOT / "ml" / "bundled" / "samples"

# Neutral, original paragraphs about generic academic topics.
TOPICS: dict[str, list[str]] = {
    "library_automation": [
        "Library automation began as a way to replace paper catalogues with searchable records. "
        "Early systems stored only titles and call numbers, and staff still walked the shelves to "
        "check availability. Modern systems add borrowing history, reservation queues and shelf "
        "location maps, which makes the catalogue a working tool rather than a static list.",
        "A useful automation project starts by measuring the tasks that consume the most staff time. "
        "Loan renewal and fine collection are usually the first candidates, because both follow "
        "predictable rules and produce frequent small errors when handled by hand.",
    ],
    "water_quality": [
        "Water quality monitoring depends on regular sampling at fixed points along a river. "
        "Turbidity, dissolved oxygen and phosphate concentration change quickly after rainfall, so "
        "a weekly sample can miss the event that matters. Continuous sensors reduce that risk but "
        "require calibration and a plan for handling missing readings.",
        "Community groups often collect the samples that official programmes cannot afford. Training "
        "volunteers to use a single probe consistently matters more than buying an expensive "
        "instrument, because inconsistency between observers hides real trends.",
    ],
    "public_transport": [
        "Bus reliability is usually described by the gap between scheduled and actual arrival times. "
        "A route can have good average punctuality while a small number of trips run very late, and "
        "those trips are the ones passengers remember and plan around.",
        "Improving reliability is often cheaper than increasing frequency. Signal priority at two "
        "junctions and a realistic timetable can remove most of the delay that passengers notice, "
        "whereas an extra vehicle costs money every day of the year.",
    ],
    "urban_trees": [
        "Street trees lower surface temperature and interrupt the wind that carries dust. Their "
        "benefit depends on canopy volume rather than the number of trunks, so a hundred saplings "
        "can matter less than twenty mature trees with connected crowns.",
        "Planting programmes fail most often at the maintenance stage. A tree needs water for the "
        "first three summers and protection from construction work, and both are easy to forget "
        "once the photographs of planting day have been taken.",
    ],
    "study_habits": [
        "Spaced practice outperforms rereading for almost every kind of material. The gain comes "
        "from retrieving information when it feels slightly difficult, which is why a quiz taken a "
        "week later teaches more than a quiz taken immediately.",
        "Students often confuse familiarity with understanding. A chapter that feels easy on the "
        "second read may still be impossible to explain aloud, and explaining aloud is closer to "
        "what an examination actually asks for.",
    ],
    "food_preservation": [
        "Drying, salting and fermenting all work by making the environment hostile to the "
        "microorganisms that spoil food. The methods differ in which variable they change, but the "
        "underlying principle is the same, and mixing them poorly is the usual cause of failure.",
        "Home preservation fails most often because a recipe is scaled up without adjusting time. "
        "A larger batch needs longer to reach the required temperature, and the recipe that worked "
        "for two kilograms may quietly under-process five.",
    ],
    "renewable_energy": [
        "Solar output depends on panel angle, air temperature and dust. A panel that is perfectly "
        "aligned in winter may be badly aligned in summer, and a single dusty week can erase the "
        "benefit of a careful installation plan.",
        "Storage changes the economics of a small solar installation more than panel efficiency "
        "does. Shifting self-consumption to the evening reduces the power drawn from the grid, "
        "which is usually the expensive part of the bill.",
    ],
    "sleep_research": [
        "Sleep studies distinguish between time in bed and time asleep, and the difference is often "
        "the most informative measurement. People who report insomnia frequently spend long periods "
        "awake in bed, which reinforces the association between the bed and wakefulness.",
        "Light exposure in the morning anchors the internal clock. Evening light delays it, so a "
        "late screen habit can shift sleep timing by more than an hour without the person noticing "
        "any change in how tired they feel.",
    ],
    "software_maintenance": [
        "Most of the cost of a software system is spent after the first release. Maintenance work "
        "includes fixing defects, adapting to new platforms and removing features that no longer "
        "earn their keep, and only the first of these feels like traditional engineering.",
        "Tests protect maintenance work only when they fail for the right reason. A suite that "
        "asserts on formatting will break with every refactor while missing a wrong calculation, "
        "which is worse than no suite because it trains the team to ignore failures.",
    ],
    "archaeology": [
        "Archaeological survey maps what survives rather than what was used. A field that produced "
        "pottery may have held a settlement, or it may have been the place where broken pots were "
        "thrown, and distinguishing the two depends on careful recording of position.",
        "Dating a site depends on several independent methods agreeing. Radiocarbon dates, "
        "stratigraphy and stylistic comparison of finds each carry uncertainty, and reporting a "
        "single number without that uncertainty misrepresents the evidence.",
    ],
    "language_learning": [
        "Comprehensible input drives language learning more reliably than memorising grammar rules. "
        "Learners who read and listen extensively build a feel for what sounds correct, and that "
        "feel supports production long before explicit rules can be recalled quickly.",
        "Vocabulary is forgotten quickly without encounters in context. A word met once in a list "
        "and once in a novel is remembered far better than a word met three times in the same list, "
        "because the different contexts provide separate retrieval routes.",
    ],
    "manual_work_ergonomics": [
        "Repetitive manual work produces injury slowly and invisibly. Discomfort at the end of a "
        "shift is easy to dismiss, yet it is the earliest signal that posture or tool height is "
        "wrong, and adjusting the workstation early costs far less than treatment later.",
        "Breaks are more effective when they are frequent and short. A five minute pause every hour "
        "reduces strain more than a single long break, because recovery happens before fatigue "
        "accumulates rather than after it has already reduced control and attention.",
    ],
}


def _documents(count: int = 30, seed: int = 42) -> list[tuple[str, str]]:
    """Combine topic paragraphs into ``count`` documents of two paragraphs each."""
    random.Random(seed)  # deterministic ordering below does not need the RNG
    names = list(TOPICS)
    docs: list[tuple[str, str]] = []
    for index in range(count):
        first = names[index % len(names)]
        second = names[(index * 5 + 3) % len(names)]
        paragraph_a = TOPICS[first][0]
        paragraph_b = TOPICS[second][1]
        title = f"{first.replace('_', ' ').title()} — sample note {index + 1:02d}"
        text = f"{title}\n\n{paragraph_a}\n\n{paragraph_b}\n"
        docs.append((f"sample_{index + 1:02d}_{first}", text))
    return docs


HUMAN_SAMPLE = """My first attempt at keeping a sourdough starter failed in a way I still find funny.

I fed it twice a day for a week, kept it warm, and then left it on the counter during a heatwave while I was away for two days. When I came back there was a layer of dark liquid on top and a smell somewhere between vinegar and old socks. I asked a friend who has baked for years and she laughed, told me that was normal, and said to pour the liquid off instead of throwing the whole thing out.

The second starter worked better, mostly because I stopped checking it every hour. I fed it once a day, kept it in the corner where the kitchen is coolest, and only looked at it when I actually wanted to bake. It doubled reliably after about three weeks.

The bread itself is still hit and miss. My first loaf came out flat because I shaped it too tightly and it could not rise, and the second burned on the bottom because I put it too low in the oven. Both were edible. I have stopped expecting the third try to be perfect."""

AI_SAMPLE = """Artificial intelligence has become an increasingly important topic in modern society. It is essential to consider the various implications of this technology as it continues to develop at a rapid pace.

One of the most significant advantages of artificial intelligence is its ability to process large amounts of data efficiently. This capability can lead to improved decision-making across a wide range of industries, including healthcare, finance, and education. Furthermore, these systems can operate continuously without experiencing fatigue, which can result in increased productivity and reduced operational costs.

However, it is important to note that the development of artificial intelligence also raises several concerns. These include issues related to privacy, security, and the potential displacement of workers. It is therefore crucial that stakeholders collaborate to ensure that these technologies are deployed in a responsible and ethical manner.

In conclusion, artificial intelligence presents both opportunities and challenges. By carefully considering these factors, society can harness the benefits of this technology while mitigating its risks."""


def main() -> int:
    BUNDLED.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    for name, text in _documents():
        (BUNDLED / f"{name}.txt").write_text(text)
    (SAMPLES / "sample_human.txt").write_text(HUMAN_SAMPLE)
    (SAMPLES / "sample_ai.txt").write_text(AI_SAMPLE)
    print(f"wrote {len(list(BUNDLED.glob('*.txt')))} bundled documents to {BUNDLED}")
    print(f"wrote demo samples to {SAMPLES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
