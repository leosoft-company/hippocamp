"""Live consolidation (reflect) with the local model, on made-up episodes.

    python evals/reflect_live.py     # needs LM Studio (or any local server) on 127.0.0.1:1234

A: extraction quality on planted signals, trivia and traps.
B: a second run with restatements, new facts, updates and a look-alike fact.
C: a shared household store where several people speak.
Prints what the model wrote and a list of checks; exit code 1 if any fails.
"""
from __future__ import annotations
import json, re, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import CachedEmbedder, fresh_dir, open_mem, ROOT
from hippocamp.reflect import local_llm

NOW = lambda: datetime.now(timezone.utc)
OUT = {}

PLANTED = [  # (episode, keyword regex that a correct distilled item must contain)
 ("Dana mentioned she has just started as a senior structural engineer at Brightwater Consulting.", "brightwater"),
 ("Helped Dana plan meals for the week; she said her partner Riley is vegetarian.", "vegetarian"),
 ("Dana asked for a peanut-free snack list because her son Theo is allergic to peanuts.", "peanut"),
 ("Dana said to always give distances in kilometres, never miles.", "kilomet|metric"),
 ("Dana asked for answers to be kept short, with no preamble.", "short|brief|concise|terse|direct"),
 ("Looked up ferry times; Dana said the family has booked a holiday in Bergen from 12 to 19 July.", "bergen"),
 ("Dana said she is training for the Loch Ness Marathon in late September.", "loch ness|marathon"),
 ("Dana said she never wants meetings before 9:30 in the morning.", "9:30"),
 ("Dana mentioned the family moved to a house in Dunbar this summer.", "dunbar"),
 ("Dana said the family dog, a whippet called Biscuit, needs walking twice a day.", "biscuit|whippet"),
 ("Dana asked for Norwegian practice sentences; she said she is at about A2 level.", "norwegian"),
 ("Dana said her daughter Isla has cello lessons every Monday at 4:30pm.", "cello"),
 ("Debugged a pandas merge error in Dana's beam load calculation script.", "pandas"),
 ("Helped Dana vectorise a pandas groupby in the wind loading script.", "pandas"),
 ("Fixed a pandas dtype bug in Dana's deflection report export.", "pandas"),
]
TRIVIA = [  # should NOT become durable memories
 ("Asked for the weather in Dunbar; it was raining.", r"weather|\brain"),
 ("Set a 12-minute timer for pasta.", "timer|pasta"),
 ("Asked what the capital of Peru is.", "peru|lima"),
 ("Converted 26 miles to kilometres.", r"\b26\b"),
 ("Asked for train times from Dunbar to York on Saturday.", "york"),
 ("Asked how tall the Eiffel Tower is.", "eiffel"),
 ("Asked for a banana bread recipe.", "banana"),
 ("Asked for the news headlines.", "headline"),
]
TRAPS = [
 ("Dana joked that if this rain keeps up she would move the whole family to Portugal.", "hypothetical", "portugal"),
 ("Dana mentioned her colleague Priya is vegan and runs ultramarathons.", "third_party", "vegan|ultramarathon"),
 ("Dana said she does not drink alcohol any more.", "negation", "alcohol"),
 ("Dana said she prefers aisle seats on trains.", "changed_mind_old", "aisle"),
 ("Dana said she has changed her mind and now prefers window seats on trains.", "changed_mind_new", "window"),
 ("Dana was frustrated today because the Tekla licence server was down all morning.", "transient", "licence|license|server"),
]

RUN_B = [
 ("Dana reminded me that Theo cannot have anything with peanuts.", "restate"),
 ("Dana again asked for short answers.", "restate"),
 ("Planned a vegetarian dinner because of Riley.", "restate"),
 ("Dana said her mother Moira lives in Inverness.", "new:moira|inverness"),
 ("Dana said the family GP is Dr Anand at Lauderdale Medical Practice.", "new:anand"),
 ("Dana said she has left Brightwater Consulting and now works at Forth Bridge Engineering.", "update:forth bridge"),
 ("Dana said meetings before 10am are now off limits, not 9:30.", "update:10"),
 ("Dana said her daughter Isla is allergic to penicillin.", "lookalike:penicillin"),
]

HOUSEHOLD = [
 "Riley said the recycling goes out on alternate Thursdays.",
 "Theo said his favourite dinosaur is the ankylosaurus.",
 "Riley said she works Thursday to Saturday this month.",
 "Dana said the boiler service is booked with Forth Heating every October.",
 "Isla said her cello lesson has moved to Tuesdays at 5pm.",
 "Riley said she is vegetarian.",
 "Theo said he wants spaghetti every Friday.",
 "Dana said he is training for a marathon and runs on Wednesday evenings.",
]


def rows(mem, where="superseded_at IS NULL"):
    return mem._store._conn.execute(
        f"SELECT id, kind, text, metadata FROM memories WHERE kind != 'episode' AND {where} ORDER BY kind, created_at").fetchall()


def show(title, rs):
    print(f"\n--- {title} ({len(rs)})")
    for _, k, t, _ in rs: print(f"   [{k[:4]}] {t}")


class Timed:
    def __init__(self, inner): self.inner, self.calls = inner, []
    def complete(self, prompt, *, system=None):
        t = time.time(); out = self.inner.complete(prompt, system=system)
        self.calls.append({"s": round(time.time() - t, 1), "prompt_chars": len(prompt), "reply_chars": len(out)}); return out


emb = CachedEmbedder(); llm = Timed(local_llm())

CHECKS = []
def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok))); print(f"   {'PASS' if ok else 'FAIL'}  {name}  {detail}")


d = fresh_dir("reflect_live") / "dana"
import random


def run_abc():
    global OUT

    mem = open_mem(d, emb, llm=llm)
    eps = [(t, "planted", k) for t, k in PLANTED] + [(t, "trivia", k) for t, k in TRIVIA]
    random.Random(3).shuffle(eps)
    order = eps[:12] + [(TRAPS[3][0], TRAPS[3][1], TRAPS[3][2])] + eps[12:] + [(t, kind, k) for t, kind, k in TRAPS if kind != "changed_mind_old"]
    for i, (t, _, _) in enumerate(order):
        mem.observe(t, at=NOW() - timedelta(hours=len(order) - i))
    rep = mem.reflect()
    A = rows(mem); texts = [r[2] for r in A]; low = [t.lower() for t in texts]
    print("A report:", rep.model_dump(), llm.calls[-1])
    show("A: everything Gemma wrote", A)
    planted_kw = sorted({k for _, k in PLANTED})
    found = {k: any(re.search(k, t) for t in low) for k in planted_kw}
    leaks = {k: [t for t in texts if re.search(k, t.lower())] for _, k in TRIVIA}
    leaks = {k: v for k, v in leaks.items() if v}
    traps = {kind: [t for t in texts if re.search(k, t.lower())] for _, kind, k in TRAPS}
    user_form = sum(t.lower().startswith(("user", "the user")) for t in texts)
    print(f"\nA planted signals captured: {sum(found.values())}/{len(found)}; missed: {[k for k, v in found.items() if not v]}")
    print("A trivia that leaked into durable memory:", leaks or "none")
    print("A traps:"); [print(f"   {k}: {v or '(nothing written)'}") for k, v in traps.items()]
    check("A: at least 12 of 13 planted signals captured", sum(found.values()) >= 12, f"{sum(found.values())}/13")
    check("A: no trivia stored", not leaks)
    check("A: joke and transient gripe not stored", not traps["hypothetical"] and not traps["transient"])
    check("A: colleague's diet not attributed to Dana", all("priya" in t.lower() or "social circle" in t.lower() for t in traps["third_party"]))
    check("A: changed mind keeps only the new preference", traps["changed_mind_new"] and not traps["changed_mind_old"])
    OUT["A"] = {"report": rep.model_dump(), "call": llm.calls[-1], "captured": f"{sum(found.values())}/{len(found)}", "missed": [k for k, v in found.items() if not v],
                "leaks": leaks, "traps": traps, "items": texts}

    # ------------------------------------------------------------------ B
    time.sleep(1)
    for t, _ in RUN_B: mem.observe(t)
    before = {r[0]: r[2] for r in A}
    rep = mem.reflect()
    B = rows(mem); B_new = [r for r in B if r[0] not in before]
    gone = [(i, t) for i, t in before.items() if i not in {r[0] for r in B}]
    print("\nB report:", rep.model_dump(), llm.calls[-1])
    show("B: newly written", B_new)
    print("\n--- B: earlier memories that were replaced (superseded)")
    for i, t in gone:
        by = [r[2] for r in B_new if i in json.loads(r[3]).get("supersedes", [])]
        print(f"   OLD {t!r}\n     -> BY {by}")
    act = [r[2].lower() for r in B]
    dup = {k: sum(bool(re.search(k, t)) for t in act) for k in ("peanut", "vegetarian", "short|brief|concise|terse|direct", "brightwater", "forth bridge", "9:30", "10 ?am|10:00", "penicillin", "moira|inverness", "anand")}
    print("\nB active memories mentioning each topic:", dup)
    facts_act = [r[2].lower() for r in B if r[1] != "reflection"]
    check("B: old employer replaced by the new one", not any("brightwater" in t for t in facts_act) and any("forth bridge" in t for t in facts_act))
    check("B: 9:30 meeting rule replaced by 10am", not any("9:30" in t for t in facts_act) and any(re.search("10 ?am|10:00", t) for t in facts_act))
    check("B: Theo's peanut allergy untouched by Isla's penicillin allergy", any("peanut" in t and "theo" in t for t in facts_act) and any("penicillin" in t for t in facts_act))
    check("B: restatements not stored again", sum("peanut" in t for t in facts_act) == 1 and sum("vegetarian" in t for t in facts_act) == 1)
    check("B: new facts stored", dup["moira|inverness"] >= 1 and dup["anand"] >= 1)
    OUT["B"] = {"report": rep.model_dump(), "call": llm.calls[-1], "new": [r[2] for r in B_new], "replaced": gone, "topic_counts": dup}

    # ------------------------------------------------------------------ C
    d2 = d.parent / "household"; hm = open_mem(d2, emb, llm=llm)
    for i, t in enumerate(HOUSEHOLD): hm.observe(t, at=NOW() - timedelta(minutes=len(HOUSEHOLD) - i))
    rep = hm.reflect(); C = rows(hm)
    print("\nC report:", rep.model_dump(), llm.calls[-1])
    show("C: household store", C)
    anon = [r[2] for r in C if re.match(r"(the )?user\b", r[2].lower()) and not re.search(r"riley|theo|isla|dana", r[2].lower())]
    print(f"\nC items that say 'User' without naming the person: {len(anon)}/{len(C)}"); [print("   ", t) for t in anon]
    check("C: household facts name the person", not anon and len(C) >= 5)
    OUT["C"] = {"report": rep.model_dump(), "items": [r[2] for r in C], "anonymous_user_items": anon}


if "--backlog" not in sys.argv:
    run_abc()

# ------------------------------------------------------------------ D: backlog
d3 = d.parent / "backlog"; bm = open_mem(d3, emb, llm=llm)
import dataset as D
for i, (t, _) in enumerate(D.noise(55, seed=11)): bm.observe(t, at=NOW() - timedelta(days=30, minutes=-i))
t0 = time.time(); rep = bm.reflect()
print("\nD report:", rep.model_dump(), f"{time.time() - t0:.0f}s")
check("D: a 55-episode month-old backlog is read in full, in batches", rep.episodes_processed == 55 and rep.batches >= 3 and rep.pending == 0, f"{rep.batches} batches")
emb.save(); (ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "reflect_latest.json").write_text(json.dumps(OUT, indent=1, default=str))
bad = [n for n, ok in CHECKS if not ok]
print(f"\nCHECKS: {len(CHECKS) - len(bad)}/{len(CHECKS)} passed"); [print("  ✗", n) for n in bad]
sys.exit(1 if bad else 0)
