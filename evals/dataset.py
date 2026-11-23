"""Synthetic household: the (fictional) Marlowe family.

QUERIES is the dev set (ranking may be tuned on it). HOLDOUT_QUERIES is only
for reporting: do not tune against it.


Everything here is invented. Each memory has a key so queries can name the
memory that *should* come back. `days` = how long ago it was written.
"""
from __future__ import annotations
import random

# key: (text, days_ago, supersedes_key_or_None)
FACTS = {
    "job_old": ("Dana works as a structural engineer at Halvorsen & Pike in Edinburgh.", 400, None),
    "job_new": ("Dana works as a senior structural engineer at Brightwater Consulting.", 60, "job_old"),
    # contradiction written the way a chat model would: a new fact, no supersedes
    "home_old": ("Dana and the family live in a flat in Leith, Edinburgh.", 400, None),
    "home_new": ("The family moved to a house in Dunbar, East Lothian.", 120, None),
    "partner": ("Dana's partner is Riley Okafor, a paediatric nurse at the Royal Infirmary.", 390, None),
    "son": ("Dana's son Theo is six years old and is in Primary 2 at Belhaven Primary.", 100, None),
    "daughter": ("Dana's daughter Isla is nine years old and plays the cello.", 380, None),
    "theo_allergy": ("Theo is allergic to peanuts and carries an EpiPen.", 370, None),
    "isla_allergy": ("Isla is lactose intolerant.", 365, None),
    "dog": ("The family dog is a whippet called Biscuit.", 350, None),
    "car": ("Dana drives a 2021 Skoda Enyaq electric car, registration SK71 DMW.", 300, None),
    "wifi": ("The home router is a Fritz!Box 7590 AX and the guest network is called Limpet-Guest.", 110, None),
    "boiler": ("The boiler is a Vaillant ecoTEC plus 832, serviced every October by Forth Heating.", 105, None),
    "bday_riley": ("Riley's birthday is 14 March.", 390, None),
    "bday_theo": ("Theo's birthday is 2 September.", 390, None),
    "bday_isla": ("Isla's birthday is 30 November.", 390, None),
    "anniversary": ("Dana and Riley's wedding anniversary is 21 June.", 390, None),
    "mum": ("Dana's mother Moira lives in Inverness and visits every Easter.", 330, None),
    "project": ("Dana is leading the Tyne footbridge refurbishment project, codenamed KITTIWAKE.", 55, None),
    "norwegian": ("Dana is learning Norwegian and is at roughly A2 level.", 200, None),
    "gp": ("The family GP is Dr Anand at Lauderdale Medical Practice.", 115, None),
    "bins": ("Recycling bins are collected on alternate Thursdays; general waste goes out on Tuesdays.", 112, None),
    "tools": ("Dana uses Tekla Structural Designer and Python with pandas for load calculations.", 280, None),
    "run_club": ("Dana runs with the Dunbar Running Club on Wednesday evenings.", 90, None),
    "marathon": ("Dana is training for the Loch Ness Marathon at the end of September.", 80, None),
    "knee": ("Dana had meniscus surgery on the left knee two years ago and avoids downhill sprints.", 300, None),
    "pickup": ("School pickup at Belhaven Primary is at 3:15pm, except Fridays when it is 12:30pm.", 95, None),
    "cello_lesson": ("Isla has cello lessons on Mondays at 4:30pm with Mrs Lindqvist.", 95, None),
    "holiday": ("The family has booked a holiday in Bergen, Norway from 12 to 19 July.", 40, None),
    "laptop": ("Dana's work laptop is a ThinkPad P1 Gen 6 on Windows 11; the home computer is a Mac mini.", 58, None),
    "riley_veg": ("Riley is vegetarian.", 385, None),
    "theo_likes": ("Theo loves dinosaurs, especially ankylosaurus, and building Lego.", 150, None),
    "riley_shifts": ("Riley works three twelve-hour shifts a week, usually Thursday to Saturday.", 70, None),
    "dentist": ("The family dentist is Tyninghame Dental; check-ups are every six months.", 110, None),
}

# key: (text, days_ago, strength)
PREFS = {
    "terse": ("Dana prefers short, direct answers with no preamble.", 395, 1.0),
    "units": ("Dana prefers metric units and 24-hour times.", 390, 0.9),
    "coffee": ("Dana drinks oat-milk flat whites and avoids caffeine after 2pm.", 250, 0.7),
    "code_style": ("Dana prefers Python type hints and pytest over unittest.", 270, 0.8),
    "meetings": ("Dana does not want meetings scheduled before 9:30am.", 59, 0.9),
    "briefing": ("Dana likes a morning briefing covering weather, rail disruptions and engineering news.", 180, 0.6),
    "music": ("Dana listens to ambient and Nordic jazz while working.", 220, 0.4),
    "dinner": ("On weeknights Dana likes one-pot dinners that take under 30 minutes.", 140, 0.7),
    "email_tone": ("Dana wants emails drafted in a warm but formal tone, signed 'Best, Dana'.", 260, 0.8),
    "bedtime_story": ("Bedtime stories for Theo should be calm and under ten minutes.", 130, 0.9),
    "travel": ("Dana prefers trains over flights for journeys under five hours.", 240, 0.8),
    "seat_old": ("Dana prefers aisle seats when travelling.", 300, 0.5),
    "seat_new": ("Dana now prefers window seats when travelling.", 20, 0.9),
    "no_emoji": ("Dana dislikes emoji in written replies.", 395, 0.8),
    "spice": ("Dana does not like very spicy food; Riley loves it.", 200, 0.6),
}

# key: (text, days_ago)
EPISODES = {
    "vet": ("Took Biscuit to the vet in North Berwick because of a limp in the back leg; sent home with anti-inflammatories and a week of rest.", 18),
    "tooth": ("Theo lost his first tooth at breakfast and was thrilled about the tooth fairy.", 33),
    "long_run": ("Dana ran a 21 km long run along the John Muir Way in 1 hour 58 minutes and felt strong.", 9),
    "boiler_fix": ("The boiler lost pressure overnight; Forth Heating came out and replaced the expansion vessel.", 47),
    "design_review": ("KITTIWAKE design review: the client asked for a revised deck vibration analysis by 5 December.", 12),
    "night_shifts": ("Riley started two weeks of night shifts on the children's ward.", 25),
    "move_day": ("Moving day: the removal van arrived at 7am and the family slept in the Dunbar house for the first time.", 120),
    "recital": ("Isla played Saint-Saens' The Swan at the school spring recital and got a standing ovation.", 210),
    "epipen": ("Renewed Theo's EpiPen prescription with Dr Anand; the old one had expired.", 75),
    "interview": ("Dana had the final interview at Brightwater Consulting and received the offer the same afternoon.", 78),
    "car_service": ("The Enyaq had its annual service at the Skoda dealer in Musselburgh; rear tyres replaced.", 64),
    "norsk_exam": ("Dana passed the Norskprove A2 written test.", 50),
    "parents_evening": ("Parents' evening for Theo: the teacher said his reading is ahead but he rushes his handwriting.", 29),
    "isla_bike": ("Isla fell off her bike on the harbour path and sprained her wrist; no cello for two weeks.", 14),
    "bergen_booking": ("Booked ferries and a cabin near Bergen for the July holiday; paid the deposit.", 40),
    "wifi_setup": ("Set up the new Fritz!Box and moved the printer onto the main network.", 108),
    "moira_visit": ("Moira came down from Inverness for the Easter weekend and took the kids to the beach.", 230),
    "half_marathon": ("Dana ran the Edinburgh half marathon in 1:49:12, a personal best.", 170),
    "pasta_fail": ("Tried a one-pot orzo recipe; Theo refused it, everyone else liked it.", 6),
    "roof_leak": ("Found a leak in the utility room roof after the storm; roofer booked for next Tuesday.", 3),
    "biscuit_bday": ("Biscuit turned four; the kids made him a dog-safe cake.", 85),
    "teams_outage": ("Lost half a day at work because the Tekla licence server was down.", 22),
    "theo_swim": ("Theo swam a full width of the pool without armbands for the first time.", 44),
    "dentist_visit": ("Whole family check-up at Tyninghame Dental; Isla needs a brace consultation.", 36),
    "site_visit": ("Site visit to the Tyne footbridge: found more corrosion on the north bearing than the survey showed.", 27),
}

# ---------------------------------------------------------------- queries
# (query, [acceptable keys], category)  -- any acceptable key counts as a hit.
QUERIES = [
    # direct: wording overlaps the memory
    ("What car does Dana drive?", ["car"], "direct"),
    ("What breed is the family dog?", ["dog"], "direct"),
    ("Who is the family GP?", ["gp"], "direct"),
    ("When is the wedding anniversary?", ["anniversary"], "direct"),
    ("What is Dana training for?", ["marathon"], "direct"),
    ("Which language is Dana learning?", ["norwegian"], "direct"),
    ("What time is school pickup?", ["pickup"], "direct"),
    ("When are Isla's cello lessons?", ["cello_lesson"], "direct"),
    # paraphrase: little or no word overlap
    ("which vehicle do we own", ["car"], "paraphrase"),
    ("what pet do we have", ["dog"], "paraphrase"),
    ("which doctor do we see", ["gp"], "paraphrase"),
    ("when do I need to put the rubbish out", ["bins"], "paraphrase"),
    ("where are we going on our summer trip", ["holiday", "bergen_booking"], "paraphrase"),
    ("what does my spouse do for a living", ["partner"], "paraphrase"),
    ("does anyone at home avoid meat", ["riley_veg"], "paraphrase"),
    ("what instrument does my girl play", ["daughter", "cello_lesson"], "paraphrase"),
    ("who heats our house and who maintains it", ["boiler"], "paraphrase"),
    ("old injuries I should be careful with when exercising", ["knee"], "paraphrase"),
    ("what software do I use for my engineering maths", ["tools"], "paraphrase"),
    ("how do visitors get online at our place", ["wifi"], "paraphrase"),
    ("which computer does the office give me", ["laptop"], "paraphrase"),
    ("when is grandma coming to stay", ["mum", "moira_visit"], "paraphrase"),
    # rare tokens: identifiers that only match literally
    ("SK71 DMW", ["car"], "rare_token"),
    ("KITTIWAKE", ["project", "design_review"], "rare_token"),
    ("ecoTEC", ["boiler"], "rare_token"),
    ("Limpet-Guest password network", ["wifi"], "rare_token"),
    ("ThinkPad P1", ["laptop"], "rare_token"),
    ("Mrs Lindqvist", ["cello_lesson"], "rare_token"),
    # entity: two similar people/things, only one is right
    ("What can Isla not eat or drink?", ["isla_allergy"], "entity"),
    ("What food is dangerous for Theo?", ["theo_allergy"], "entity"),
    ("When is Theo's birthday?", ["bday_theo"], "entity"),
    ("When is Riley's birthday?", ["bday_riley"], "entity"),
    ("When is Isla's birthday?", ["bday_isla"], "entity"),
    ("How old is my son and what class is he in?", ["son"], "entity"),
    ("How old is my daughter?", ["daughter"], "entity"),
    # updated: the current value must win over the stale one
    ("Where does Dana work?", ["job_new"], "updated"),
    ("Who is my employer?", ["job_new"], "updated"),
    ("Where do we live?", ["home_new"], "updated"),
    ("What is our home town?", ["home_new"], "updated"),
    ("Do I like aisle or window seats?", ["seat_new"], "updated"),
    # preferences
    ("How long should replies be?", ["terse"], "preference"),
    ("Is 8am okay for a call?", ["meetings"], "preference"),
    ("Should I fly or take the train to London?", ["travel"], "preference"),
    ("What should I cook on a Tuesday night?", ["dinner"], "preference"),
    ("How should I sign off this email?", ["email_tone"], "preference"),
    ("Can I have a coffee at 4pm?", ["coffee"], "preference"),
    ("Tell Theo a story before sleep", ["bedtime_story"], "preference"),
    ("Should distances be in miles or kilometres?", ["units"], "preference"),
    ("what should I put on in the background while I work", ["music"], "preference"),
    ("which test framework should this project use", ["code_style"], "preference"),
    # episodic
    ("What happened with the dog at the vet?", ["vet"], "episodic"),
    ("When did Theo lose a tooth?", ["tooth"], "episodic"),
    ("How did my last long run go?", ["long_run"], "episodic"),
    ("What did the heating engineer fix?", ["boiler_fix"], "episodic"),
    ("What did the client ask for at the bridge review?", ["design_review"], "episodic"),
    ("How did Isla hurt herself?", ["isla_bike"], "episodic"),
    ("What did the teacher say about Theo?", ["parents_evening"], "episodic"),
    ("What is wrong with the roof?", ["roof_leak"], "episodic"),
    ("What was my half marathon time?", ["half_marathon"], "episodic"),
    ("What did we find on the site visit?", ["site_visit"], "episodic"),
    # inference: needs one step of reasoning from query to memory
    ("Can Theo have a Snickers bar?", ["theo_allergy"], "inference"),
    ("Is a cheese pizza fine for Isla?", ["isla_allergy"], "inference"),
    ("Planning a steak dinner for two on our anniversary, any problem?", ["riley_veg", "anniversary"], "inference"),
    ("Can Riley do the school run on Friday?", ["riley_shifts", "pickup"], "inference"),
    ("What should I get Theo for his birthday?", ["theo_likes", "bday_theo"], "inference"),
    ("Is it okay to add hill repeats to my training plan?", ["knee"], "inference"),
    ("Do I need to charge anything before the drive to Inverness?", ["car"], "inference"),
    ("What phrases should I practise before the July trip?", ["norwegian", "holiday"], "inference"),
]

# Things the store knows nothing about. A good memory should signal "nothing".
NEGATIVE_QUERIES = [
    "What is Dana's blood type?",
    "Which football team does Dana support?",
    "What is the mortgage interest rate?",
    "Who is Dana's accountant?",
    "What is the cat's name?",
    "What is the capital of Mongolia?",
    "How do I reverse a linked list in Rust?",
    "What is Riley's shoe size?",
    "Which university did Isla attend?",
    "What is the garage door code?",
]

# ---------------------------------------------------------------- noise
# Journal-style filler, the kind an assistant logs every session. Deliberately
# brushes against the real topics (Norway, dogs, cars, coffee, bridges...).
_TOPICS = [
    "Asked for the weather forecast for {place}; it was {wx}.",
    "Asked how to convert {n} miles to kilometres.",
    "Debugged a pandas {err} in the {script} script; fixed by {fix}.",
    "Asked for a recipe for {dish}; saved it for later.",
    "Asked what the capital of {country} is.",
    "Drafted an email to {who} about {subject}.",
    "Asked for a summary of an article about {article}.",
    "Set a timer for {n} minutes while cooking {dish}.",
    "Asked for train times from {place} to {place2}.",
    "Looked up opening hours for the {shop} in {place}.",
    "Asked for a quick explanation of {concept}.",
    "Asked for help naming a variable for {concept} in the {script} script.",
    "Compared prices of {product} across three shops.",
    "Asked for ideas for a rainy afternoon with the kids in {place}.",
    "Asked for the news headlines; main story was about {article}.",
    "Asked for a stretching routine after a {n} km run.",
    "Asked which {product} is best for a family of four.",
    "Asked how long to roast {dish}.",
    "Asked for a reminder to call {who} about {subject}.",
    "Asked to translate '{phrase}' into Norwegian.",
]
_F = dict(
    place=["Dunbar", "Edinburgh", "Inverness", "Bergen", "Glasgow", "North Berwick", "Newcastle", "Oslo", "Leith", "Haddington"],
    place2=["London", "York", "Aberdeen", "Durham", "Perth", "Stirling"],
    wx=["rain", "sunny spells", "a gale warning", "fog", "light snow", "overcast"],
    err=["KeyError", "SettingWithCopyWarning", "MergeError", "ValueError", "dtype mismatch"],
    script=["load_cases", "beam_check", "deflection", "report_export", "wind_loading"],
    fix=["renaming a column", "resetting the index", "casting to float", "dropping duplicates"],
    dish=["lentil dal", "roast chicken", "banana bread", "mushroom risotto", "fish pie", "peanut noodles", "veggie chilli", "pancakes"],
    country=["Norway", "Portugal", "Canada", "Kenya", "Japan", "Iceland"],
    who=["the plumber", "the school office", "a colleague", "the roofer", "the client", "Moira", "the vet", "the dentist"],
    subject=["an invoice", "a schedule change", "a late delivery", "the site survey", "a birthday party", "a booster jab", "a quote"],
    article=["bridge inspections", "electric car charging", "dog nutrition", "marathon training", "school funding", "heat pumps", "rail strikes", "coffee and sleep"],
    shop=["pharmacy", "garden centre", "bike shop", "library", "swimming pool", "pet shop"],
    concept=["eigenfrequency", "shear lag", "Python dataclasses", "compound interest", "tidal range", "the offside rule", "lactose", "torsional stiffness"],
    product=["dog food", "running shoes", "roof boxes", "kids' bikes", "espresso machines", "mesh wifi routers", "cellos", "electric car chargers"],
    phrase=["where is the ferry", "two coffees please", "my son is six", "we are on holiday", "is this vegetarian"],
    n=[3, 5, 8, 10, 12, 15, 20, 26, 45],
)


def noise(n: int, seed: int = 7, max_days: int = 400) -> list[tuple[str, float]]:
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        t = rng.choice(_TOPICS)
        text = t.format(**{k: rng.choice(v) for k, v in _F.items()})
        text = f"{text} (session {len(out) + 1})" if text in seen else text
        seen.add(text)
        out.append((text, rng.uniform(0.05, max_days)))
    return out


# ---------------------------------------------------------------- holdout
# Written before any tuning. Report on these; never tune on them.
HOLDOUT_QUERIES = [
    ("What is the dog's name?", ["dog"], "direct"),
    ("Who is Dana's partner?", ["partner"], "direct"),
    ("What router do we have?", ["wifi"], "direct"),
    ("Who is the family dentist?", ["dentist"], "direct"),
    ("what do I drive", ["car"], "paraphrase"),
    ("who is our physician", ["gp"], "paraphrase"),
    ("which day is the recycling picked up", ["bins"], "paraphrase"),
    ("what job does my other half have", ["partner"], "paraphrase"),
    ("which sports club am I a member of", ["run_club"], "paraphrase"),
    ("what race am I preparing for", ["marathon"], "paraphrase"),
    ("which foreign tongue am I studying", ["norwegian"], "paraphrase"),
    ("when does my little boy finish school at the end of the week", ["pickup"], "paraphrase"),
    ("who teaches my daughter music", ["cello_lesson"], "paraphrase"),
    ("what does my lad enjoy playing with", ["theo_likes"], "paraphrase"),
    ("where does my mum live", ["mum"], "paraphrase"),
    ("how often do we get our teeth checked", ["dentist", "dentist_visit"], "paraphrase"),
    ("Fritz!Box", ["wifi", "wifi_setup"], "rare_token"),
    ("Enyaq", ["car", "car_service"], "rare_token"),
    ("Belhaven", ["son", "pickup"], "rare_token"),
    ("Vaillant", ["boiler"], "rare_token"),
    ("Which child has the nut allergy?", ["theo_allergy"], "entity"),
    ("Which of the kids can't have milk?", ["isla_allergy"], "entity"),
    ("Whose birthday is in November?", ["bday_isla"], "entity"),
    ("Whose birthday is in March?", ["bday_riley"], "entity"),
    ("Which company am I with now?", ["job_new"], "updated"),
    ("What town is our house in?", ["home_new"], "updated"),
    ("Which seat should you book me on the train?", ["seat_new"], "updated"),
    ("Do I want emoji in messages?", ["no_emoji"], "preference"),
    ("How spicy should the curry be?", ["spice"], "preference"),
    ("What goes in my morning summary?", ["briefing"], "preference"),
    ("When is the earliest you can book a meeting for me?", ["meetings"], "preference"),
    ("Shall I write the tests with unittest?", ["code_style"], "preference"),
    ("What milk do I take in coffee?", ["coffee"], "preference"),
    ("How did the car service go?", ["car_service"], "episodic"),
    ("What piece did Isla play at the recital?", ["recital"], "episodic"),
    ("Did I pass my Norwegian test?", ["norsk_exam"], "episodic"),
    ("What happened when the licence server went down?", ["teams_outage"], "episodic"),
    ("How was Theo's swimming?", ["theo_swim"], "episodic"),
    ("When did we move in?", ["move_day", "home_new"], "episodic"),
    ("Is satay sauce okay for the kids' dinner?", ["theo_allergy"], "inference"),
    ("Should I put ice cream on the shopping list for Isla?", ["isla_allergy"], "inference"),
    ("Who can collect Theo at half twelve on Friday?", ["pickup", "riley_shifts"], "inference"),
    ("Should I book the 6am flight to London for the client meeting?", ["travel", "meetings"], "inference"),
]

HOLDOUT_NEGATIVES = [
    "What is Dana's passport number?",
    "Who won the 1998 World Cup?",
    "What colour is the front door?",
    "What is my sister's name?",
    "Which bank do we use?",
    "What is the recipe for sourdough?",
    "What is Riley's middle name?",
    "How many bedrooms does the house have?",
]
