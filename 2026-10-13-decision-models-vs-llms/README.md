# Decision models vs. LLMs

**Started** October 13, 2026 · **Status** In progress · [Write-up](https://usmankhan.io/lab/decision-models-vs-llms/)

How much better is a decision model than an LLM at ranking a few hundred options?

## The question

flights.cx runs ~43 searches per trip and ends up with ~280 itineraries. One step ranks those into three plans. That step was an LLM, and a search took 4+ min. I replaced it with a typed classifier and a search dropped to ~1.5 min.

That's one task in one product, and I only measured time. I didn't check if the picks got worse. I want to run it properly: same searches, same inputs, a decision model against an LLM, measuring speed, cost, consistency and how good the picks are.

Then the same again with messier input, since itineraries are clean structured data and most ranking problems aren't.

## What I expect

The decision model wins on speed and cost, probably by a lot, and it's more consistent: same input, same pick. Quality is the part I don't know. My guess is they're close on structured data and the LLM wins once there's free text to read.

## Method

1. Save 30–50 real searches from flights.cx, ~280 itineraries each, with anything personal stripped out. I pick the best 3 for each by hand and get two or three friends to do the same. That's the answer key.
2. Run every search through four pickers: the decision model, an LLM with a decent prompt, an LLM with structured output, and a plain scoring function as the floor.
3. Record time and cost per search, how often each picker matches the answer key (top 1 and top 3), and whether it gives the same answer across 5 runs.
4. Repeat with a second task where part of the input is free text.

To pin down before running:

- Which models and versions. Write them here with the date.
- How many searches. Enough that top-3 overlap isn't noise; probably 30–50.
- What to do when the friends disagree on the answer key.
- Per-call pricing on the day of the run, written here.

## Log

- 2026-10-13: started. Wrote the question, the guess and the method before touching any code.

## Results

Nothing yet.
