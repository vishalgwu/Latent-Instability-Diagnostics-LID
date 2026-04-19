# Annotation Guidelines v1
**LID Research — Token-Level Hallucination Annotation**
**Required agreement: Fleiss κ ≥ 0.60**
**Date:** Week 3

---

## What You Are Annotating

For each model-generated response, you will label every token as either:

```
0 = CORRECT    The token is factually accurate and appropriate
1 = HALLUCINATED    The token is factually wrong or fabricated
```

---

## The Golden Rule

> **A token is hallucinated if and only if it contributes to a factually false claim that cannot be verified from general knowledge.**

---

## ALWAYS Label 0 (Never Hallucinated)

| Token type | Example | Reason |
|---|---|---|
| Grammar words | `the`, `a`, `is`, `of`, `and` | Cannot be factually wrong |
| Punctuation | `.`, `,`, `?`, `!` | Cannot be factually wrong |
| Pronouns | `it`, `they`, `he` | Structural, not factual |
| Hedging words | `probably`, `I think`, `approximately` | Signals uncertainty correctly |
| Question echoing | Model repeats the question | Not a factual claim |
| Correct facts | `Paris is the capital of France` | Verifiably true |
| Reasonable inference | `Einstein was a physicist` (when asked about relativity) | Correct |

---

## ALWAYS Label 1 (Hallucinated)

| Token type | Example | Reason |
|---|---|---|
| Wrong named entity | `Einstein invented the telephone` | Fabricated attribution |
| Wrong date/number | `WW2 ended in 1950` | Factually incorrect |
| Fabricated citation | `According to a 2019 MIT study...` (when no such study exists) | Fabricated |
| Wrong location | `The Eiffel Tower is in London` | Factually wrong |
| Invented person | Claims about a person who doesn't exist | Fabricated |

---

## GREY AREA RULES (memorize these)

### Rule 1: Label at the FIRST wrong token
If the model says `Einstein was born in 1900` (correct year is 1879):
```
Einstein   → 0  (correct name)
was        → 0  (grammar)
born       → 0  (correct claim so far)
in         → 0  (grammar)
1900       → 1  ← first wrong token
```

### Rule 2: Label the SPAN, not just the wrong word
If the model says `Einstein invented the telephone`:
```
Einstein   → 0
invented   → 1  ← wrong verb — the claim becomes wrong here
the        → 1  ← part of wrong span
telephone  → 1  ← part of wrong span
```

### Rule 3: Correct claims about the wrong topic = 0
If the model answers a question about France with correct facts about Germany:
```
Germany is a country in Europe  → all 0
```
The facts are correct, just irrelevant. Relevance is NOT hallucination.

### Rule 4: Incomplete ≠ hallucinated
`The capital of France is a city in Europe` — incomplete but not wrong → all `0`

### Rule 5: Repetition is NOT hallucination
If the model repeats itself, label each repetition the same as the original.

---

## Worked Examples

### Example 1 — Mostly correct response
```
Q: What is the capital of France?
A: The capital of France is Paris, a beautiful city known for the Eiffel Tower.

Token labels:
The          → 0
capital      → 0
of           → 0
France       → 0
is           → 0
Paris        → 0  (correct)
,            → 0
a            → 0
beautiful    → 0  (subjective, not factual claim)
city         → 0
known        → 0
for          → 0
the          → 0
Eiffel       → 0  (correct)
Tower        → 0  (correct)
.            → 0

Hallucination rate: 0/16 = 0%
```

### Example 2 — Contains hallucination
```
Q: Who invented the telephone?
A: The telephone was invented by Thomas Edison in 1876.

Token labels:
The          → 0
telephone    → 0
was          → 0
invented     → 0
by           → 0
Thomas       → 1  ← wrong person (Bell invented it, not Edison)
Edison       → 1  ← part of wrong span
in           → 1  ← part of wrong claim
1876         → 0  ← this date is actually correct for Bell's patent

Note: 1876 is correct (Bell's patent year) so label 0 even though
      the surrounding claim is wrong. Label each token independently.

Hallucination rate: 3/9 = 33% — above 15%, reasonable for this response.
```

### Example 3 — Model repeats question
```
Q: What year did WW2 end?
A: What year did WW2 end? 1945. The war ended in 1945 with the surrender of Germany and Japan.

Token labels:
What         → 0  (repetition of question — not a factual claim)
year         → 0
did          → 0
WW2          → 0
end          → 0
?            → 0
1945         → 0  (correct)
.            → 0
The          → 0
war          → 0
ended        → 0
in           → 0
1945         → 0  (correct)
with         → 0
the          → 0
surrender    → 0  (correct)
of           → 0
Germany      → 0  (correct)
and          → 0
Japan        → 0  (correct)
.            → 0

Hallucination rate: 0%
```

---

## Calibration Check

Before annotating all 100 examples, all 3 annotators do a **pilot on 10 examples together.**

Expected:
- Agreement rate ≥ 85% on pilot
- If below: discuss the cases you disagreed on and update guidelines
- Only proceed when all 3 annotators agree on the worked examples above

---

## What To Do When Unsure

1. Re-read the question and gold answer
2. Ask: "Is this token part of a factually false claim?"
3. If still unsure → label `0` (conservative)
4. Make a note in the skip/comment function
5. Bring it to the calibration session

**When in doubt → label 0. False negatives (missing a hallucination) are less harmful to the research than false positives (calling correct text hallucinated).**

---

## Expected Statistics

| Metric | Expected | Warning if |
|---|---|---|
| Hallucination rate | 5–15% | > 20% or < 3% |
| Examples with 0% hallucination | 40–60% | > 80% |
| Examples with > 30% hallucination | < 10% | > 25% |

---

*Document version: v1 — update after pilot calibration session*
