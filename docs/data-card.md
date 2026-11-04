# Data card

Second Ear is evaluated on five sets. Every test label was made by people under a published protocol and a public license; no model wrote or pre-filled a test label. The dev set is the exception: Claude wrote and labeled it, and its results are reported as dev results only.

| Set | What | Labels by | Size | Used for | Lock |
|---|---|---|---|---|---|
| Dev | 36 scripted mock hearings in es, pt, zh, hi, ar, ta, voiced with open TTS (Arabic text only) | Claude (AI-written, disclosed) | 773 renditions: 251 planted errors in 15 types, 108 acceptable paraphrases | Development, ablations, the 6 error types the human sets lack | none (development data) |
| T1 | Text pairs with MQM error annotations, plus up to 2 preceding segments as context | Professional translators and language experts | 1,040 pairs (520 error, 520 clean): 832 test, 208 calibration | Judge precision and recall vs a one-prompt baseline | `eval/locks/human_test.json` |
| T3 | Clean interpreted sessions: a FLEURS speaker in the source language, then a FLEURS English speaker reading the professional FLORES translation | Clean by construction (professional translation, human voices) | 303 pairs, 2.0 h, 6 languages | False alarms per clean hour | `eval/locks/clean_sessions.json` |
| T4 | The T1 pairs voiced as long interpreted sessions, one per language (speaker turn, then interpreter turn) | T1's human labels; speech is TTS | test: 832 pairs in 4 sessions of 52 to 201 minutes (6.7 h); calibration: 208 pairs (1.6 h) | End-to-end audio recall on human labels, long-audio baselines | `eval/locks/t4_test.json` |

## Sources and licenses

| Source | What people labeled | Languages | License |
|---|---|---|---|
| [Google MQM](https://github.com/google/wmt-mqm-human-evaluation), WMT newstest2020 | Real errors by human translators, each segment rated by 3 professional translators | zh→en | Apache-2.0 |
| Google MQM, WMT24 | MT errors (15 systems and a human reference), 1 professional rater per segment, incl. a speech domain | en→es | Apache-2.0 |
| [IndicMT Eval](https://github.com/ai4bharat/indicmt-eval) (ACL 2023) | MT errors (7 systems), MQM spans by language experts | en→hi, en→ta | CC BY-SA 4.0 |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) + [FLORES-200](https://github.com/facebookresearch/flores) | Human speech; professional human translations | all 6 + en | CC BY 4.0; CC BY-SA 4.0 |

Third-party text and audio are never committed. The repo carries item IDs, labels, text hashes, and the scripts that rebuild each set from pinned sources (`scripts/build_human_test.py`, `scripts/build_clean_sessions.py`, `scripts/build_t4.py`).

## Label mapping (T1)

| Source label | Our label |
|---|---|
| MQM Major Accuracy/Mistranslation, or IndicMT High/Very High | distortion |
| Accuracy/Omission | omission |
| Accuracy/Addition | addition |
| No accuracy error of any severity, and no Major error | clean |
| Minor accuracy errors, rater splits, untranslated text | left out of both classes |

With 3 raters, at least 2 must mark a Major accuracy error.

## Protocol

- Label rules live in code with unit tests, written before any pipeline run.
- Sampling is deterministic (seed 2026). Each lock is the SHA-256 of the item list; a fresh clone reproduces it.
- The calibration splits may tune thresholds. The test splits run once, at the freeze, and the scripts refuse them without `--locked-test-run`.
- Results are reported on gold transcripts and on ASR output separately.

## Not used

MockConf (real student interpreting with professional error labels) is CC BY-NC-SA 4.0, and a prize entry for a future product is arguably commercial use. TED-based MQM subsets are CC BY-NC-ND. MQM files with automatic prior ratings are AI-assisted. The TIPp corpus of real criminal trials is not released.

## What the human sets can't test

Hedges, pronoun shifts, softening, yes/no reversals, wrong questions, self-corrections, and uninterpreted exchanges have dev labels only (AI-written). Arabic and Portuguese have no human-labeled test data.

## Limitations

Our test labels come from public annotations by professional translators and language experts, not from interpreters reviewing hearing audio. Most of those errors occur in written news or machine translation, not live interpretation, because no permissively licensed, human-labeled interpreting data exists. Arabic and Portuguese have no human-labeled test data. A production service should validate on interpreted hearings labeled by certified court interpreters, with two independent annotators per hearing. We expect those manual labels to give more reliable, and likely lower, accuracy estimates than ours.
