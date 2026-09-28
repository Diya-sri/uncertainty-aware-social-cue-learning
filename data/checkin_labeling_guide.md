# Check-in feedback: labeling guide

Each message is something a learner might say to another person after noticing a social cue.
It is labeled on four independent binary criteria. The overall verdict is derived, never labeled directly.

| Criterion | True when the message… | Examples (true) | Examples (false) |
|---|---|---|---|
| `assumes_feeling` | states or presupposes the other person's inner state, motive or intent as fact | "You look sad." · "Why are you so angry?" · "You're ignoring me." · "You must be tired." | "Are you okay?" · "You seem quiet. Is that right, or am I misreading?" · "I don't want to guess how you feel." |
| `judgmental` | labels, blames, mocks, dismisses or minimises the person or their reaction (including sarcasm) | "Why are you being weird?" · "Cheer up, it's not a big deal." · "Wow, great job listening." | "The plan changed fast." · "That sounds hard." |
| `pressuring` | demands disclosure or action, removes the option to decline, or issues an ultimatum | "Tell me what happened." · "Look at me when I talk." · "Admit it." · "Answer me now." | "Would you like to talk?" · "No pressure, I'm around later." |
| `checks_in` | genuinely asks, or offers support/options the person can accept or decline | "Want to take a break?" · "I'm here if you want to talk later." | "You look sad." · "Stop wasting time." |

**Derived verdict:** `respectful = checks_in AND NOT assumes_feeling AND NOT judgmental AND NOT pressuring`.

## Edge-case rules

1. **Hedged observation + confirmation question is not an assumption.** "You seem upset — did I get that right?" → `assumes_feeling=false`.
   A hedge alone, without asking, still assumes: "You seem upset, so let's stop." → `true`.
2. **Presupposing questions assume.** "Why are you sad?" presupposes sadness → `assumes_feeling=true`, even though it is a question. It still `checks_in` only if it leaves room to answer freely; presupposing "why" questions are labeled `checks_in=false`.
3. **Describing an observable is not assuming.** "You covered your ears when it got loud." → `false`.
4. **Questions phrased as commands pressure.** "Can you just tell me what's wrong already?" → `pressuring=true`.
5. **Sarcasm is judgmental** regardless of its literal words.
6. **Reassurance that dismisses is judgmental.** "Don't be sad, it's fine." → `judgmental=true`.
7. **Talking about oneself is fine.** "I felt confused when the plan changed. Did you?" → no assumption.

## Provenance and limitations

All messages were written and labeled by the project author, a single annotator, so no inter-annotator agreement
has been measured. The `dev` split was used while writing the rule-based baseline. The `test` split was written
before the baseline rules and was not used to tune them. However, the same person wrote both, so the test split is
not fully independent. A second annotator relabeling the test split is the most valuable next step.
