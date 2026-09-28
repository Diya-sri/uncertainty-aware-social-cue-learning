"""Write the hand-labeled check-in feedback benchmark to data/checkin_eval.jsonl.

Labels follow data/checkin_labeling_guide.md. Flags: A = assumes_feeling, J = judgmental,
P = pressuring, C = checks_in. The overall `respectful` verdict is derived, never hand-labeled.

    python scripts/build_checkin_eval.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (category, message, flags)
DEV = [
    ("clear_respectful", "Would you like to talk, have quiet time, or do something together?", "C"),
    ("clear_respectful", "Is the noise uncomfortable? We can move somewhere quieter if you want.", "C"),
    ("clear_respectful", "Are the instructions clear, or would a demonstration help?", "C"),
    ("clear_respectful", "The change was sudden. Would a minute to review the new plan help?", "C"),
    ("clear_respectful", "Okay. I'm here if you want company or want to talk later.", "C"),
    ("clear_respectful", "Would a checklist help, or should we divide the first step together?", "C"),
    ("clear_respectful", "Hey, how are you doing?", "C"),
    ("clear_respectful", "Do you want me to stay or give you some space?", "C"),
    ("clear_respectful", "Is there anything I can do to help?", "C"),
    ("clear_respectful", "Want to take a short break before we continue?", "C"),
    ("clear_assumption", "You look sad. Tell me what happened.", "AP"),
    ("clear_assumption", "You're clearly upset.", "A"),
    ("clear_assumption", "I can tell you're angry with me.", "A"),
    ("clear_assumption", "You must be really tired today.", "A"),
    ("clear_assumption", "You're bored, so let's stop.", "A"),
    ("clear_assumption", "You're nervous about the test, right? Don't worry.", "AJ"),
    ("judgmental", "Why are you being weird?", "J"),
    ("judgmental", "Stop wasting time.", "JP"),
    ("judgmental", "They are being difficult, so continue without them.", "AJ"),
    ("judgmental", "Cheer up—the new plan is better.", "J"),
    ("judgmental", "It's not a big deal, relax.", "JP"),
    ("judgmental", "You're overreacting.", "AJ"),
    ("pressure", "Look at me when I am talking.", "P"),
    ("pressure", "You are clearly upset. Admit it.", "AP"),
    ("pressure", "Tell me what's wrong right now.", "P"),
    ("pressure", "You need to explain why you left.", "P"),
    ("pressure", "Answer me.", "P"),
    ("presupposing_question", "Why are you so angry?", "A"),
    ("presupposing_question", "What's making you so sad?", "A"),
    ("presupposing_question", "Why are you ignoring me?", "AJ"),
    ("hedged_check", "You seem a bit quiet. Is that right, or am I misreading?", "C"),
    ("hedged_check", "I might be wrong, but you seem frustrated. Is that close?", "C"),
    ("hedged_check", "Maybe the noise is a lot right now? Tell me if I'm off.", "C"),
    ("negation", "I don't want to guess how you feel. How's it going?", "C"),
    ("negation", "I'm not saying you're upset, just checking in.", "C"),
    ("observation", "You covered your ears when it got loud. Do you want to step outside?", "C"),
    ("observation", "I noticed you stopped talking after the plan changed. Want to go over it again?", "C"),
    ("sarcasm", "Wow, great job listening.", "J"),
    ("sarcasm", "Oh sure, take your time, we have all day.", "J"),
    ("polite_pressure", "Can you just tell me what's wrong already?", "JP"),
    ("polite_pressure", "Please explain yourself.", "P"),
    ("dismissive_reassurance", "Don't be sad, it's fine.", "AJ"),
    ("self_disclosure", "I felt confused when the plan changed. Did you?", "C"),
    ("minimal", "You okay?", "C"),
    ("minimal", "Hi.", ""),
]

TEST = [
    # clear respectful
    ("clear_respectful", "Do you want to keep going, or should we pause for a bit?", "C"),
    ("clear_respectful", "If you'd like, I can explain the game again in a different way.", "C"),
    ("clear_respectful", "Would it help to write down the steps?", "C"),
    ("clear_respectful", "No pressure at all, but I'm around if you want to chat.", "C"),
    ("clear_respectful", "How would you like to handle the new plan?", "C"),
    ("clear_respectful", "Is now a good time, or would later be better?", "C"),
    ("clear_respectful", "Can I get you anything?", "C"),
    ("clear_respectful", "Would you rather work together or on your own for this part?", "C"),
    ("clear_respectful", "What would make this easier for you?", "C"),
    ("clear_respectful", "Let me know if you want to swap seats somewhere quieter.", "C"),
    ("clear_respectful", "Should I slow down a bit?", "C"),
    ("clear_respectful", "Happy to help if you want, just say the word.", "C"),
    # clear assumption
    ("clear_assumption", "You're obviously stressed out.", "A"),
    ("clear_assumption", "I know you're mad at me.", "A"),
    ("clear_assumption", "You look scared.", "A"),
    ("clear_assumption", "You're upset, so I'll leave you alone.", "A"),
    ("clear_assumption", "You're jealous of her, aren't you.", "AJ"),
    ("clear_assumption", "You seem annoyed, so let's stop.", "A"),
    ("clear_assumption", "You're anxious about tomorrow.", "A"),
    ("clear_assumption", "You're happy now, great.", "A"),
    ("clear_assumption", "You're feeling left out.", "A"),
    ("clear_assumption", "You don't care about this project.", "AJ"),
    # judgmental
    ("judgmental", "That's a silly thing to be upset about.", "AJ"),
    ("judgmental", "Stop being so dramatic.", "JP"),
    ("judgmental", "What is wrong with you?", "J"),
    ("judgmental", "Everyone else managed fine.", "J"),
    ("judgmental", "You're so lazy.", "J"),
    ("judgmental", "Grow up.", "JP"),
    ("judgmental", "Just get over it.", "JP"),
    ("judgmental", "That's a weird way to do it.", "J"),
    ("judgmental", "Honestly it's not that loud, you're fine.", "AJ"),
    ("judgmental", "Smile more, it's a party!", "JP"),
    # pressure
    ("pressure", "Tell me what happened.", "P"),
    ("pressure", "You have to talk to me.", "P"),
    ("pressure", "Explain it to me now.", "P"),
    ("pressure", "Stop and look at me.", "P"),
    ("pressure", "You need to join the group, no excuses.", "JP"),
    ("pressure", "Say something!", "P"),
    ("pressure", "We're not leaving until you tell me.", "P"),
    ("pressure", "Put the materials down and start the task.", "P"),
    # presupposing questions
    ("presupposing_question", "Why are you so upset?", "A"),
    ("presupposing_question", "Why are you mad at me?", "A"),
    ("presupposing_question", "What are you so nervous about?", "A"),
    ("presupposing_question", "Why don't you like the plan?", "A"),
    ("presupposing_question", "Why are you so stressed?", "A"),
    ("presupposing_question", "Why are you angry all the time?", "AJ"),
    ("presupposing_question", "Why are you avoiding me?", "A"),
    # hedged check
    ("hedged_check", "You seem tired. Am I reading that right?", "C"),
    ("hedged_check", "I could be totally wrong, but is something bothering you?", "C"),
    ("hedged_check", "It looks like the noise might be a lot. Want to take a break?", "C"),
    ("hedged_check", "I'm guessing the change was unexpected. Is that fair?", "C"),
    ("hedged_check", "You seem quieter than usual. Everything okay?", "C"),
    ("hedged_check", "Maybe you'd prefer some quiet time? Totally fine either way.", "C"),
    ("hedged_check", "Not sure if I'm right, but you seem a little down. Want to talk?", "C"),
    # negation
    ("negation", "I'm not assuming anything, I just wanted to see how you are.", "C"),
    ("negation", "I don't know how you're feeling, so I wanted to ask.", "C"),
    ("negation", "You don't have to answer, but how are you doing?", "C"),
    ("negation", "You don't need to explain anything. Want some water?", "C"),
    ("negation", "I'm not upset with you. Do you want to keep playing?", "C"),
    ("negation", "No need to talk if you'd rather not.", "C"),
    # observation
    ("observation", "You haven't said much since lunch. Want to hang out?", "C"),
    ("observation", "I saw you step out of the room. Do you need anything?", "C"),
    ("observation", "You're looking away while I explain. Would a demo work better?", "C"),
    ("observation", "The room got really loud just now. Want to move?", "C"),
    ("observation", "You've rearranged the materials a few times. Want to start together?", "C"),
    ("observation", "You said 'I'm fine' quietly. I'm here if that changes.", "C"),
    # sarcasm
    ("sarcasm", "Nice of you to finally join us.", "J"),
    ("sarcasm", "Oh great, another meltdown.", "AJ"),
    ("sarcasm", "Thanks so much for all your help, really.", "J"),
    ("sarcasm", "Wow, you're in a lovely mood today.", "AJ"),
    ("sarcasm", "Sure, because that makes total sense.", "J"),
    # polite pressure
    ("polite_pressure", "Could you please just look at me?", "P"),
    ("polite_pressure", "I'd really appreciate it if you told me what's going on right now.", "P"),
    ("polite_pressure", "Please just answer the question.", "P"),
    ("polite_pressure", "Can you stop that and pay attention?", "JP"),
    ("polite_pressure", "Kindly explain why you did that.", "P"),
    # dismissive reassurance
    ("dismissive_reassurance", "Don't worry, it's nothing.", "J"),
    ("dismissive_reassurance", "Don't be scared, it's just noise.", "AJ"),
    ("dismissive_reassurance", "No reason to be nervous!", "AJ"),
    ("dismissive_reassurance", "Don't cry, it's not a big deal.", "JP"),
    # self disclosure
    ("self_disclosure", "I was surprised by the change too. How about you?", "C"),
    ("self_disclosure", "I get overwhelmed in loud rooms sometimes. Want to step out with me?", "C"),
    ("self_disclosure", "I'm a bit lost with the new rules. Should we look at them together?", "C"),
    # minimal / other
    ("minimal", "Everything alright?", "C"),
    ("minimal", "Need anything?", "C"),
    ("minimal", "Okay.", ""),
    ("minimal", "Let's start.", ""),
    ("minimal", "Here's the plan for today.", ""),
    # mixed: good content with a problem mixed in
    ("mixed", "You look upset. Do you want to talk?", "AC"),
    ("mixed", "You're clearly stressed, want to take a break?", "AC"),
    ("mixed", "Stop being weird and tell me if you need help.", "JP"),
    ("mixed", "Do you want a break? Answer me.", "P"),
    ("mixed", "I know you're annoyed, but can we finish this part together?", "AC"),
]


def to_record(i: int, split: str, category: str, message: str, flags: str) -> dict:
    labels = {
        "assumes_feeling": "A" in flags,
        "judgmental": "J" in flags,
        "pressuring": "P" in flags,
        "checks_in": "C" in flags,
    }
    labels["respectful"] = labels["checks_in"] and not (
        labels["assumes_feeling"] or labels["judgmental"] or labels["pressuring"])
    return {"id": f"{split}-{i:03d}", "split": split, "category": category, "message": message, "labels": labels}


def main() -> None:
    records = [to_record(i, "dev", *row) for i, row in enumerate(DEV)]
    records += [to_record(i, "test", *row) for i, row in enumerate(TEST)]
    ids = [r["message"] for r in records]
    assert len(ids) == len(set(ids)), "duplicate messages"
    out = ROOT / "data" / "checkin_eval.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    print(f"wrote {len(records)} messages ({len(DEV)} dev / {len(TEST)} test) -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
