"""Everyday communication practice for the Notice → Consider → Ask activity.

There is deliberately no emotion ground-truth here. Each situation separates observable
facts from possible explanations and rewards a response that preserves the other person's
agency. This is educational content, not therapy or clinical assessment.
"""

COMMUNICATION_SCENARIOS = [
    {
        "situation": "Your friend is looking down and answering with only one or two words.",
        "notice": "They are looking down and giving shorter answers than usual.",
        "possibilities": ["They may be tired.", "Something may be bothering them.", "They may want quiet time."],
        "responses": [
            "You look sad. Tell me what happened.",
            "Would you like to talk, have quiet time, or do something together?",
            "Stop ignoring me.",
        ],
        "best": 1,
        "why": "It checks instead of assuming and gives the person comfortable choices.",
    },
    {
        "situation": "A classmate covers their ears when the room becomes noisy.",
        "notice": "They cover their ears after the noise gets louder.",
        "possibilities": ["The sound may hurt or overwhelm them.", "They may need a break.", "They may be concentrating."],
        "responses": [
            "Why are you being weird?",
            "Ignore them because they are probably angry.",
            "Is the noise uncomfortable? We can move somewhere quieter if you want.",
        ],
        "best": 2,
        "why": "It names the observable situation and offers support without judging them.",
    },
    {
        "situation": "Someone does not make eye contact while you are explaining a game.",
        "notice": "They are looking away while listening.",
        "possibilities": ["Looking away may help them listen.", "Eye contact may feel uncomfortable.", "They may not understand yet."],
        "responses": [
            "Are the instructions clear, or would a demonstration help?",
            "Look at me when I am talking.",
            "You are not listening.",
        ],
        "best": 0,
        "why": "Eye contact is not proof of attention. A neutral check makes the instructions more accessible.",
    },
    {
        "situation": "Your teammate becomes quiet after the plan suddenly changes.",
        "notice": "They stop contributing after an unexpected change.",
        "possibilities": ["They may need time to adjust.", "They may disagree.", "They may be thinking through the new plan."],
        "responses": [
            "They are being difficult, so continue without them.",
            "The change was sudden. Would a minute to review the new plan help?",
            "Cheer up—the new plan is better.",
        ],
        "best": 1,
        "why": "It acknowledges the change and offers processing time without assigning an emotion.",
    },
    {
        "situation": "A friend says, ‘I’m fine,’ but their voice is quieter than usual.",
        "notice": "They say they are fine and speak quietly.",
        "possibilities": ["They may truly be fine.", "They may not want to talk now.", "They may appreciate a later check-in."],
        "responses": [
            "You are clearly upset. Admit it.",
            "Okay. I’m here if you want company or want to talk later.",
            "Tell everyone something is wrong.",
        ],
        "best": 1,
        "why": "Their words have priority. The response respects them while leaving support available.",
    },
    {
        "situation": "A group member keeps arranging the materials before starting the task.",
        "notice": "They organize the materials repeatedly before beginning.",
        "possibilities": ["Organizing may help them feel ready.", "The instructions may be unclear.", "They may have a useful system."],
        "responses": [
            "Would a checklist help, or should we divide the first step together?",
            "Stop wasting time.",
            "Take the materials away from them.",
        ],
        "best": 0,
        "why": "It offers structure and collaboration instead of treating a different work style as a problem.",
    },
]
