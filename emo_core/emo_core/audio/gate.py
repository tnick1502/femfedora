"""Cheap eligibility checks, not an emotional classifier or a clinical rule."""

import re

WORDS = re.compile(r"[а-яёa-z]+", re.I)
FILLERS = frozenset(
    "привет здравствуйте здравствуй спасибо пожалуйста пока добрый доброе день утро вечер да нет ага угу ок окей хорошо ладно алло раз два три проверка микрофона меня слышно как дела ты вы там тут ну вот это и а спасибо большое".split()
)
ROUTINE = re.compile(
    r"^(?:который час|сколько (?:сейчас )?времени|какая сегодня (?:дата|погода)|"
    r"где (?:найти|находятся|открыть) (?:настройки|опросники|тесты)|"
    r"как (?:открыть|настроить|выключить|включить) (?:настройки|микрофон|голос|тему)|"
    r"что ты (?:можешь|умеешь))"
    r"(?: (?:в приложении|на телефоне|пожалуйста|сейчас|у вас|мне))*$"
)


def emotion_gate(text, seconds):
    words = WORDS.findall(text.lower())
    evidence = {"version": "voice-gate-1", "words": len(words), "seconds": round(seconds, 1)}
    if seconds < 3 or len(words) < 4 or sum(map(len, words)) < 16:
        reason = "short_speech_or_transcript"
    elif set(words) <= FILLERS or len(set(words)) < 3:
        reason = "greeting_acknowledgement_or_mic_check"
    elif ROUTINE.fullmatch(" ".join(words)):
        reason = "routine_interface_or_factual_request"
    else:
        reason = "substantive_speech"
    return {**evidence, "eligible": reason == "substantive_speech", "reason": reason}
