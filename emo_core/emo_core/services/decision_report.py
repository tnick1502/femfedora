from time import perf_counter
from uuid import uuid4

from emo_core.db.base import utcnow


class DecisionReport:
    """Executed evidence and a private prompt manifest; no raw chat, secrets or model reasoning."""

    def __init__(self, models):
        """Создаёт отчёт с Trace ID, снимком моделей и началом отсчёта времени."""
        self.started = perf_counter()
        self.data = {
            "version": "1.0",
            "trace_id": str(uuid4()),
            "created_at": utcnow().isoformat(),
            "models": models,
            "stages": [],
            "timings_ms": {},
        }

    def stage(self, name, status, reasons=None, **facts):
        """Добавляет статус, коды причин и факты выполненного этапа.

        Факты записываются как переданы: вызывающий код отвечает за отсутствие
        текста переписки, секретов и иных лишних персональных данных.
        """
        self.data["stages"].append(
            {
                "stage": name,
                "status": status,
                "reasons": reasons or [],
                "facts": facts,
                "elapsed_ms": self.elapsed(self.started),
            }
        )

    def elapsed(self, start):
        """Возвращает миллисекунды после start по perf_counter с округлением до сотых."""
        return round((perf_counter() - start) * 1000, 2)

    def finish(self, **result):
        """Добавляет итог и общую длительность в миллисекундах; возвращает накопленный отчёт."""
        self.data["result"] = result
        self.data["total_ms"] = self.elapsed(self.started)
        return self.data


def describe_models(settings):
    """Описывает модели режима live/mock/disabled из настроек без адресов и ключей."""
    mode = settings.ai_mode
    return {
        "ai_mode": mode,
        "revision": 0,
        "llm": {
            "provider": "openai_compatible" if mode == "live" else mode,
            "model": settings.llm_model
            if mode == "live"
            else "deterministic-support-v1"
            if mode == "mock"
            else None,
        },
        "jev": {
            "provider": "typesafe" if mode == "live" else mode,
            "model": settings.jev_model
            if mode == "live"
            else "deterministic-demo-v1"
            if mode == "mock"
            else None,
        },
    }
