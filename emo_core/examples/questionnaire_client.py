"""Interactive API client. Run from the independent EMO Core Poetry environment."""

import argparse
import os
from pathlib import Path
from uuid import uuid4

import httpx
from dotenv import load_dotenv


def main():
    """Запускает консольное прохождение шкалы через REST API с сохранением каждого ответа.

    Читает EMO_API_URL и EMO_API_TOKEN из examples/client.env. --attempt продолжает
    черновик, --code выбирает новую шкалу. q завершает клиент; сохранённые ответы
    остаются на сервере. Итог считает сервер после явного завершения.
    """
    load_dotenv(Path(__file__).with_name("client.env"))
    parser = argparse.ArgumentParser(description="Пройти опросник через EMO REST API")
    parser.add_argument("--code", choices=["GAD2", "GAD7", "EPDS", "PHQ9"], default="GAD2")
    parser.add_argument("--attempt", help="UUID сохранённого черновика")
    args = parser.parse_args()
    token = os.getenv("EMO_API_TOKEN")
    if not token:
        raise SystemExit("Заполните EMO_API_TOKEN в examples/client.env")
    base = os.getenv("EMO_API_URL", "http://127.0.0.1:8086").rstrip("/") + "/api/v1/emo"
    with httpx.Client(
        base_url=base + "/", headers={"Authorization": "Bearer " + token}, timeout=30
    ) as client:

        def call(method, path, **kwargs):
            """Выполняет запрос API и читает JSON; при ошибке завершает клиент с кодом HTTP и detail."""
            response = client.request(method, path, **kwargs)
            if not response.is_success:
                raise SystemExit(f"API: {response.status_code}; {response.json().get('detail')}")
            return response.json()

        if args.attempt:
            attempt = call("GET", "questionnaire-attempts/" + args.attempt)
        else:
            material = next(
                q for q in call("GET", "questionnaires")["items"] if q["code"] == args.code
            )
            attempt = call(
                "POST",
                "questionnaire-attempts",
                json={"questionnaire_id": material["id"], "client_request_id": str(uuid4())},
            )
        print("Попытка:", attempt["id"])
        q = attempt["questionnaire"]
        print(q["definition"]["title"], "\n", q["definition"]["instructions"])
        print(q["definition"]["attribution"])
        path = "questionnaire-attempts/" + attempt["id"]
        if attempt["status"] != "completed":
            groups = [
                ("answers", q["definition"]["items"]),
                ("auxiliary_answers", q["definition"].get("auxiliary_items", [])),
            ]
            for field, items in groups:
                for item in items:
                    if item["id"] in attempt[field]:
                        continue
                    optional = field == "auxiliary_answers" and not item.get("required")
                    print("\n", item["id"], item["text"])
                    if field == "auxiliary_answers":
                        print("Дополнительный вопрос, в сумму не входит.")
                    for index, option in enumerate(item["options"]):
                        print(f"{index}: {option}")
                    while True:
                        prompt = "Индекс 0–3; q — сохранить и выйти"
                        if optional:
                            prompt += "; Enter — пропустить"
                        value = input(prompt + ": ").strip()
                        if value.lower() == "q":
                            return
                        if value in ("0", "1", "2", "3") or (optional and not value):
                            break
                    if not value:
                        continue
                    attempt = call(
                        "PATCH",
                        path,
                        json={
                            "expected_revision": attempt["revision"],
                            field: {item["id"]: int(value)},
                        },
                    )
                    if attempt["safety_guidance"]:
                        print(attempt["safety_guidance"])
            attempt = call(
                "POST", path + "/complete", json={"expected_revision": attempt["revision"]}
            )
        print("Сумма:", attempt["total_score"], "/", q["scoring_rules"]["maximum"])
        if attempt["safety_guidance"]:
            print(attempt["safety_guidance"])
        if attempt["result"]:
            print(attempt["result"]["next_step"])
        print("Самооценка симптомов, не диагноз.")


if __name__ == "__main__":
    main()
