"""Create a local portable research/demo bundle, including public input data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

from src.ring_health.data import RESULTS,ROOT,save_json
from src.ring_health.predict import predict


def main():
    registry=json.loads((ROOT/"models/registry/ring_health_v1.json").read_text())
    for item in registry["artifacts"].values():
        assert hashlib.sha256((ROOT/item["path"]).read_bytes()).hexdigest()==item["sha256"]
    api_path=Path("/private/tmp/ring_api_response.json")
    if api_path.exists():
        expected=predict(json.loads((ROOT/"data/samples/ring_health/fertility_demo_input.json").read_text()))
        assert expected==json.loads(api_path.read_text()),"HTTP and direct inference differ"
    roots=["src/ring_health","data/external/ring_health","data/processed/ring_health","data/samples/ring_health","research/experiments/ring_health/v1","models/exported/ring_health/v1"]
    files=[]
    for root in roots:
        files += [p for p in (ROOT/root).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name!="package_manifest.json"]
    files += [ROOT/x for x in ["src/__init__.py","requirements-ring-health.txt","models/registry/ring_health_v1.json","models/configs/training/ring_health_simulator_v1.json","docs/ring_health_input_contract_ru.md","research/reports/ring_health_mvp_ru.md","research/reports/ring_health_metrics_ru.md","data/registry/ring_health_sources.json","data/registry/ring_health_downloads.json","tests/unit/test_ring_health.py"]]
    files += list((ROOT/"tools").glob("*ring_health*.py"))
    files=sorted(set(files))
    output=ROOT/"output/ring_health_mvp_v1.zip"
    output.parent.mkdir(exist_ok=True)
    instructions="""Ring health MVP v1 — 9 ноября 2026

Это локальный исследовательский пакет: две задачи анализа кольца, 8 экспортированных моделей,
публичные исходные и подготовленные данные, синтетические сценарии, метрики и примеры.
Модели кольца обучены только на синтетике. Точность на реальных данных Oura неизвестна.
Будущий риск/время до менопаузы не предсказываются. Рекомендации не формируются.

Откройте research/reports/ring_health_mvp_ru.md и docs/ring_health_input_contract_ru.md.

После распаковки перейдите в ring_health_mvp_v1 и создайте отдельное окружение Python 3.12:

python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-ring-health.txt
.venv/bin/python -m src.ring_health.predict --input data/samples/ring_health/fertility_demo_input.json
PYTHONPATH=. .venv/bin/python tools/serve_ring_health.py --port 8787

GET http://127.0.0.1:8787/health
POST http://127.0.0.1:8787/v1/analyze с телом JSON из примера.

Обучение: .venv/bin/python -m src.ring_health.train
Проверки: .venv/bin/python -m pytest tests/unit/test_ring_health.py -q
Подготовленные CSV и бинарные артефакты включены; сеть для демонстрации не нужна после установки зависимостей.
Каталог источников и лицензий: data/registry/ring_health_sources.json.
Архив содержит открытые обезличенные медицинские данные; не добавляйте весь архив в обычный Git.
"""
    with ZipFile(output,"w",ZIP_DEFLATED,compresslevel=6) as archive:
        archive.writestr("ring_health_mvp_v1/START_RU.txt",instructions)
        for file in files:archive.write(file,"ring_health_mvp_v1/"+str(file.relative_to(ROOT)))
        assert archive.testzip() is None
    save_json(RESULTS/"package_manifest.json",{"path":str(output.relative_to(ROOT)),"files":len(files)+1,"bytes":output.stat().st_size,"sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"http_matches_direct_inference":api_path.exists()})
    print(json.dumps({"path":str(output),"files":len(files)+1,"megabytes":round(output.stat().st_size/1e6,2)},indent=2))


if __name__=="__main__":main()
