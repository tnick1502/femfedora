"""Independent export/ID/code checks for JNU 02; no clinical accuracy claims."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import nbformat

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.features.jnu_ctg_report_extraction import SOURCE,DEST,OUT,extract_report_features,sha256
from src.features.jnu_ctg_report_analysis import validate_exports


def validate():
    measurements=pd.read_parquet(DEST/'measurements.parquet')
    receipt=validate_exports(measurements)
    manifest=json.loads((OUT/'extraction_manifest.json').read_text())
    for relative,expected in manifest['artifacts_sha256'].items():
        assert sha256(ROOT/relative)==expected,relative
    splits=pd.read_parquet(SOURCE/'splits.parquet')
    index=pd.read_parquet(SOURCE/'signal_index.parquet')
    signals=np.load(SOURCE/'signals_original_float32.npy',mmap_mode='r')
    sample=splits.loc[splits.split=='train',['record_id']].sample(32,random_state=20261017)
    sample=sample.merge(index,on='record_id',validate='one_to_one')
    lookup=measurements.set_index('record_id')
    for row in sample.itertuples():
        actual=extract_report_features(signals[row.signal_row])
        for key,value in actual.items():
            np.testing.assert_allclose(lookup.loc[row.record_id,key],value,rtol=1e-10,atol=1e-10,equal_nan=True)
    # Both sensitivity files must contain only original train IDs; the prefixes
    # carry no inherited record-level FIGO target.
    train_ids=set(splits.loc[splits.split=='train','record_id'])
    for filename in ['sensitivity_train.parquet','duration_prefix_train.parquet']:
        frame=pd.read_parquet(OUT/filename)
        assert set(frame.record_id)<=train_ids
        assert not {'figo_label','neonatal_asphyxia'}&set(frame.columns)
    sample_groups=pd.read_parquet(OUT/'sensitivity_train_sample.parquet')
    assert len(sample_groups)==128 and sample_groups.patient_component.is_unique
    old=json.loads((ROOT/'research/experiments/jnu_ctg/preparation_v1/manifest.json').read_text())
    for relative,expected in old['artifacts_sha256'].items():
        assert sha256(ROOT/relative)==expected,relative
    old_presentation=json.loads((ROOT/'research/experiments/jnu_ctg/preparation_v1/presentation_receipt.json').read_text())
    old_book='research/notebooks/jnu_ctg/01_prepare_data.ipynb'
    assert sha256(ROOT/old_book)==old_presentation['sha256'][old_book]
    assert sha256(ROOT/'research/notebooks/jnu_ctg/history/ANALYSIS_PLAN_v1.md')==old_presentation['sha256']['research/notebooks/jnu_ctg/ANALYSIS_PLAN.md']
    frozen=json.loads((ROOT/'research/experiments/ctg_uci/audited_v2/validation_receipt.json').read_text())
    for relative,expected in frozen['artifact_sha256'].items():
        assert sha256(ROOT/relative)==expected,relative
    book=nbformat.read(ROOT/'research/notebooks/jnu_ctg/02_report_features.ipynb',as_version=4)
    nbformat.validate(book)
    code=[c for c in book.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    receipt.update(passed=True,source_row_reextraction_checked=32,sensitivity_only_train=True,
        prefix_labels_not_inherited=True,preparation_01_preserved=True,
        old_plan_archived_exactly=True,previous_UCI_artifacts_preserved=True,
        notebook_code_cells_executed=len(code),
        code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [
            ROOT/'src/features/jnu_ctg_report_extraction.py',ROOT/'src/features/jnu_ctg_report_analysis.py',
            ROOT/'src/features/ctg_russian_reports.py',ROOT/'tests/unit/test_jnu_ctg_report_extraction.py',
            Path(__file__).resolve()]})
    (OUT/'validation_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['artifacts_sha256','code_sha256']},ensure_ascii=False,indent=2))
    return receipt


if __name__=='__main__':validate()
