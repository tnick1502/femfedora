"""Export all executed CTG notebooks, check links and hash the final handoff."""
from pathlib import Path
import base64,hashlib,json,sys,re,os,subprocess,importlib.metadata
from datetime import datetime,timezone
from urllib.parse import urlsplit,unquote,quote
import nbformat
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[1]
REVIEW=ROOT/'research/reviews/ctg_20261017'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def finalize():
    validation=json.loads((REVIEW/'validation_receipt.json').read_text());assert validation['passed']
    test_log=(ROOT/'tmp/ctg_review_all_unit_tests_final.log').read_text()
    assert re.search(r'79 passed',test_log),test_log[-1000:]
    books=sorted([* (ROOT/'research/notebooks/ctg_uci').glob('*.ipynb'),* (ROOT/'research/notebooks/jnu_ctg').glob('*.ipynb')])
    assert len(books)==12
    destinations={p.resolve():REVIEW/'html'/p.parent.name/(p.stem+'.html') for p in books}
    rows=[];retained=[]
    for path in books:
        book=nbformat.read(path,4);nbformat.validate(book);code=[c for c in book.cells if c.cell_type=='code']
        assert all(c.execution_count is not None for c in code),path
        assert not any(o.output_type=='error' for c in code for o in c.outputs),path
        pngs=[base64.b64decode(o.data['image/png']) for c in code for o in c.outputs if 'image/png' in o.get('data',{})]
        if path.name.startswith('07_'):assert len(re.findall(r'### Пункт \d+\.', '\n'.join(c.source for c in book.cells if c.cell_type=='markdown')))==23
        html,_=HTMLExporter(template_name='lab',exclude_input_prompt=True,exclude_output_prompt=True).from_notebook_node(book)
        target=destinations[path.resolve()];target.parent.mkdir(parents=True,exist_ok=True)
        soup=BeautifulSoup(html,'html.parser');assert soup.find('h1')
        assert len(soup.find_all('img'))==len(pngs),(path,len(pngs),len(soup.find_all('img')))
        for anchor in soup.find_all('a',href=True):
            original=anchor['href'];u=urlsplit(original)
            if u.scheme or u.netloc or not u.path:continue
            absolute=(path.parent/unquote(u.path)).resolve()
            assert absolute.exists(),(path,original,absolute)
            destination=destinations.get(absolute,absolute)
            relative=quote(os.path.relpath(destination,target.parent),safe='/')
            anchor['href']=relative+('?' + u.query if u.query else '')+('#'+u.fragment if u.fragment else '')
        target.write_text(str(soup))
        for anchor in soup.find_all('a',href=True):
            u=urlsplit(anchor['href'])
            if u.scheme or u.netloc or not u.path:continue
            resolved=(target.parent/unquote(u.path)).resolve()
            # Another HTML destination may be written later in this loop.
            assert resolved in set(p.resolve() for p in destinations.values()) or resolved.exists(),(path,anchor['href'])
        rows.append(dict(notebook=str(path.relative_to(ROOT)),executed_code_cells=len(code),figures=len(pngs),html=str(target.relative_to(ROOT))))
        if path.parent.name=='jnu_ctg' and path.name[:2] in ['01','02']:
            original=nbformat.read(REVIEW/'before_revision/jnu_ctg'/path.name,4)
            old=[base64.b64decode(o.data['image/png']) for c in original.cells if c.cell_type=='code' for o in c.outputs if 'image/png' in o.get('data',{})]
            assert all(a==b for a,b in zip(old,pngs)),path
            retained.append(dict(notebook=path.name,unchanged_previously_inspected_figures=len(old)))
    # Validate current documentation links in its native location too.
    docs=[]
    for folder in ['ctg_uci','jnu_ctg']:
        docs.extend((ROOT/'research/notebooks'/folder).glob('*.md'))
    for path in docs:
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            u=urlsplit(link)
            if u.scheme or u.netloc or not u.path:continue
            assert (path.parent/unquote(u.path)).resolve().exists(),(path,link)
    git=subprocess.run(['git','status','--short'],cwd=ROOT,text=True,capture_output=True,check=True).stdout
    (REVIEW/'git_status.txt').write_text(git)
    (REVIEW/'unit_test_receipt.json').write_text(json.dumps({'passed':79,'log':test_log.strip(),'files':[str(p.relative_to(ROOT)) for pattern in ['test_ctg_*.py','test_jnu_*.py'] for p in sorted((ROOT/'tests/unit').glob(pattern))]},indent=2))
    files=set(books+docs+[ROOT/'README.md',ROOT/'data/registry/open_datasets.json'])
    files.update(p for p in REVIEW.rglob('*') if p.is_file() and p.name!='presentation_receipt.json')
    for base in ['research/experiments/ctg_uci/reviewed_v3','research/experiments/jnu_ctg/reviewed_v2','models/exported/ctg_uci/reviewed_v3','models/exported/jnu_ctg/reviewed_v2']:
        files.update(p for p in (ROOT/base).rglob('*') if p.is_file())
    for pattern in ['src/training/ctg_review*.py','src/training/ctg_prefix_benchmark.py','src/training/jnu_review_experiment.py','src/features/ctg_report_contract.py','src/features/jnu_review*.py','tools/*ctg_review*.py','tools/build_*ctg*notebook*.py','tools/build_jnu*notebook*.py','tests/unit/test_ctg_*.py','tests/unit/test_jnu_*.py']:
        files.update(ROOT.glob(pattern))
    for relative in ['src/features/ctg_preprocessing.py','src/features/ctg_russian_reports.py','src/features/jnu_ctg_preparation.py','src/features/jnu_ctg_report_extraction.py','src/training/ctg_audit_folds.py','src/training/jnu_report_models.py','src/training/jnu_report_assessment.py','data/processed/jnu_ctg/review_v2/raw_minute_features.parquet','data/processed/ctg_uci/reviewed_v3/ctg_primary_grouped.parquet','models/exported/.history/jnu_ctg_reference_before_portability.zip']:
        files.add(ROOT/relative)
    environment={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','catboost','joblib','nbformat','nbconvert']}
    result=dict(passed=True,created_at_utc=datetime.now(timezone.utc).isoformat(),notebooks=rows,
        expert_review_points=23,unit_tests_passed=79,independent_model_validation_passed=True,
        revised_model_bundles=18,reference_fold_models=35,retained_figures=retained,
        new_figures_visually_inspected=[p.name for p in sorted((REVIEW/'figures').glob('*.png'))],
        notebook_figures=sum(r['figures'] for r in rows),html_structure_and_links_checked=True,
        html_browser_render_verified=False,html_limitation='Native PNG outputs inspected directly; retained 18 figures bit-identical to earlier inspected renders. Local HTML browser preview unavailable.',
        current_source_environment=environment,deterministic_seed=20261018,
        clinical_validation=False,new_independent_test=False,git_commit_created=False,
        sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(files) if p.is_file()})
    (REVIEW/'presentation_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('Final review validated:',len(rows),'executed notebooks,',result['notebook_figures'],'figures,',len(files),'hashed artifacts')
    return result


if __name__=='__main__':finalize()
