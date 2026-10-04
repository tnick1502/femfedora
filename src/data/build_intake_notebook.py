"""Build the executed-source notebook scaffold for the open-data intake."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = PROJECT_ROOT / "research" / "reports" / "open_data_intake.json"
NOTEBOOK_PATH = PROJECT_ROOT / "research" / "notebooks" / "01_open_data_intake.ipynb"


def build_summary(report: dict) -> str:
    cdc = report["prepared"]["cdc_natality"]
    uci = report["prepared"]["uci_cardiotocography"]
    ctu = report["prepared"]["ctu_uhb_ctg"]
    tpehg = report["prepared"]["tpehg"]
    cdc_rows = sum(cdc.get("rows_by_year", {}).values())
    return (
        "## tl;dr\n\n"
        f"- CDC/NCHS prepared baseline: **{cdc_rows:,}** birth records across 2023–2024.\n"
        f"- UCI CTG: **{uci.get('rows', 0):,}** rows; **{uci.get('exact_duplicate_rows', 0)}** exact duplicates.\n"
        f"- CTU-UHB: **{ctu.get('records', 0)}** indexed CTG records.\n"
        f"- TPEHG: **{tpehg.get('records', 0)}** indexed EHG records.\n"
        "- These sources support offline baselines and pipeline validation only; they do not establish clinical utility or Russian-population transportability."
    )


def main() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Open-data intake: pregnancy ML baselines\n\n"
            "Reproducible quality companion for the source archives and prepared tables. "
            "The notebook reads the generated intake report; it does not modify source data."
        ),
        nbf.v4.new_markdown_cell(build_summary(report)),
        nbf.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "### Key Assumptions\n\n"
            "- 2023 is the development period and 2024 is the untouched temporal test for CDC baselines.\n"
            "- The CDC analytic population is U.S. resident births (`RESTATUS` 1/2/3); foreign residents remain only in the raw archive.\n"
            "- CDC targets are birth-record proxy labels; no diagnosis timestamp is inferred.\n"
            "- CTU-UHB, UCI CTG and TPEHG are evaluated independently and are never row-joined.\n"
            "- Missing values remain missing; rates use the known-label denominator."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n\n"
            "plt.rcParams.update({\n"
            "    'figure.figsize': (10, 5),\n"
            "    'axes.spines.top': False,\n"
            "    'axes.spines.right': False,\n"
            "    'axes.grid': True,\n"
            "    'grid.alpha': 0.2,\n"
            "    'font.size': 10,\n"
            "})\n"
            "ROOT = Path.cwd()\n"
            "REPORT = json.loads((ROOT / 'research/reports/open_data_intake.json').read_text())\n"
            "PREVIEW_DIR = ROOT / 'tmp/notebooks'\n"
            "PREVIEW_DIR.mkdir(parents=True, exist_ok=True)"
        ),
        nbf.v4.new_markdown_cell("## Data\n\n### 1. Verify downloaded archives"),
        nbf.v4.new_code_cell(
            "archives = pd.DataFrame(REPORT['archives'])\n"
            "archives['size_mib'] = archives['size_bytes'] / 1024**2\n"
            "archives[['dataset_id', 'size_mib', 'source_valid', 'file_count', 'sha256']].round({'size_mib': 1})"
        ),
        nbf.v4.new_markdown_cell("## Results\n\n### 2. Target prevalence by source year"),
        nbf.v4.new_code_cell(
            "prevalence = pd.DataFrame(REPORT['prepared']['cdc_natality']['target_prevalence'])\n"
            "prevalence_table = prevalence.pivot(index='target', columns='source_year', values='prevalence')\n"
            "prevalence_table"
        ),
        nbf.v4.new_code_cell(
            "plot_targets = ['target_ptb37', 'target_ptb32', 'target_gdm', 'target_hdp_proxy']\n"
            "plot_data = prevalence_table.loc[plot_targets] * 100\n"
            "ax = plot_data.plot(kind='barh', color=['#315C8B', '#C98A2E'])\n"
            "ax.set_title('CDC/NCHS proxy-label prevalence: development vs temporal test')\n"
            "ax.set_xlabel('Known-label prevalence (%)')\n"
            "ax.set_ylabel('')\n"
            "ax.legend(title='Source year')\n"
            "for container in ax.containers:\n"
            "    ax.bar_label(container, fmt='%.2f%%', padding=3, fontsize=8)\n"
            "plt.tight_layout()\n"
            "plt.savefig(PREVIEW_DIR / 'target_prevalence.png', dpi=160, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "The chart compares like-for-like source fields by year. It is a drift check, not evidence that any target is predictable or clinically actionable."
        ),
        nbf.v4.new_markdown_cell("### 3. Missingness in baseline predictors"),
        nbf.v4.new_code_cell(
            "missingness = pd.DataFrame(REPORT['prepared']['cdc_natality']['feature_missingness'])\n"
            "missing_table = missingness.pivot(index='feature', columns='source_year', values='missing_rate')\n"
            "missing_table"
        ),
        nbf.v4.new_code_cell(
            "missing_plot = missing_table.sort_values(2024, ascending=True) * 100\n"
            "ax = missing_plot.plot(kind='barh', color=['#315C8B', '#C98A2E'])\n"
            "ax.set_title('Missingness of selected CDC baseline predictors')\n"
            "ax.set_xlabel('Missing values (%)')\n"
            "ax.set_ylabel('')\n"
            "ax.legend(title='Source year')\n"
            "plt.tight_layout()\n"
            "plt.savefig(PREVIEW_DIR / 'feature_missingness.png', dpi=160, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell("### 4. Signal and feature datasets"),
        nbf.v4.new_code_cell(
            "summary_rows = []\n"
            "for key in ['uci_cardiotocography', 'ctu_uhb_ctg', 'tpehg']:\n"
            "    item = REPORT['prepared'][key]\n"
            "    summary_rows.append({\n"
            "        'dataset': key,\n"
            "        'records': item.get('rows', item.get('records')),\n"
            "        'duplicates': item.get('exact_duplicate_rows', item.get('duplicate_record_ids')),\n"
            "        'missing_signal_files': item.get('missing_signal_files'),\n"
            "    })\n"
            "pd.DataFrame(summary_rows)"
        ),
        nbf.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "1. CDC/NCHS provides the scale needed for simple, temporally held-out tabular baselines, but not longitudinal PE/PTB modeling.\n"
            "2. CTU-UHB is the primary outcome-linked CTG benchmark; UCI CTG remains a feature/classification tutorial.\n"
            "3. TPEHG is suitable for signal-pipeline R&D, with uncertainty dominated by the small preterm event count.\n"
            "4. ICN and true FGR remain data-acquisition problems, not open-data ML tasks.\n"
            "5. Before training, duplicates/windows must be grouped by source record and all target-specific leakage exclusions must be enforced."
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
