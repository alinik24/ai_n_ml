"""python scripts/integrate_results.py

Integrate latest results from results/ into REPORT.md and generate a final Word file.

Usage:
  1. Place the latest results in the repository under ./results with the following (recommended) structure:
     results/
       metrics/
         evaluation_results.csv    # per-agent summary rows (mean, std, success_rate, collision_rate, per-seed means optional)
         statistical_tests.csv     # optional: p-values, test names
         metadata.json             # optional: run metadata (date, git commit, seeds, episodes)
       figures/
         performance_comparison.png
         robust_statistics.png
         statistical_significance.png
         absolute_metrics.png
         radar_chart_baseline_normalized.png

  2. Install dependencies (if you want docx fallback):
       pip install pandas python-docx
     (If pandoc is available, the script will use pandoc to convert markdown to .docx with better formatting.)

  3. Run the script from repository root:
       python scripts/integrate_results.py

What the script does:
  - Reads REPORT.md
  - Replaces the entire "## 5. Experimental Results" section with a regenerated section based on files in results/metrics/
  - Writes REPORT_updated.md
  - If pandoc is installed, converts REPORT_updated.md -> REPORT_final.docx via pandoc
  - Otherwise, creates a basic REPORT_final.docx using python-docx and embeds available figures

Notes:
  - The script expects CSV files to contain certain columns. See README or contact me if your files differ; I can adapt the script.
  - If results files are not present, the script will abort and print helpful messages.

"""

import os
import sys
import json
import shutil
from pathlib import Path

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from docx import Document
    from docx.shared import Inches
except Exception:
    Document = None

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = REPO_ROOT / 'REPORT.md'
RESULTS_DIR = REPO_ROOT / 'results'
METRICS_DIR = RESULTS_DIR / 'metrics'
FIGURES_DIR = RESULTS_DIR / 'figures'


def load_evaluation_results(csv_path: Path):
    if not csv_path.exists():
        print(f"Expected evaluation results CSV at {csv_path} not found.")
        return None
    if pd is None:
        print("pandas is required to read CSV files. Install with: pip install pandas")
        return None
    df = pd.read_csv(csv_path)
    return df


def compose_results_section(df_eval: 'pd.DataFrame', df_tests: 'pd.DataFrame' = None, metadata: dict = None):
    """
    Build a markdown string for the Experimental Results section from dataframes.
    Expects df_eval to contain rows with Agent column, and numeric columns like mean_return, ci_low, ci_up,
    median, iqm, std, success_rate, success_ci_low, success_ci_up, avg_length, collision_rate.
    """
    lines = []
    lines.append('## 5. Experimental Results')
    lines.append('')
    if metadata:
        # Accept either 'evaluation_date' or 'date' keys
        eval_date = metadata.get('evaluation_date', metadata.get('date', 'N/A'))
        lines.append(f"**Evaluation Date:** {eval_date}")
        lines.append('')

    # Training configuration paragraph: keep as-is from original but appended summary of metrics
    lines.append('### 5.1 Training Configuration')
    lines.append('')
    lines.append('Refer to the Training Configuration in the report. Below we summarize the actual evaluation results from the latest runs.')
    lines.append('')

    lines.append('### 5.2 Evaluation Results')
    lines.append('')

    if df_eval is None:
        lines.append('_No evaluation results CSV provided._')
        return '\n'.join(lines)

    # Summary table header
    lines.append('| Agent | Mean Return | 95% CI | Median | IQM | Std | Success Rate | Success 95% CI | Avg Length | Collision Rate |')
    lines.append('|-------|-------------:|:-------:|------:|----:|----:|-------------:|:---------------:|-----------:|---------------:|')

    for _, row in df_eval.iterrows():
        agent = row.get('Agent', row.get('agent', 'Unknown'))

        # Handle different column namings and string-formatted values in the CSV
        # Mean return
        mean_raw = row.get('Mean Return', row.get('mean_return', ''))
        try:
            mean = f"{float(mean_raw):.1f}"
        except Exception:
            mean = str(mean_raw)

        # 95% CI may be provided as a string like "[-2763.63, -2473.18]"
        ci_raw = row.get('95% CI', row.get('ci', ''))
        ci = ''
        if isinstance(ci_raw, str) and ci_raw.startswith('[') and ',' in ci_raw:
            ci = ci_raw
        else:
            # fallback to separate columns
            ci_low = row.get('ci_low', '')
            ci_up = row.get('ci_up', '')
            try:
                if pd.notna(ci_low) and pd.notna(ci_up):
                    ci = f"[{float(ci_low):.1f}, {float(ci_up):.1f}]"
            except Exception:
                ci = ''

        # Median, IQM, Std
        def fmt_col(keys):
            for k in keys:
                v = row.get(k, None)
                if pd.notna(v):
                    try:
                        return f"{float(v):.1f}"
                    except Exception:
                        return str(v)
            return ''

        median = fmt_col(['Median', 'median'])
        iqm = fmt_col(['IQM', 'iqm'])
        std = fmt_col(['Std', 'std'])

        # Success rate and its CI
        succ_raw = row.get('Success Rate', row.get('success_rate', ''))
        succ_pct = ''
        if isinstance(succ_raw, str) and succ_raw.strip().endswith('%'):
            succ_pct = succ_raw.strip()
        else:
            try:
                if pd.notna(succ_raw):
                    val = float(succ_raw)
                    if val <= 1.0:
                        succ_pct = f"{val * 100:.2f}%"
                    else:
                        succ_pct = f"{val:.2f}%"
            except Exception:
                succ_pct = str(succ_raw)

        succ_ci_raw = row.get('Success 95% CI', row.get('success_95_ci', ''))
        succ_ci = ''
        if isinstance(succ_ci_raw, str) and succ_ci_raw.startswith('['):
            succ_ci = succ_ci_raw

        # Avg length
        avg_len_raw = row.get('Avg Length', row.get('avg_length', ''))
        try:
            avg_len = f"{float(avg_len_raw):.1f}"
        except Exception:
            avg_len = str(avg_len_raw) if avg_len_raw != '' else ''

        # Collision rate
        coll_raw = row.get('Collision Rate', row.get('collision_rate', ''))
        coll_pct = ''
        if isinstance(coll_raw, str) and coll_raw.strip().endswith('%'):
            coll_pct = coll_raw.strip()
        else:
            try:
                if pd.notna(coll_raw):
                    val = float(coll_raw)
                    if val <= 1.0:
                        coll_pct = f"{val * 100:.2f}%"
                    else:
                        coll_pct = f"{val:.2f}%"
            except Exception:
                coll_pct = str(coll_raw)

        lines.append(f"| **{agent}** | **{mean}** | {ci} | {median} | {iqm} | {std} | {succ_pct} | {succ_ci} | {avg_len} | {coll_pct} |")

    lines.append('')

    # Add statistical test summary if available
    if df_tests is not None:
        lines.append('#### Statistical Significance Tests')
        lines.append('')
        lines.append('| Test | Group A | Group B | p-value | Significant |')
        lines.append('|------|---------|---------:|--------:|:-----------:|')
        for _, r in df_tests.iterrows():
            # Flexible column handling
            test = r.get('test', r.get('Test', 'Mann-Whitney U'))
            g1 = r.get('group1', r.get('Group1', r.get('Agent 1', 'Q-Learning')))
            g2 = r.get('group2', r.get('Group2', r.get('Agent 2', 'DQN')))
            p = None
            for key in ['p_value', 'p-value', 'p', 'p_value', 'pvalue']:
                if key in r:
                    p = r.get(key)
                    break
            if pd.notna(p):
                try:
                    p_str = f"{float(p):.4f}"
                except Exception:
                    p_str = str(p)
            else:
                p_str = ''
            sig = 'Yes' if (p_str != '' and float(p_str) < 0.05) else 'No'
            lines.append(f"| {test} | {g1} | {g2} | {p_str} | {sig} |")
        lines.append('')

    # Mention available figures
    if FIGURES_DIR.exists():
        fig_list = sorted([p.name for p in FIGURES_DIR.glob('*.png')])
        if fig_list:
            lines.append('#### Figures')
            lines.append('')
            for f in fig_list:
                lines.append(f"![{f}](results/figures/{f})")
            lines.append('')

    lines.append('---')
    lines.append('\n')

    # Add per-seed breakdown if present in csv with per_seed_* columns
    # This is optional and not implemented in detail here.

    return '\n'.join(lines)


def main():
    if not REPORT_MD.exists():
        print(f"REPORT.md not found at {REPORT_MD}. Abort.")
        return

    eval_csv = METRICS_DIR / 'evaluation_results.csv'
    tests_csv = METRICS_DIR / 'statistical_tests.csv'
    metadata_json = METRICS_DIR / 'metadata.json'

    df_eval = None
    df_tests = None
    metadata = None

    if eval_csv.exists():
        if pd is None:
            print('pandas not installed. Install pandas to read CSV metrics. Aborting.')
            return
        df_eval = pd.read_csv(eval_csv)
    else:
        print(f'Warning: {eval_csv} not found. The script expects evaluation_results.csv in results/metrics/.')

    if tests_csv.exists():
        if pd is None:
            print('pandas not installed. Install pandas to read CSV metrics. Aborting.')
            return
        df_tests = pd.read_csv(tests_csv)

    if metadata_json.exists():
        try:
            metadata = json.loads(metadata_json.read_text())
        except Exception:
            metadata = None

    # Read REPORT.md and replace section 5
    try:
        # Read REPORT.md using utf-8 and fall back to latin-1 if necessary (handles Windows encoding issues)
        original = REPORT_MD.read_text(encoding='utf-8')
    except Exception:
        original = REPORT_MD.read_text(encoding='latin-1')
    start_marker = '\n## 5. Experimental Results'
    end_marker = '\n## 6. Discussion'

    if start_marker not in original or end_marker not in original:
        print('Could not locate Experimental Results section markers in REPORT.md. Aborting to avoid unintended edits.')
        return

    pre = original.split(start_marker)[0]
    post = original.split(end_marker)[1]

    new_results_md = compose_results_section(df_eval, df_tests, metadata)

    new_report = pre + start_marker + '\n' + new_results_md + '\n## 6. Discussion' + post

    updated_md_path = REPO_ROOT / 'REPORT_updated.md'
    # Write using utf-8 to avoid Windows encoding errors
    updated_md_path.write_text(new_report, encoding='utf-8')
    print(f'Wrote updated markdown to: {updated_md_path}')

    # Convert to docx
    pandoc_path = shutil.which('pandoc')
    docx_out = REPO_ROOT / 'REPORT_final.docx'

    if pandoc_path:
        print('pandoc found. Converting REPORT_updated.md -> REPORT_final.docx using pandoc...')
        cmd = f'"{pandoc_path}" "{updated_md_path}" -o "{docx_out}"'
        rc = os.system(cmd)
        if rc == 0:
            print(f'Generated Word file at: {docx_out}')
        else:
            print('pandoc conversion failed (non-zero exit).')
    else:
        print('pandoc not found. Falling back to python-docx minimal writer.')
        if Document is None:
            print('python-docx not installed. Install with: pip install python-docx')
            print(f'You can also install pandoc and rerun: https://pandoc.org')
            return

        doc = Document()
        # Add title
        doc.add_heading('Warehouse AMR Navigation Using Reinforcement Learning', level=1)

        # Add the updated markdown as plain text paragraphs (simple fallback)
        for line in new_report.splitlines():
            if line.startswith('# '):
                doc.add_heading(line.lstrip('# ').strip(), level=1)
            elif line.startswith('## '):
                doc.add_heading(line.lstrip('# ').strip(), level=2)
            elif line.startswith('### '):
                doc.add_heading(line.lstrip('# ').strip(), level=3)
            elif line.startswith('|'):
                # Add table lines as normal paragraphs (simpler)
                doc.add_paragraph(line)
            elif line.strip().startswith('!['):
                # image reference: ![name](path)
                try:
                    path = line.split('](')[1].rstrip(')')
                    img_path = REPO_ROOT / path
                    if img_path.exists():
                        doc.add_picture(str(img_path), width=Inches(6))
                        doc.add_paragraph(img_path.name)
                    else:
                        doc.add_paragraph(line)
                except Exception:
                    doc.add_paragraph(line)
            else:
                doc.add_paragraph(line)

        # Optionally embed all figures at the end
        if FIGURES_DIR.exists():
            fig_files = sorted(FIGURES_DIR.glob('*.png'))
            if fig_files:
                doc.add_page_break()
                doc.add_heading('Figures', level=2)
                for f in fig_files:
                    try:
                        doc.add_picture(str(f), width=Inches(6))
                        doc.add_paragraph(f.name)
                    except Exception as e:
                        doc.add_paragraph(f'Could not add figure {f.name}: {e}')

        doc.save(docx_out)
        print(f'Generated Word file at: {docx_out} (basic formatting)')


if __name__ == '__main__':
    main()
