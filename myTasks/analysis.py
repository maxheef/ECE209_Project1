import json
import re
import pandas as pd
from pathlib import Path
from IPython.display import display, Markdown

def parse_metrics_txt(path: Path):
    if not path.exists(): return None
    text = path.read_text()
    def g(name):
        m = re.search(rf"^{name}:\s*([0-9.]+)", text, re.MULTILINE)
        return float(m.group(1)) if m else 0.0
    return {'Accuracy': g('Accuracy'), 'Precision': g('Precision'), 'Recall': g('Recall'), 'F1': g('F1')}

def parse_metrics_json(path: Path):
    if not path.exists(): return None
    data = json.loads(path.read_text())
    return {
        'Accuracy': float(data.get('accuracy', 0.0)), 
        'Precision': float(data.get('precision', 0.0)), 
        'Recall': float(data.get('recall', 0.0)), 
        'F1': float(data.get('f1_score') or data.get('f1') or 0.0) 
    }

def show_comparison_table(output_dir, seed=55):
    out_dir = Path(output_dir)
    rows = []
    
    configs = [
        ('random', 'Regular', out_dir / f'metrics_random_regular_seed{seed}.txt', parse_metrics_txt),
        ('random', 'VCD', out_dir / f'metrics_random_vcd_seed{seed}.txt', parse_metrics_txt),
        ('random', 'MFCD', out_dir / 'mfcd_random.json', parse_metrics_json),
        ('popular', 'Regular', out_dir / f'metrics_popular_regular_seed{seed}.txt', parse_metrics_txt),
        ('popular', 'VCD', out_dir / f'metrics_popular_vcd_seed{seed}.txt', parse_metrics_txt),
        ('popular', 'MFCD', out_dir / 'mfcd_popular.json', parse_metrics_json),
    ]

    for split, method, path, parser in configs:
        m = parser(path)
        if m: rows.append({'Split': split, 'Method': method, **m})

    if not rows:
        print("No output files found.")
        return

    df = pd.DataFrame(rows)
    # Gain Calculation Logic
    df['Gain vs Prev (%)'] = 0.0
    for split in df['Split'].unique():
        idx = df[df['Split'] == split].index
        for i in range(1, len(idx)):
            prev_f1, curr_f1 = df.loc[idx[i-1], 'F1'], df.loc[idx[i], 'F1']
            if prev_f1 > 0:
                df.at[idx[i], 'Gain vs Prev (%)'] = ((curr_f1 - prev_f1) / prev_f1) * 100

    # Styling
    styled_df = df.style.format({
        'Accuracy': '{:.4f}', 'Precision': '{:.4f}', 'Recall': '{:.4f}', 
        'F1': '{:.4f}', 'Gain vs Prev (%)': '{:+.2f}%'
    }).map(lambda v: f'color: {"green" if v > 0 else "red"}; font-weight: bold', 
           subset=['Gain vs Prev (%)']).hide(axis='index')

    display(Markdown(f"### Regular vs VCD vs MFCD (Seed: {seed})"))
    display(styled_df)
