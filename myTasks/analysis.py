import json
import pandas as pd
from pathlib import Path
from IPython.display import display, Markdown

def parse_metrics_json(path: Path):
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        # Handles both lowercase (MFCD) and title case (Regular/VCD) keys
        return {
            'Accuracy': float(data.get('accuracy', data.get('Accuracy', 0.0))),
            'Precision': float(data.get('precision', data.get('Precision', 0.0))),
            'Recall': float(data.get('recall', data.get('Recall', 0.0))),
            'F1': float(data.get('f1_score', data.get('f1', data.get('F1', 0.0))))
        }
    except Exception:
        return None

def show_comparison_table(output_dir='/content/VCD_project/output'):
    out_dir = Path(output_dir)
    rows = []
    
    # Corrected config to match your exact file list
    configs = [
        ('random', 'Regular', out_dir / 'metrics_random_regular.json'),
        ('random', 'VCD',     out_dir / 'metrics_random_vcd.json'),
        ('random', 'MFCD',    out_dir / 'metrics_mfcd_random.json'),
        ('popular', 'Regular', out_dir / 'metrics_popular_regular.json'),
        ('popular', 'VCD',     out_dir / 'metrics_popular_vcd.json'),
        ('popular', 'MFCD',    out_dir / 'metrics_mfcd_popular.json'),
    ]

    for split, method, path in configs:
        m = parse_metrics_json(path)
        if m:
            rows.append({'Split': split, 'Method': method, **m})

    if not rows:
        print(f"No files found in {out_dir}. Please check task outputs.")
        return

    df = pd.DataFrame(rows)
    
    # Calculate Gain vs Previous row (e.g., VCD vs Regular, MFCD vs VCD)
    df['Gain vs Prev (%)'] = 0.0
    for split in df['Split'].unique():
        idx = df[df['Split'] == split].index
        for i in range(1, len(idx)):
            prev_f1 = df.loc[idx[i-1], 'F1']
            curr_f1 = df.loc[idx[i], 'F1']
            if prev_f1 > 0:
                df.at[idx[i], 'Gain vs Prev (%)'] = ((curr_f1 - prev_f1) / prev_f1) * 100

    # Styling for notebook display
    def color_gain(val):
        color = 'green' if val > 0 else ('red' if val < 0 else 'gray')
        return f'color: {color}; font-weight: bold'

    styled_df = df.style.format({
        'Accuracy': '{:.4f}', 'Precision': '{:.4f}', 'Recall': '{:.4f}', 
        'F1': '{:.4f}', 'Gain vs Prev (%)': '{:+.2f}%'
    }).map(color_gain, subset=['Gain vs Prev (%)']).hide(axis='index')

    display(Markdown("### VQA Hallucination Mitigation: Regular vs VCD vs MFCD"))
    display(styled_df)
