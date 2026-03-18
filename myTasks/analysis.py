import json
import pandas as pd
from pathlib import Path
from IPython.display import display, Markdown

def parse_metrics_json(path: Path):
    """Fuzzy parser to catch metrics across different LLava/VCD/MFCD formats."""
    if not path.exists():
        return None
    try:
        raw_data = json.loads(path.read_text())
        
        # Some scripts wrap metrics in a 'metrics' or 'results' key
        data = raw_data.get('metrics', raw_data.get('results', raw_data))
        
        # Standardize keys (looking for 'accuracy', 'Accuracy', etc.)
        def get_val(keys):
            for k in keys:
                if k in data: return float(data[k])
            return 0.0

        return {
            'Accuracy':  get_val(['accuracy', 'Accuracy', 'acc']),
            'Precision': get_val(['precision', 'Precision', 'prec']),
            'Recall':    get_val(['recall', 'Recall']),
            'F1':        get_val(['f1_score', 'f1', 'F1', 'f1-score'])
        }
    except Exception as e:
        print(f"Error parsing {path.name}: {e}")
        return None

def show_comparison_table(output_dir='/content/VCD_project/output'):
    out_dir = Path(output_dir)
    rows = []
    
    # Matching your exact file list provided earlier
    configs = [
        ('random',  'Regular', out_dir / 'metrics_random_regular.json'),
        ('random',  'VCD',     out_dir / 'metrics_random_vcd.json'),
        ('random',  'MFCD',    out_dir / 'metrics_mfcd_random.json'),
        ('popular', 'Regular', out_dir / 'metrics_popular_regular.json'),
        ('popular', 'VCD',     out_dir / 'metrics_popular_vcd.json'),
        ('popular', 'MFCD',    out_dir / 'metrics_mfcd_popular.json'),
    ]

    for split, method, path in configs:
        m = parse_metrics_json(path)
        if m and m['Accuracy'] > 0: # Ensure we didn't just get 0s
            rows.append({'Split': split, 'Method': method, **m})
        else:
            print(f"Skipping {method} ({split}): File missing or empty metrics.")

    if not rows:
        print(f"No valid metric data found in {out_dir}.")
        return

    df = pd.DataFrame(rows)
    
    # Logic to calculate percentage improvement
    df['Gain vs Prev (%)'] = 0.0
    for split in df['Split'].unique():
        idx = df[df['Split'] == split].index
        for i in range(1, len(idx)):
            prev_f1, curr_f1 = df.loc[idx[i-1], 'F1'], df.loc[idx[i], 'F1']
            if prev_f1 > 0:
                df.at[idx[i], 'Gain vs Prev (%)'] = ((curr_f1 - prev_f1) / prev_f1) * 100

    styled_df = df.style.format({
        'Accuracy': '{:.4f}', 'Precision': '{:.4f}', 'Recall': '{:.4f}', 
        'F1': '{:.4f}', 'Gain vs Prev (%)': '{:+.2f}%'
    }).map(lambda v: f'color: {"green" if v > 0 else "red"}; font-weight: bold', 
           subset=['Gain vs Prev (%)']).hide(axis='index')

    display(Markdown("### VQA Mitigation Comparison: Regular vs VCD vs MFCD"))
    display(styled_df)
