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
        if m and m['Accuracy'] > 0: 
            rows.append({'Split': split, 'Method': method, **m})
        else:
            print(f"Skipping {method} ({split}): File missing or empty metrics.")

    if not rows:
        print(f"No valid metric data found in {out_dir}.")
        return

    df = pd.DataFrame(rows)
    
    # Logic to calculate percentage improvement vs Regular
    df['Gain vs Regular (%)'] = 0.0
    for split in df['Split'].unique():
        base_rows = df[(df['Split'] == split) & (df['Method'] == 'Regular')]
        if base_rows.empty:
            continue
        base_f1 = base_rows.iloc[0]['F1']
        if base_f1 <= 0:
            continue
        for i in df[df['Split'] == split].index:
            curr_f1 = df.loc[i, 'F1']
            df.at[i, 'Gain vs Regular (%)'] = ((curr_f1 - base_f1) / base_f1) * 100
        # Use "-" for the Regular baseline row
        reg_idx = df[(df['Split'] == split) & (df['Method'] == 'Regular')].index
        if len(reg_idx) > 0:
            df.at[reg_idx[0], 'Gain vs Regular (%)'] = '-'
    # Make column object to allow mixed types (string for baseline)
    df['Gain vs Regular (%)'] = df['Gain vs Regular (%)'].astype(object)

    styled_df = df.style.format({
        'Accuracy': '{:.4f}', 'Precision': '{:.4f}', 'Recall': '{:.4f}', 
        'F1': '{:.4f}',
        'Gain vs Regular (%)': lambda v: v if isinstance(v, str) else f'{v:+.2f}%'
    }).map(lambda v: f'color: {"green" if v > 0 else "red"}; font-weight: bold',
           subset=['Gain vs Regular (%)']).hide(axis='index')

    display(Markdown("### VQA Mitigation Comparison: Regular vs VCD vs MFCD"))
    display(styled_df)
