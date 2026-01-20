import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuración
EXPERIMENTS_DIR = "experiments"
RESULTS_FILE = "logs/train/results.csv"
OUTPUT_DIR = "analysis_results"

# Métricas a analizar
METRICS = [
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss"
]


def find_experiments(root_dir):
    """Finds all experiment folders containing results.csv"""
    experiments = []
    for d in os.listdir(root_dir):
        exp_path = os.path.join(root_dir, d)
        if os.path.isdir(exp_path):
            results_path = os.path.join(exp_path, RESULTS_FILE)
            if os.path.isfile(results_path):
                experiments.append({
                    'name': d,
                    'path': exp_path,
                    'results_csv': results_path
                })
            else:
                print(f"[WARN] {d} does not contain {RESULTS_FILE}")
    return experiments


def load_experiment_data(experiments):
    """Loads data from all experiments"""
    data = {}
    for exp in experiments:
        try:
            df = pd.read_csv(exp['results_csv'])
            df['experiment'] = exp['name']
            data[exp['name']] = df
            print(f"✓ Loaded: {exp['name']} ({len(df)} epochs)")
        except Exception as e:
            print(f"✗ Error in {exp['name']}: {e}")
    return data


def create_metric_plots(data, output_dir):
    """Creates evolution plots for each metric per experiment"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    for metric in METRICS:
        plt.figure()
        
        for exp_name, df in data.items():
            if metric in df.columns:
                plt.plot(df['epoch'], df[metric], label=exp_name, marker='o', 
                        markersize=3, linewidth=2, alpha=0.7)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.title(f'Evolution of {metric} per Experiment', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        safe_name = metric.replace('/', '_').replace('(', '').replace(')', '')
        plt.savefig(os.path.join(output_dir, f'{safe_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plot saved: {safe_name}.png")


def create_comparison_table(data):
    """Creates comparison table with final and best metric values"""
    comparison = []
    
    for exp_name, df in data.items():
        row = {'Experiment': exp_name}
        
        # Metrics from last epoch
        last_epoch = df.iloc[-1]
        row['Epochs'] = len(df)
        
        # Main metrics
        for metric in METRICS:
            if metric in df.columns:
                # Best value of the metric
                if 'loss' in metric.lower():
                    best_val = df[metric].min()
                else:
                    best_val = df[metric].max()
                
                # Final value
                final_val = last_epoch[metric]
                
                row[f'{metric}_best'] = best_val
                row[f'{metric}_final'] = final_val
        
        comparison.append(row)
    
    return pd.DataFrame(comparison)


def create_summary_table(data):
    """Creates summary table with the most important metrics"""
    summary = []
    
    for exp_name, df in data.items():
        row = {
            'Experiment': exp_name,
            'Epochs': len(df),
        }
        
        # Key metrics
        key_metrics = {
            'mAP50-95 (best)': ('metrics/mAP50-95(B)', 'max'),
            'mAP50-95 (final)': ('metrics/mAP50-95(B)', 'last'),
            'mAP50 (best)': ('metrics/mAP50(B)', 'max'),
            'Precision (best)': ('metrics/precision(B)', 'max'),
            'Recall (best)': ('metrics/recall(B)', 'max'),
            'Val Loss (best)': ('val/box_loss', 'min'),
        }
        
        for display_name, (metric, agg) in key_metrics.items():
            if metric in df.columns:
                if agg == 'max':
                    row[display_name] = df[metric].max()
                elif agg == 'min':
                    row[display_name] = df[metric].min()
                elif agg == 'last':
                    row[display_name] = df[metric].iloc[-1]
        
        summary.append(row)
    
    summary_df = pd.DataFrame(summary)
    
    # Sort by mAP50-95 (best)
    if 'mAP50-95 (best)' in summary_df.columns:
        summary_df = summary_df.sort_values('mAP50-95 (best)', ascending=False)
    
    return summary_df


def create_heatmap(data, output_dir):
    """Creates comparative heatmap of final metrics"""
    # Prepare data for heatmap
    heatmap_data = []
    
    for exp_name, df in data.items():
        row = {'Experiment': exp_name}
        last_epoch = df.iloc[-1]
        
        for metric in ['metrics/mAP50-95(B)', 'metrics/mAP50(B)', 
                      'metrics/precision(B)', 'metrics/recall(B)']:
            if metric in df.columns:
                row[metric.split('/')[-1]] = last_epoch[metric]
        
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data).set_index('Experiment')
    
    # Create heatmap
    plt.figure(figsize=(10, len(heatmap_df) * 0.5 + 2))
    sns.heatmap(heatmap_df, annot=True, fmt='.4f', cmap='RdYlGn', 
                cbar_kws={'label': 'Value'}, linewidths=0.5)
    plt.title('Comparison of Final Metrics per Experiment', 
             fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Experiment', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_final_metrics.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Heatmap saved: heatmap_final_metrics.png")


def main():
    print("="*60)
    print("YOLO EXPERIMENTS ANALYSIS")
    print("="*60)
    
    # Find experiments
    print(f"\n1. Searching for experiments in '{EXPERIMENTS_DIR}'...")
    experiments = find_experiments(EXPERIMENTS_DIR)
    
    if not experiments:
        print("✗ No valid experiments found.")
        return
    
    print(f"✓ Found {len(experiments)} experiments")
    
    # Load data
    print("\n2. Loading experiment data...")
    data = load_experiment_data(experiments)
    
    if not data:
        print("✗ Could not load data.")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate plots
    print(f"\n3. Generating plots in '{OUTPUT_DIR}'...")
    create_metric_plots(data, OUTPUT_DIR)
    
    # Generate heatmap
    print("\n4. Generating comparative heatmap...")
    create_heatmap(data, OUTPUT_DIR)
    
    # Create summary table
    print("\n5. Generating summary table...")
    summary_df = create_summary_table(data)
    
    # Save summary table
    summary_path = os.path.join(OUTPUT_DIR, 'experiment_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Summary table saved: {summary_path}")
    
    # Display summary table
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY TABLE")
    print("="*60)
    print(summary_df.to_string(index=False))
    
    # Create complete comparison table
    print("\n6. Generating complete comparison table...")
    comparison_df = create_comparison_table(data)
    comparison_path = os.path.join(OUTPUT_DIR, 'complete_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"✓ Comparison table saved: {comparison_path}")
    
    # Display best experiment
    print("\n" + "="*60)
    print("BEST EXPERIMENT")
    print("="*60)
    best_exp = summary_df.iloc[0]
    print(f"Experiment: {best_exp['Experiment']}")
    if 'mAP50-95 (best)' in best_exp:
        print(f"mAP50-95: {best_exp['mAP50-95 (best)']:.4f}")
    print("\n✓ Analysis completed successfully!")


if __name__ == "__main__":
    main()