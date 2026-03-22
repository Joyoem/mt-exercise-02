import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def save_table_as_png(df, filename, title):
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.6 + 1))
    ax.axis('off')
    ax.set_title(title, fontsize=14, weight='bold', pad=10)
    
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#444444')
        elif row % 2 == 0:
            cell.set_facecolor('#f2f2f2')

    plt.savefig(filename, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Successfully saved table: {filename}")

def run_analysis():
    labels = ['0', '0.2', '0.4', '0.6', '0.8']
    test_ppl_values = [49.64, 50.45, 55.66, 68.68, 112.93]
    test_ppl_dict = dict(zip(labels, test_ppl_values))
    
    csv_files = {l: f"results_drop_{l}.csv" for l in labels}
    all_dfs = {}
    for l, f in csv_files.items():
        if os.path.exists(f):
            all_dfs[l] = pd.read_csv(f)
        else:
            print(f"Warning: {f} not found!")

    if not all_dfs:
        return

    # line table
    for metric in ['train_ppl', 'valid_ppl']:
        plt.figure(figsize=(10, 6))
        for l in labels:
            if l in all_dfs:
                df = all_dfs[l]
                plt.plot(df['epoch'], df[metric], marker='o', label=f'Dropout {l}')
        
        plt.xlabel('Epoch')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} Trend')
        
        if metric == 'valid_ppl':
            plt.ylim(40, 150)
        else:
            plt.autoscale(enable=True, axis='y')
            
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(f'plot_{metric}.png')
        plt.close()
        print(f"Successfully saved plot: plot_{metric}.png")

    num_epochs = len(next(iter(all_dfs.values())))
    epoch_list = [f"Epoch {i+1}" for i in range(num_epochs)]

    for metric in ['train_ppl', 'valid_ppl']:
        table_data = pd.DataFrame(index=epoch_list)
        for l in labels:
            if l in all_dfs:
                table_data[f"Drop {l}"] = all_dfs[l][metric].round(2).values
        
        table_df = table_data.reset_index().rename(columns={'index': 'Epoch'})
        save_table_as_png(table_df, f'table_{metric}.png', f"{metric.replace('_',' ').title()} Table")

    # Test PPL table
    test_df = pd.DataFrame([test_ppl_values], columns=[f"Drop {l}" for l in labels])
    test_df.insert(0, "Metric", ["Test PPL"])
    save_table_as_png(test_df, 'table_test_ppl.png', "Final Test Perplexity")

if __name__ == "__main__":
    run_analysis()