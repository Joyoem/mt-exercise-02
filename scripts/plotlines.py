import pandas as pd
import matplotlib.pyplot as plt
import glob

def create_plots():
    # Read all log files
    files = glob.glob('results/logs/dropout_*.log')
    dfs = []
    
    for f in files:
        df = pd.read_csv(f, sep='\t')
        dfs.append(df)
    
    full_df = pd.concat(dfs)
    
    # Create tables
    for ppl_type in ['Train_PPL', 'Valid_PPL', 'Test_PPL']:
        table = full_df.pivot(index='Epoch', columns='Dropout', values=ppl_type)
        print(f"\n{ppl_type} Table:")
        print(table.to_markdown())
        table.to_csv(f'results/tables/{ppl_type.lower()}_table.csv')
    
    # Create line plots
    plt.figure(figsize=(12,6))
    for dropout, group in full_df.groupby('Dropout'):
        plt.plot(group['Epoch'], group['Valid_PPL'], label=f'Dropout={dropout}')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Validation Perplexity by Dropout Rate')
    plt.legend()
    plt.savefig('results/plots/valid_ppl.png')
    plt.close()
