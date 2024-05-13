import os
import pandas as pd
import matplotlib.pyplot as plt

from plot_data import calculate_accuracy

def calculate_mean_tokens_usage(df):
    return (df['prompt_tokens'].mean() + df['completion_tokens'].mean())/2.0

def calculate_mean_latency(df):
    return df['latency_in_seconds'].mean()

def create_dataframe(dataset): 
    results = []   
    for file in os.listdir("data"):
        if file.endswith(".csv") and dataset in file:
            df = pd.read_csv(os.path.join("data", file))
            acc = calculate_accuracy(df)
            method_name = file.split("_")[0] + " " + file.split("_")[1]
            tokens = calculate_mean_tokens_usage(df)
            latency = calculate_mean_latency(df)
            results.append({
                'Technique': method_name,
                'Accuracy': acc,
                'Mean Token usage': tokens,
                'Mean Latency': latency
            })
    return pd.DataFrame(results)

def mean_of_dataframes_3(df1, df2, df3):
    numeric_columns = df1.select_dtypes(include=[float, int]).columns
    mean_df = df1[numeric_columns].add(df2[numeric_columns]).add(df3[numeric_columns]) / 3
    mean_df['Technique'] = df1['Technique']
    column_order = ['Technique'] + list(numeric_columns)
    mean_df = mean_df[column_order]
    return mean_df


def plot_scatter(df, dataset):
    # Set up the plotting environment
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # plt.set_title(f'Accuracy vs Mean Token Usage and Mean Latency on {dataset}')
    # Scatter plot for Mean Token Usage
    axes[0].scatter(df['Mean Token usage'], df['Accuracy'])
    axes[0].set_xlabel('Mean Token Usage')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy vs Mean Token Usage')
    for i, txt in enumerate(df['Technique']):
        axes[0].annotate(txt, (df['Mean Token usage'][i], df['Accuracy'][i]), textcoords="offset points", xytext=(0,10), ha='center')
    
    # Scatter plot for Mean Latency
    axes[1].scatter(df['Mean Latency'], df['Accuracy'])
    axes[1].set_xlabel('Mean Latency')
    axes[1].set_title('Accuracy vs Mean Latency')
    for i, txt in enumerate(df['Technique']):
        axes[1].annotate(txt, (df['Mean Latency'][i], df['Accuracy'][i]), textcoords="offset points", xytext=(0,10), ha='center')

    # Show plots
    plt.tight_layout()
    plt.show()
    
def plot_scatter(df):
    # Set up the plotting environment
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Define colors and markers
    technique_colors = {
        'Baseline': 'red', 'CoT': 'green', 'DeclarativeSymPy': 'blue', 
        'ModelSelection': 'purple', 'PaL': 'orange', 'RolePlaying': 'brown'
    }
    markers = {'Few-shot': 'o', 'Zero-shot': 's'}

    # Scatter plots
    for index, row in df.iterrows():
        tech_base = row['Technique'].split()[0]
        shot_type = row['Technique'].split()[1]
        color = technique_colors[tech_base]
        marker = markers[shot_type]
        axes[0].scatter(row['Mean Token usage'], row['Accuracy'], label=row['Technique'] if tech_base not in axes[0].get_legend_handles_labels()[1] else "", 
                        color=color, marker=marker, s=100, alpha=0.7)
        axes[1].scatter(row['Mean Latency'], row['Accuracy'], label=row['Technique'] if tech_base not in axes[1].get_legend_handles_labels()[1] else "", 
                        color=color, marker=marker, s=100, alpha=0.7)

    # Set labels, titles, and grid
    axes[0].set_xlabel('Mean Token Usage', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Accuracy vs Mean Token Usage', fontsize=14)
    axes[0].grid(True)

    axes[1].set_xlabel('Mean Latency', fontsize=12)
    axes[1].set_title('Accuracy vs Mean Latency', fontsize=14)
    axes[1].grid(True)

    # Handle legend (only draw it once in a suitable position)
    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc='upper center', ncol=len(labels)//2, title="Techniques")

    # Improve layout and show plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect to make space for the legend
    plt.show()
    

def calculate_score(df, weight_accuracy=0.7, weight_token=0.15, weight_latency=0.15):
    # Normalize the metrics
    # Accuracy is already between 0 and 1, no need to change
    normalized_token_usage = 1 - (df['Mean Token usage'] - df['Mean Token usage'].min()) / (df['Mean Token usage'].max() - df['Mean Token usage'].min())
    normalized_latency = 1 - (df['Mean Latency'] - df['Mean Latency'].min()) / (df['Mean Latency'].max() - df['Mean Latency'].min())
    # Calculate the score with given weights
    df['Score'] = (df['Accuracy'] * weight_accuracy +
                   normalized_token_usage * weight_token +
                   normalized_latency * weight_latency)
    return df.sort_values(by='Score', ascending=False)
            
if __name__ == '__main__':
    df_arith = create_dataframe('arithmetic_100')
    df_mwp = create_dataframe('wordProblems_100')
    df_geo = create_dataframe('geometry_100')
    
    df_total = mean_of_dataframes_3(df_arith, df_mwp, df_geo)
    
    # plot_scatter(df_total)  
    
    df = calculate_score(df_total) 
    print(df.to_markdown(index=False))
    # print(df) 
    
    
    
    