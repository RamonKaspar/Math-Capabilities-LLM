import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import os


def read_data(technique_name, few_shot_or_zero_shot, dataset, service, model):
    assert(dataset in ["arithmetic_100", "wordProblems_100", "geometry_100", "arithmetic_1000", "wordProblems_1000", "geometry_1000"])
    assert(few_shot_or_zero_shot in ["Few-shot", "Zero-shot"])
    return pd.read_csv(f"data/{technique_name}_{few_shot_or_zero_shot}_{dataset}_{service}_{model}.csv")

def classify(row) -> str:
    """Classifies the answer as Correct Answer, Incorrect Answer, or Error."""
    try:
        if row['answer'] is None or pd.isnull(row['answer']):
            return "Error"
        elif abs(float(row['correct_answer']) - float(row['answer'])) <= 1e-4:
            return "Correct"
        else:
            return "Incorrect"
    except ValueError:
        return "Incorrect"
    
def calculate_accuracy(df):
    """Calculates the accuracy of a given dataframe."""
    df['classification'] = df.apply(classify, axis=1)
    accuracy = (df['classification'] == 'Correct').mean()
    return accuracy

def print_questions_with_incorrect_answers(df):
    incorrect_answers = df[df['classification'] == 'Incorrect']
    for index, row in incorrect_answers.iterrows():
        print(f"Question: {row['question']}")
        print(f"Correct answer: {row['correct_answer']}")
        print(f"Model answer: {row['answer']}")
        print()
        
def print_questions_with_no_answers(df):
    no_answers = df[df['classification'] == 'Error']
    for index, row in no_answers.iterrows():
        print(f"Question: {row['question']}")
        print(f"Correct answer: {row['correct_answer']}")
        print()
        
def add_classification(df):
    """Adds a classification column to the dataframe."""
    df['classification'] = df.apply(classify, axis=1)
    return df

def overview(methods, dataset, service, model):
    """Prints an overview of the results for each method."""
    for method in methods:
        df = pd.read_csv(f"data/{method}_Zero-shot_{dataset}_{service}_{model}.csv")
        accuracy = calculate_accuracy(df)
        print(f"Accuracy for {method}: {accuracy:.2f}")
        print(f"Number of questions: {len(df)}")
        print(f"Number of correct answers: {len(df[df['classification'] == 'Correct'])}")
        print(f"Number of incorrect answers: {len(df[df['classification'] == 'Incorrect'])}")
        print(f"Number of abstentions: {len(df[df['classification'] == 'Error'])}")
        print()
        print("Questions with incorrect answers:")
        print_questions_with_incorrect_answers(df)
        print("Questions with no answers:")
        print_questions_with_no_answers(df)
        print("------------------------------------------")
        print("\n\n")

def plot_results_grid(dataset, ax=None):
    """Plots a grid of the results for each method (i.e. classification for each answer)."""
    n = 100
    classifications_per_method = []
    methods = []
    # Read and classify data for each method
    for file in os.listdir("data"):
        if file.endswith(".csv") and dataset in file:
            df = pd.read_csv(os.path.join("data", file))
            df = add_classification(df)
            classifications_per_method.append(df['classification'])
            method_name = file.split("_")[0]
            method_few_shot_or_zero_shot = file.split("_")[1]
            methods.append(f"{method_name}-{method_few_shot_or_zero_shot}")

    class_to_idx = {'Correct': 0, 'Incorrect': 1, 'Error': 2}
    color_map = np.array([
        sns.color_palette("hls", 8)[2],  # Pastel green
        sns.color_palette("hls", 8)[0],  # Pastel red
        sns.color_palette("hls", 8)[1]  # Orange
    ])

    results_array_numeric = [np.array([class_to_idx[cls] for cls in method_cls]) for method_cls in classifications_per_method]
    results_array = np.column_stack(results_array_numeric)

    unique_classifications = np.unique(results_array)
    cmap = ListedColormap(color_map[unique_classifications])
    norm = BoundaryNorm(np.arange(len(unique_classifications) + 1), cmap.N)  # Set boundaries for color bins

    aspect_ratio = len(methods) / float(n) 

    if ax is None:
        plt.figure(figsize=(10 * aspect_ratio, 10))
    sns.heatmap(results_array, cmap=cmap, norm=norm, linewidths=0.5,
                     cbar_kws={"ticks": np.arange(len(unique_classifications)) + 0.5}, ax=ax)
    ax.figure.axes[-1].set_yticklabels(['Correct', 'Incorrect', 'Error'][i] for i in unique_classifications)  # Colorbar label setup

    # Plot formatting
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticks(np.arange(len(methods)) + 0.5)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_yticks([])  # Optionally hide y-ticks if they are irrelevant
    ax.set_title(f"Classification Grid for {dataset} on azure gpt-3.5-turbo")
    if ax is None: 
        plt.show()
    

def plot_accuracy(dataset, ax = None):
    """Plots an overview of the accracies for each method."""
    # Set the color palette from Seaborn
    sns.set_palette('deep')
    
    techniques, zero_shot, few_shot = [], [], []
    for file in os.listdir("data"):
        if file.endswith(".csv") and dataset in file:
            df = pd.read_csv(os.path.join("data", file))
            acc = calculate_accuracy(df)
            method_name = file.split("_")[0]
            if method_name not in techniques:
                techniques.append(method_name)
            method_few_shot_or_zero_shot = file.split("_")[1]
            if (method_few_shot_or_zero_shot == "Zero-shot"):
                zero_shot.append(round(acc*100, 1))
            elif (method_few_shot_or_zero_shot == "Few-shot"):
                few_shot.append(round(acc*100, 1))
    x = range(len(techniques))  # the label locations

    if ax is None:
        fig, ax = plt.subplots()
    rects1 = ax.bar(x, zero_shot, width=0.4, label='Zero-shot')
    rects2 = ax.bar([p + 0.4 for p in x], few_shot, width=0.4, label='Few-shot')

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f"Accuracies for {dataset}")
    ax.set_xticks([p + 0.2 for p in x])
    ax.set_xticklabels(techniques, rotation=45, ha="right")
    ax.legend()
    if ax is not None:
        ax.set_ylim(0, 100)  # Set y-axis limits from 0 to 100
        
    # Function to add labels on the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    if ax is None:
        plt.show()
    

import tiktoken

def plot_answer_length(dataset, ax = None):
    """ 
    Devides the reasoning length (in tokens) into bins and plots the number of correct answers in each bin.
    """
    result = []
    enc = tiktoken.get_encoding("cl100k_base")  # Define encoding
    for file in os.listdir("data"):
        if file.endswith(".csv") and dataset in file:
            df = pd.read_csv(os.path.join("data", file))
            df = add_classification(df)
            for index, d in df.iterrows():
                reasoning_length = len(enc.encode(d['reasoning'])) if isinstance(d['reasoning'], str) else 0
                if reasoning_length != 0:
                    result.append({
                        "classification": d['classification'],
                        "reasoning_length": reasoning_length
                    })
    if result:  # Check if the result list is populated
        result_df = pd.DataFrame(result)
        bin_edges = np.arange(0, max(result_df['reasoning_length']) + 10, 10)  # Bins of size 10 tokens
        sns.histplot(data=result_df, x="reasoning_length", hue="classification", multiple="stack", bins=bin_edges, ax=ax)
        ax.set_title(f"Reasoning Length for {dataset}")
        if ax is not None:
            ax.set_ylim(0, 360)  # Set y-axis limits from 0 to 100
    else:
        ax.set_title(f"No data available for {dataset}")  # Indicate no data to plot
    if ax is  None:
        plt.show()
            

def calculate_money_used():
    """Calculates the overall money usage for the azure API."""
    # https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
    # Per 1000 tokens!
    INPUT_TOKENS_IN_CHF = 0.0028
    OUTPUT_TOKENS_IN_CHF = 0.0037
    total_money_used = 0
    for file in os.listdir("data"):
        if file.endswith(".csv") and 'azure' in file:
            df = pd.read_csv(os.path.join("data", file))
            input_tokens = df['prompt_tokens'].sum()
            output_tokens = df['completion_tokens'].sum()
            money_used = (input_tokens / 1000 * INPUT_TOKENS_IN_CHF) + (output_tokens / 1000 * OUTPUT_TOKENS_IN_CHF)
            total_money_used += money_used
    return total_money_used



if __name__ == "__main__":
    # Plot grid in one plot
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    plot_results_grid("arithmetic_100", axs[0])
    plot_results_grid("wordProblems_100", axs[1])
    plot_results_grid("geometry_100", axs[2])
    plt.tight_layout()
    plt.show()
    
    # Plot accuracies in one plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    plot_accuracy("arithmetic_100", axs[0])
    plot_accuracy("wordProblems_100", axs[1])
    plot_accuracy("geometry_100", axs[2])
    plt.tight_layout()
    plt.show()
    
    # Plot reasoning length classificiation
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    plot_answer_length("arithmetic_100", axs[0])
    plot_answer_length("wordProblems_100", axs[1])
    plot_answer_length("geometry_100", axs[2])
    plt.tight_layout()
    plt.show()