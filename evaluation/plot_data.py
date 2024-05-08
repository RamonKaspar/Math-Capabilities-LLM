import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_data(technique_name, dataset, service, model):
    return pd.read_csv(f"data/{technique_name}_{dataset}_{service}_{model}.csv")

def is_correct(row) -> bool:
    """Returns True if the answer is correct, False otherwise."""
    try:
        correct = abs(float(row['correct_answer']) - float(row['answer'])) < 1e-5
    except ValueError:
        correct = False # Cannot be parsed to float
    return correct

def classify_CA_IC_AB(row) -> str:
    """Classifies the answer as Correct Answer (CA), Incorrect Answer (IC), or Abstention (AB)."""
    try:
        if row['answer'] is None:
            return "AB"
        elif abs(float(row['correct_answer']) - float(row['answer'])) < 1e-5:
            return "CA"
        else:
            return "IC"
    except ValueError:
        return "IC"
    

def calculate_accuracy(df):
    
    df['correct'] = df.apply(is_correct, axis=1)
    return df['correct'].mean()

if __name__ == "__main__":
    METHODS = ["Baseline", "PaL"]
    
    # Calculate accuracy
    for method in METHODS:
        df = read_data(method, "arithmetic_100", "azure", "gpt-35-turbo")
        accuracy = calculate_accuracy(df)
        print(f"Accuracy for {method}: {accuracy}")