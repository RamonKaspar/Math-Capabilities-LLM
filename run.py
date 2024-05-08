import pandas as pd
import time
from tqdm import tqdm

from techniques.Baseline import Baseline
from techniques.PaL import PaL

def technique_factory(technique_name, dataset, service, model, temperature, max_token):
    """ Factory function to create an instance of a technique based on the input parameters.
    NOTE: If you add a new technique, you should add a new condition to this function.
    """
    if technique_name == "Baseline":
        return Baseline(name=technique_name, dataset=dataset, service=service, model=model, temperature=temperature, max_token=max_token)
    elif technique_name == "PaL":
        return PaL(name=technique_name, dataset=dataset, service=service, model=model, temperature=temperature, max_token=max_token)
    else:
        raise ValueError("Unsupported technique name")


def run_evaluation(technique_name, dataset, service, model, temperature, max_token):
    """Executes the evaluation of a specified technique on a given dataset and saves the results as a CSV file."""
    assert(dataset in ["arithmetic_100", "wordProblems_100", "geometry_100", "arithmetic_1000", "wordProblems_1000", "geometry_1000"])
    technique = technique_factory(technique_name, dataset.split('_')[0], service, model, temperature, max_token)
    
    dataset_df = pd.read_csv(f"datasets/{dataset}.csv")
    results = []
    for index, sample in tqdm(dataset_df.iterrows(), total=dataset_df.shape[0]):
        question = str(sample['question'])
        correct_answer = float(sample['answer']) 
        category = sample.get('category', 'N/A')  
        subcategory = sample.get('subcategory', 'N/A') 
        try: 
            start_time = time.time()
            response = technique.query_with_detailed_response(question)
            latency = time.time() - start_time
        except Exception as e:
            print(f"An error occurred while processing the question: {question}")
            print(f"Error: {e}")
            continue
        response['correct_answer'] = correct_answer  
        response['category'] = category 
        response['subcategory'] = subcategory  
        response['latency_in_seconds'] = latency
        results.append(response)
        break
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"evaluation/data/{technique_name}_{dataset}_{service}_{model}.csv", index=False)
    print(f"Results saved to evaluation/data/{technique_name}_{dataset}_{service}_{model}.csv.csv")
    
if __name__ == "__main__":
    run_evaluation("PaL", "arithmetic_100", "azure", "gpt-35-turbo", 0.1, 200)
    




