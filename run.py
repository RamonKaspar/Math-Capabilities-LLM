import pandas as pd
import time
from time import sleep
from tqdm import tqdm

from techniques.Baseline import Baseline
from techniques.PaL import PaL
from techniques.CoT import CoT
from techniques.RolePlaying import RolePlaying
from techniques.DeclarativeSymPy import DeclarativeSymPy
from techniques.ModelSelection import ModelSelection

def technique_factory(technique_name, few_shot_prompting, dataset, service, model, temperature, max_token):
    """ Factory function to create an instance of a technique based on the input parameters.
    NOTE: If you add a new technique, you should add a new condition to this function.
    """
    if technique_name == "Baseline":
        return Baseline(name=technique_name, few_shot_prompting=few_shot_prompting, dataset=dataset, service=service, model=model, temperature=temperature, max_token=max_token)
    elif technique_name == "PaL":
        return PaL(name=technique_name, few_shot_prompting=few_shot_prompting, dataset=dataset, service=service, model=model, temperature=temperature, max_token=max_token)
    elif technique_name == "CoT":
        return CoT(name=technique_name, few_shot_prompting=few_shot_prompting, dataset=dataset, service=service, model=model, temperature=temperature, max_token=max_token)
    elif technique_name == "RolePlaying":
        # if few_shot_prompting:
        #     raise ValueError("RolePlaying technique does not support few-shot prompting.")
        return RolePlaying(name=technique_name, few_shot_prompting=few_shot_prompting, dataset=dataset, service=service, model=model, temperature=temperature, max_token=max_token)
    elif technique_name == "DeclarativeSymPy":
        return DeclarativeSymPy(name=technique_name, few_shot_prompting=few_shot_prompting, dataset=dataset, service=service, model=model, temperature=temperature, max_token=max_token)
    elif technique_name == "ModelSelection":
        return ModelSelection(name=technique_name, few_shot_prompting=few_shot_prompting, dataset=dataset, service=service, model=model, temperature=temperature, max_token=max_token)
    else:
        raise ValueError("Unsupported technique name")


def run_evaluation(technique_name, few_shot_prompting, dataset, service, model, temperature, max_token):
    """Executes the evaluation of a specified technique on a given dataset and saves the results as a CSV file."""
    assert(dataset in ["arithmetic_100", "wordProblems_100", "geometry_100", "arithmetic_1000", "wordProblems_1000", "geometry_1000"])
    technique = technique_factory(technique_name, few_shot_prompting, dataset.split('_')[0], service, model, temperature, max_token)
    
    dataset_df = pd.read_csv(f"datasets/{dataset}.csv")
    results = []
    for index, sample in tqdm(dataset_df.iterrows(), total=dataset_df.shape[0]):
        question = str(sample['question'])
        try:
            correct_answer = float(sample['answer'])
        except Exception as e:
            print(f"Error parsing the answer to float for question '{question}': {e}")
            continue
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
        
        # Sleep for 2 second to avoid rate limits
        sleep(2)
        
    results_df = pd.DataFrame(results)
    few_shot_or_zero_shot = "Few-shot" if few_shot_prompting else "Zero-shot"
    results_df.to_csv(f"evaluation/data/{technique_name}_{few_shot_or_zero_shot}_{dataset}_{service}_{model}.csv", index=False)
    print(f"Results saved to evaluation/data/{technique_name}_{few_shot_or_zero_shot}_{dataset}_{service}_{model}.csv")


if __name__ == "__main__":
    # Fix the testing parameters
    SERVICE = "azure"
    MODEL = "gpt-35-turbo"
    TEMPERATURE = 0
    MAX_TOKEN = 400
    
    for technique_name in ["Baseline", "PaL", "CoT", "RolePlaying", "DeclarativeSymPy", "ModelSelection"]:
        for dataset in ["arithmetic_100", "wordProblems_100", "geometry_100"]:
            for few_shot_prompting in [True, False]:
                print(f"Running the evaluation for the {technique_name} {'Few-shot' if few_shot_prompting else 'Zero-shot'} technique on the {dataset} dataset using the {MODEL} model.")
                run_evaluation(technique_name, few_shot_prompting, dataset, SERVICE, MODEL, TEMPERATURE, MAX_TOKEN)
                print("Finished.")