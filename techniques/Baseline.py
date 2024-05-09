from .TechniqueInterface import TechniqueInterface

from .util import extract_number

class Baseline(TechniqueInterface):
    
    def __init__(self, name: str, few_shot_prompting: bool, dataset: str, service: str, model: str, temperature: float, max_token: int):
        super().__init__(name, few_shot_prompting, dataset, service, model, temperature, max_token)
    
    def query(self, question: str) -> tuple[float, str, int, int]:
        response, prompt_tokens, completion_tokens = self.get_llm_response(question)
        # Extract the answer from the response
        answer = extract_number(response)   
        return answer, None, prompt_tokens, completion_tokens

    def get_chat_introduction(self) -> str:
        return "Just return the answer to the problem."
    
    def get_question_prelude(self) -> str:
        return "Question: "
    
    def get_few_shot_solutions(self) -> list[str]:
        if self.dataset == 'arithmetic':
            return FEW_SHOT_SOLUTIONS_ARITHMETIC
        elif self.dataset == 'wordProblems':
            return FEW_SHOT_SOLUTIONS_WORDPROBLEMS
        elif self.dataset == 'geometry':
            return FEW_SHOT_SOLUTIONS_GEOMETRY
        else:
            raise ValueError("Unsupported dataset type")
    

# ======== PROMPTS ================

FEW_SHOT_SOLUTIONS_ARITHMETIC = [
    "36",
    "12",
    "4",
    "4",
    "9"
]

FEW_SHOT_SOLUTIONS_WORDPROBLEMS = [
    "8",
    "33",
    "29",
    "9",
    "8"
]

FEW_SHOT_SOLUTIONS_GEOMETRY = []