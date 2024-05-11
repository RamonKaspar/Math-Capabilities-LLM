from .TechniqueInterface import TechniqueInterface

from .util import extract_number
from .util import create_prompt_gpt35
from .shared_prompts import get_few_shot_examples, get_system_prompt

class RolePlaying(TechniqueInterface):
    
    def __init__(self, name: str, few_shot_prompting: bool, dataset: str, service: str, model: str, temperature: float, max_token: int):
        # assert few_shot_prompting == False, "RolePlaying technique only supports zero-shot prompting"
        super().__init__(name, few_shot_prompting, dataset, service, model, temperature, max_token)
    
    def query(self, question: str) -> tuple[float, str, int, int]:
        # We can't reuse the get_llm_response method, therefore we need to reimplement it here
        role_setting = "From now on, you are an excellent math teacher and always teach your students math problems correctly. And I am one of your students."
        reply = "That's great to hear! As your math teacher, I'll do my best to explain mathematical concepts correctly so that you can understand them easily. Feel free to ask any math problems or questions you have, and I'll be glad to assist you. Let's dive into the world of mathematics and explore its wonders together!"
        if not self.few_shot_prompting:
            conversation = [
                        {"role": "user", "content": role_setting},
                        {"role": "assistant", "content": reply}, 
                        {"role": "user", "content": question}]
            response, prompt_tokens, completion_tokens = self.client.make_request(conversation)
        else:
            # Add role
            conversation = [
                        {"role": "user", "content": role_setting},
                        {"role": "assistant", "content": reply}, ]
            # Add few-shots
            conversation.append({"role": "user", "content": self.get_chat_introduction() + "\n" + "Here is one example how to do it:" + "\n\n" + self.get_question_prelude() + get_few_shot_examples(self.dataset)[0] + "\n\n" + self.get_few_shot_solutions()[0] + "\n\n" + "Now it's your turn."})
            # Add remaining four examples and answers to the conversation
            for i in range(1, len(get_few_shot_examples(self.dataset))):        
                conversation.append({"role": "user", "content": self.get_question_prelude() + " " + get_few_shot_examples(self.dataset)[i]})
                conversation.append({"role": "assistant", "content": self.get_few_shot_solutions()[i]})
            conversation.append({"role": "user", "content": self.get_question_prelude() + question})
            response, prompt_tokens, completion_tokens = self.client.make_request(conversation)
        # Extract the answer from the response
        answer = extract_number(response)  
        return answer, response, prompt_tokens, completion_tokens

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

FEW_SHOT_SOLUTIONS_GEOMETRY = [
    "3.0",
    "24.0",
    "351.85837720205683",
    "150.0",
    "14.137166941154069"
]