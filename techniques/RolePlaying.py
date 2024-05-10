from .TechniqueInterface import TechniqueInterface

from .util import extract_number

class RolePlaying(TechniqueInterface):
    
    def __init__(self, name: str, few_shot_prompting: bool, dataset: str, service: str, model: str, temperature: float, max_token: int):
        assert few_shot_prompting == False, "RolePlaying technique only supports zero-shot prompting"
        super().__init__(name, few_shot_prompting, dataset, service, model, temperature, max_token)
    
    def query(self, question: str) -> tuple[float, str, int, int]:
        # We can't reuse the get_llm_response method, therefore we need to reimplement it here
        role_setting = "From now on, you are an excellent math teacher and always teach your students math problems correctly. And I am one of your students."
        reply = "That's great to hear! As your math teacher, I'll do my best to explain mathematical concepts correctly so that you can understand them easily. Feel free to ask any math problems or questions you have, and I'll be glad to assist you. Let's dive into the world of mathematics and explore its wonders together!"
        conversation = [
                    {"role": "user", "content": role_setting},
                    {"role": "assistant", "content": reply}, 
                    {"role": "user", "content": question}]
        response, prompt_tokens, completion_tokens = self.client.make_request(conversation)
        # Extract the answer from the response
        answer = extract_number(response)  
        return answer, response, prompt_tokens, completion_tokens

    def get_chat_introduction(self) -> str:
        # NOT NEEDED
        return None
    
    def get_question_prelude(self) -> str:
        # NOT NEEDED
        return None
    
    def get_few_shot_solutions(self) -> list[str]:
        # NOT NEEDED
        return None
    