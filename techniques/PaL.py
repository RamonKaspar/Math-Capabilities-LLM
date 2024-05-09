from .TechniqueInterface import TechniqueInterface

from .util import execute_solution_function

class PaL(TechniqueInterface):
    
    def __init__(self, name: str, few_shot_prompting: bool, dataset: str, service: str, model: str, temperature: float, max_token: int):
        super().__init__(name, few_shot_prompting, dataset, service, model, temperature, max_token)
    
    def query(self, question: str) -> tuple[float, str, int, int]:
        pal_code, prompt_tokens, completion_tokens = self.get_llm_response(question)
        # Execute the PaL Code
        answer = execute_solution_function(pal_code)
        return answer, pal_code, prompt_tokens, completion_tokens

    def get_chat_introduction(self) -> str:
        return "Let's use python to solve math problems. You need to write python code to answer these math questions."
    
    def get_question_prelude(self) -> str:
        return "Answer the following question in Python: "
    
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
'''
def solution():
    """Find the next term in the sequence 1, 4, 9, 16, 25."""
    sequence = [1, 4, 9, 16, 25]
    next_term = (len(sequence) + 1) ** 2
    return next_term
''',
'''
def solution():
    """Calculate 3 multiplied by 4."""
    result = 3 * 4
    return result

''',
'''
def solution():
    """Calculate the number of hours between 3 PM and 7 PM."""
    start_time = 15  # 3 PM in 24-hour format
    end_time = 19    # 7 PM in 24-hour format
    hours_between = end_time - start_time
    return hours_between
''',
'''
def solution():
    """Calculate the sum of 2 and 2."""
    result = 2 + 2
    return result
''',
'''
def solution():
    """Find the tens digit of the number 839291."""
    number = 839291
    tens_digit = (number // 10) % 10
    return tens_digit
'''
]

# These prompts where taken from: https://github.com/reasoning-machines/pal/blob/main/pal/prompt/math_prompts.py
FEW_SHOT_SOLUTIONS_WORDPROBLEMS = [
'''
def solution():
    """Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"""
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    money_spent = bagels * bagel_cost
    money_left = money_initial - money_spent
    result = money_left
    return result
''',
'''
def solution():
    """Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?"""
    golf_balls_initial = 58
    golf_balls_lost_tuesday = 23
    golf_balls_lost_wednesday = 2
    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
    result = golf_balls_left
    return result
''',
'''
def solution():
    """There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?"""
    computers_initial = 9
    computers_per_day = 5
    num_days = 4  # 4 days between monday and thursday
    computers_added = computers_per_day * num_days
    computers_total = computers_initial + computers_added
    result = computers_total
    return result
''',
'''
def solution():
    """Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?"""
    toys_initial = 5
    mom_toys = 2
    dad_toys = 2
    total_received = mom_toys + dad_toys
    total_toys = toys_initial + total_received
    result = total_toys
    return result
''',
'''
def solution():
    """Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?"""
    jason_lollipops_initial = 20
    jason_lollipops_after = 12
    denny_lollipops = jason_lollipops_initial - jason_lollipops_after
    result = denny_lollipops
    return result
'''
]

FEW_SHOT_SOLUTIONS_GEOMETRY = []