from .TechniqueInterface import TechniqueInterface

import random
import re

from .util import create_prompt_gpt35
from .PaL import PaL
from .CoT import CoT

class ModelSelection(TechniqueInterface):
    
    def __init__(self, name: str, few_shot_prompting: bool, dataset: str, service: str, model: str, temperature: float, max_token: int):
        super().__init__(name, few_shot_prompting, dataset, service, model, temperature, max_token)
    
    def query(self, question: str) -> tuple[float, str, int, int]:
        """ 
        1. Query CoT for solutions
        2. Query PAL for solutions
        3. Query model selection answers
        
        Note that we only query selection answers when CoT and PAL answers are different. Otherwise, we directly use CoT or PAL answers.
        """
        # Query CoT
        cot_technique = CoT(name="CoT", few_shot_prompting=self.few_shot_prompting, dataset=self.dataset, service=self.service, model=self.model, temperature=self.temperature, max_token=self.max_token)
        try:
            cot_response, cot_reasoning, cot_prompt_tokens, cot_completion_tokens = cot_technique.query(question)
        except:
            cot_response, cot_reasoning, cot_prompt_tokens, cot_completion_tokens = None, None, 0, 0
        # Query PaL
        pal_technique = PaL(name="PaL", few_shot_prompting=self.few_shot_prompting, dataset=self.dataset, service=self.service, model=self.model, temperature=self.temperature, max_token=self.max_token)
        try:
            pal_response, pal_reasoning, pal_prompt_tokens, pal_completion_tokens = pal_technique.query(question)
        except:
            pal_response, pal_reasoning, pal_prompt_tokens, pal_completion_tokens = None, None, 0, 0
        # Do selection
        if cot_response is not None and pal_response is not None:
            check = False
            try:    # Do a save check
                if abs(float(cot_response) - float(pal_response)) < 1e-3:   
                    check = True
            except:
                pass
            if check:
                # Answers are the same, We return CoT reasoning (PaL reasoning would be an option as well)
                return cot_response, cot_reasoning, cot_prompt_tokens+pal_prompt_tokens, cot_completion_tokens+pal_completion_tokens
            else:
                # We have different answers from CoT and PaL. We need to query selection.
                try: 
                    selection_response, selection_prompt_tokens, selection_completion_tokens = self.query_selection(question, cot_reasoning, pal_reasoning)
                    selection_choice = self.extract_choice(selection_response)
                    if selection_choice == '(A)':
                        return cot_response, cot_reasoning, cot_prompt_tokens+pal_prompt_tokens+selection_prompt_tokens, cot_completion_tokens+pal_completion_tokens+selection_completion_tokens
                    elif selection_choice == '(B)':
                        return pal_response, pal_reasoning, cot_prompt_tokens+pal_prompt_tokens+selection_prompt_tokens, cot_completion_tokens+pal_completion_tokens+selection_completion_tokens
                except Exception as e:
                    # On average, PaL is better than CoT
                    return pal_response, pal_reasoning, cot_prompt_tokens+pal_prompt_tokens, cot_completion_tokens+pal_completion_tokens
        elif cot_response is not None and pal_response is None:
            return cot_response, cot_reasoning, cot_prompt_tokens+pal_prompt_tokens, cot_completion_tokens+pal_completion_tokens
        elif cot_response is None and pal_response is not None:
            return pal_response, pal_reasoning, cot_prompt_tokens+pal_prompt_tokens, cot_completion_tokens+pal_completion_tokens
        else:
            return None, None, cot_prompt_tokens+pal_prompt_tokens, cot_completion_tokens+pal_completion_tokens
        
    def get_chat_introduction(self) -> str:
        return "There are two choices to the same math problem. One uses natural language to answer the question, while the other uses Python program to answer it. Either of them can correctly answer the math problem. You need to identify which choice can correctly answer the math problem."
    
    def get_question_prelude(self) -> str:
        return "Math problem: "
        
    def get_few_shot_solutions(self) -> list[str]:
        if self.dataset == 'arithmetic':
            return FEW_SHOT_SOLUTIONS_ARITHMETIC
        elif self.dataset == 'wordProblems':
            return FEW_SHOT_SOLUTIONS_WORDPROBLEMS
        elif self.dataset == 'geometry':
            return FEW_SHOT_SOLUTIONS_GEOMETRY
        else:
            raise ValueError("Unsupported dataset type")
        
    def get_few_shot_examples(self) -> list[str]:
        if self.dataset == 'arithmetic':
            return FEW_SHOT_EXAMPLES_ARITHMETIC
        elif self.dataset == 'wordProblems':
            return FEW_SHOT_EXAMPLES_WORDPROBLEMS
        elif self.dataset == 'geometry':
            return FEW_SHOT_EXAMPLES_GEOMETRY
        else:
            raise ValueError("Unsupported dataset type")
    
    # ======== SELECTION ================   
    def query_selection(self, question: str, cot_solution: str, pal_solution: str):
        """
        This function is used to query OpenAI for selection solutions.
        """
        question_extended = question + "\n\n" + "(A)" + "\n" + cot_solution + "\n\n" + "(B)" + "\n" + "\n\n" + pal_solution + "\n\n" + "Which of the above two choices can correctly answer the math problem?"
        selection_message = create_prompt_gpt35(
            few_shot_prompting=self.few_shot_prompting,
            system_prompt=SELECT_SYSTEM,
            introduction=self.get_chat_introduction(),
            question_prelude=self.get_question_prelude(),
            few_shot_examples=self.get_few_shot_examples(),
            few_shot_answers=self.get_few_shot_solutions(),
            question=question_extended
        )
        try:
            selection_solution, prompt_tokens, completion_tokens = self.client.make_request(messages=selection_message)
        except Exception as e:
            selection_solution = None, 0, 0
        return selection_solution, prompt_tokens, completion_tokens
    
    def extract_choice(self, selection: str):
        if selection.startswith('Both') or selection.startswith('Neither'):
            if random.random() < 0.5:
                choices_a_b = '(A)'
            else:
                choices_a_b = '(B)'
        else:
            try:
                choices = re.findall(r'(\(A\)|\(B\)) can correctly', selection)
                if len(choices) == 0:
                    choices = re.findall(
                        r'(\(A\)|\(B\)) is(?:\sthe)? correct', selection)
                choices_a_b = choices[0]
            except:
                if random.random() < 0.5:
                    choices_a_b = '(A)'
                else:
                    choices_a_b = '(B)'
        return choices_a_b
    

# ======== PROMPTS ================

SELECT_SYSTEM = '''You are a helpful assistant that can identify the correct answer to the math problem.'''

# Arithmetic
FEW_SHOT_EXAMPLES_ARITHMETIC = [
"""
What is the next term in 1, 4, 9, 16, 25?

(A)
Answer:
These numbers are squares of consecutive integers.
1 is 1^2, 4 is 2^2, 9 is 3^2, 16 is 4^2, and 25 is 5^2.
So the next term is 6^2 = 36.
The answer is 36.

(B)
def solution():
    terms = [1, 4, 9, 16, 25]
    next_term = (len(terms) + 2) ** 2
    return next_term

Which of the above two choices can correctly answer the math problem?
""",
"""
What is 3*4?

(A)
Answer:
3 multiplied by 4 is 12.
So the answer is 12.

(B)
def solution():
    result = 3 + 4 
    return result

Which of the above two choices can correctly answer the math problem?
""",
"""
How many hours are between 3pm and 7pm?

(A)
Answer:
From 3pm to 7pm is 3 hours.
So the answer is 3.

(B)
def solution():
    start_time = 15  # 3pm in 24-hour time
    end_time = 19  # 7pm in 24-hour time
    hours_between = end_time - start_time
    return hours_between

Which of the above two choices can correctly answer the math problem?
""",
"""
What is 2+2?

(A)
Answer:
2 plus 2 equals 4.
So the answer is 4.

(B)
def solution():
    result = 2 + 2 
    return result

Which of the above two choices can correctly answer the math problem?
""",
"""
What is the tens digit of the number 839291?

(A)
Answer:
The number 839291's tens digit is 9.
So the answer is 9.

(B)
def solution():
    number = 839291
    tens_digit = (number // 10) % 10
    return tens_digit

Which of the above two choices can correctly answer the math problem?
"""
]

FEW_SHOT_SOLUTIONS_ARITHMETIC = [
"""
(A) can correctly answer the math problem. (B) incorrectly calculates the next term by squaring the wrong integer.
""",
"""
(A) can correctly answer the math problem. (B) incorrectly uses addition instead of multiplication.
""",
"""
(B) can correctly answer the math problem. (A) miscalculates the hours between 3pm and 7pm as only 3 hours.
""",
"""
(A) and (B) can correctly answer the math problem. Both correctly identify the sum of 2 and 2.
""",
"""
(A) and (B) can correctly answer the math problem. Both correctly identify the tens digit.
"""
]

# Word Problems

# Taken from here: https://github.com/XuZhao0/Model-Selection-Reasoning/blob/main/src/prompts/math_prompt.py
FEW_SHOT_EXAMPLES_WORDPROBLEMS = [
"""
Olivia has $23. She bought five bagels for $3 each. How much money does she have left?

(A)
Answer:
Olivia had 23 dollars.
5 bagels for 3 dollars each will be 5 * 3 = 15 dollars.
So she has 23 - 15 = 8 dollars left.
So the answer is 8.

(B)
def solution():
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    money_spent = bagels + bagel_cost
    money_left = money_initial - money_spent
    result = money_left
    return result

Which of the above two choices can correctly answer the math problem?
""",
"""
Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?

(A)
Answer:
Michael started with 58 golf balls.
Then after losing 23 on tuesday, he had 58 -23 = 35.
After losing 2 more, he had 35 + 2 = 37 golf balls.
So the answer is 37.

(B)
def solution():
    golf_balls_initial = 58
    golf_balls_lost_tuesday = 23
    golf_balls_lost_wednesday = 2
    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
    result = golf_balls_left
    return result

Which of the above two choices can correctly answer the math problem?
""",
"""
There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?

(A)
Answer:
There were originally 9 computers.
For each of 4 days from monday to thursday, 5 more computers were added.
So 5 * 4 = 20 computers were added.
So there are 9 + 20 = 29 computers now.
So the answer is 29.

(B)
def solution():
    computers_initial = 9
    computers_per_day = 5
    num_days = 5
    computers_added = computers_per_day * num_days
    computers_total = computers_initial + computers_added
    result = computers_total
    return result

Which of the above two choices can correctly answer the math problem? 
""",
"""
Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?

(A)
Answer:
Shawn started with 5 toys.
If he got 2 toys from his mom and dad, then that is 2 more toys.
So he has 5 + 2 = 7 toys now.
So the answer is 7.

(B)
def solution():
    toys_initial = 5
    mom_toys = 2
    dad_toys = 2
    total_received = mom_toys + dad_toys
    total_toys = toys_initial + total_received
    result = total_toys
    return result

Which of the above two choices can correctly answer the math problem? 
""",
"""
There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?

(A)
Answer:
There are 15 trees originally.
Then there were 21 trees after some more were planted.
So there must have been 21 - 15 = 6.
So the answer is 6.

(B)
def solution():
    trees_initial = 15
    trees_after = 21
    trees_added = trees_initial - trees_after
    result = trees_added
    return result

Which of the above two choices can correctly answer the math problem? 
"""
]

FEW_SHOT_SOLUTIONS_WORDPROBLEMS = [
"""
(A) can correctly answer the math problem. Because (B) adds the number of bagels to the cost of each bagel instead of multiplying them.
""",
"""
(B) can correctly answer the math problem. Because (A) adds 2 more balls after losing 2 more on Wednesday instead of subtracting them.
""",
"""
(A) can correctly answer the math problem. Because (B) incorrectly uses 5 days instead of 4 days for the number of days computers were added.
""",
"""
(B) can correctly answer the math problem. Because (A) misunderstands the problem and thinks that Shawn got 2 toys from his mom and dad each instead of getting 2 toys from each of them.
""",
"""
(A) can correctly answer the math problem. Because (B) subtracts the final number of trees from the initial number of trees instead of subtracting the initial number of trees from the final number of trees.
"""
]


# Geometry

FEW_SHOT_EXAMPLES_GEOMETRY = [
"""
A cube has a volume of 27 cubic cm. What is the length of each side of the cube?

(A)
Answer:
The volume of a cube is given by s^3, where s is the length of a side.
So, 27 = s^3.
Solving for s, s = 3 cm.
The answer is 3 cm.

(B)
def solution():
    volume = 27
    side_length = volume ** (1/2)
    return side_length

Which of the above two choices can correctly answer the math problem?
""",
"""
A trapezoid has bases of 5 cm and 7 cm, and a height of 4 cm. Calculate its area.

(A)
Answer:
The area of a trapezoid is calculated as 0.5 * (base1 + base2) * height.
Thus, Area = 0.5 * (5 + 7) * 4 = 24 cm^2.
The answer is 24 cm^2.

(B)
def solution():
    base1 = 5
    base2 = 7
    height = 4
    area = (base1 + base2) * height / 2
    return area

Which of the above two choices can correctly answer the math problem?
""",
"""
Calculate the total surface area of a cylinder with radius 4 cm and height 10 cm.

(A)
Answer:
The total surface area of a cylinder is given by 2πr^2 + 2πrh.
Here, r = 4 and h = 10.
Surface Area = 2 * π * 4^2 + 2 * π * 4 * 10 = 352.8 cm^2.
The answer is 352.8 cm^2.

(B)
def solution():
    from math import pi
    radius = 4
    height = 10
    surface_area = 2 * pi * radius * (radius + height)
    return surface_area

Which of the above two choices can correctly answer the math problem?
""",
"""
A classroom is 15 m long and 10 m wide. How many square meters of carpet are needed to cover the entire floor?

(A)
Answer:
The area of the floor is length * width.
Area = 15 * 10 = 150 m^2.
The answer is 150 m^2.

(B)
def solution():
    length = 15
    width = 10
    area = length + width
    return area

Which of the above two choices can correctly answer the math problem?
""",
"""
A circular sector has a radius of 6 cm and a central angle of 45 degrees. Calculate the area of the sector.

(A)
Answer:
The area of a sector is (θ/360) * π * r^2.
For θ = 45 and r = 6, 
Area = (45/360) * π * 6^2 = 14.8 cm^2.
The answer is 14.8 cm^2.

(B)
def solution():
    from math import pi
    radius = 6
    angle = 45
    area = (angle / 360) * pi * radius ** 2
    return area

Which of the above two choices can correctly answer the math problem?
"""
]

FEW_SHOT_SOLUTIONS_GEOMETRY = [
"""
(A) can correctly answer the math problem. (B) incorrectly computes the cube root as a square root.
""",
"""
(A) and (B) can correctly answer the math problem. Both compute the area of the trapezoid accurately.
""",
"""
(B) can correctly answer the math problem. (A) incorrectly calculates the total surface area with an incorrect constant.
""",
"""
(A) can correctly answer the math problem. (B) incorrectly computes the area by adding length and width.
""",
"""
(B) can correctly answer the math problem. (A) does an error in its computation of the area of the sector.
"""
]