from .TechniqueInterface import TechniqueInterface

from .util import extract_number

class CoT(TechniqueInterface):
    
    def __init__(self, name: str, few_shot_prompting: bool, dataset: str, service: str, model: str, temperature: float, max_token: int):
        super().__init__(name, few_shot_prompting, dataset, service, model, temperature, max_token)
    
    def query(self, question: str) -> tuple[float, str, int, int]:
        response, prompt_tokens, completion_tokens = self.get_llm_response(question)
        # Extract the answer from the response
        answer = extract_number(response, "So the answer is ")   
        return answer, response, prompt_tokens, completion_tokens

    def get_chat_introduction(self) -> str:
        return "Let's solve the following math problems. You need to solve these math problems step by step."
    
    def get_question_prelude(self) -> str:
        return "Think step by step. Question: "
    
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
"""
Answer:
The sequence given is 1, 4, 9, 16, 25.
These are the squares of the first five natural numbers.
1 = 1^2, 4 = 2^2, 9 = 3^2, 16 = 4^2, 25 = 5^2.
The next natural number is 6.
So, the next term is 6^2=36.
So the answer is 36.
""",
"""
Answer:
The problem is to find 3x4.
Note that 3x4 is the same as 3+3+3+3.
Thus, 3x4 = 3+3+3+3 = 12.
So the answer is 12.
""",
"""
Answer:
Start from 3 PM.
Count each hour after 3 PM until 7 PM: 4 PM, 5 PM, 6 PM, 7 PM.
This includes four time points: 4 PM, 5 PM, 6 PM, and 7 PM, which represent four hours.
Therefore, there are 4 hours between 3 PM and 7 PM.
So the answer is 4.
""",
"""
Answer:
Recognize the operation is addition.
Add the two numbers together: 2+2.
Result from adding two and two is four.
So the answer is 4.
""",
"""
Answer:
The number 839291 consists of the digits:
1 in the units place,
9 in the tens place,
2 in the hundreds place,
9 in the thousands place,
3 in the ten thousands place,
8 in the hundred thousands place.
The tens digit in the number 839291 is 9.
So, the answer is 9.
"""
]

# These prompts where taken from: https://github.com/reasoning-machines/pal/blob/main/pal/prompt/math_prompts.py
FEW_SHOT_SOLUTIONS_WORDPROBLEMS = [
"""
Answer:
Olivia had 23 dollars.
And she bought 5 bagels.
And each bagel costs 3 dollars.
So she spent 5 * 3 = 15 dollars.
So she has 23 - 15 = 8 dollars left.
So the answer is 8.
""",
"""
Answer:
Michael started with 58 golf balls.
And he lost 23 golf balls on tuesday.
So after losing 23 on tuesday, he had 58 -23 = 35.
And then he lost 2 more golf balls on wednesday.
So after losing 2 more on wednesday, he had 35 - 2 = 33 golf balls.
So the answer is 33.
""",
"""
Answer:
There were originally 9 computers.
And 5 more computers were added from monday to thursday.
There are 4 days between monday and thursday.
So 5 * 4 = 20 computers were added in total.
So there are 9 + 20 = 29 computers now.
So the answer is 29.
""",
"""
Answer:
Shawn started with 5 toys.
And he got 2 toys each from his mom and dad.
So he got 2 + 2 = 4 toys.
Therefore, he has 5 + 4 = 9 toys now.
So the answer is 9.
""",
"""
Answer:
Jason started with 20 lollipops.
Then he had 12 after giving some to Denny.
So he gave Denny 20 - 12 = 8 lollipops.
So the answer is 8.
"""
]

FEW_SHOT_SOLUTIONS_GEOMETRY = [
"""
Answer:
The volume V of a cube is calculated using the formula: V = s^3, where s is the length of one side of the cube.
The given volume is 27 cubic cm.
Calculate the side length: We need to solve for s in the equation:
s^3 = 27
To find s, we take the cube root of 27:
s = cube root of 27 = 3
So the answer is 3.
""",
"""
Answer:
The area A of a trapezoid can be found using the formula: 
A = (1/2) * (a + b) * h, where a and b are the lengths of the bases, and h is the height.
Here, a=5 cm, b = 7 cm, and h = 4 cm.
Substitute the values into the formula:
A = (1/2) * (5 + 7) * 4
Calculate the sum of the bases: 5 + 7 = 12
Complete the calculation: A = (1/2) * 12 * 4
A = 6 * 4 = 24
So the answer is 24.
""",
"""
Answer:
The formula for the total surface area A of a cylinder is: A = 2 * pi * r * (h + r), where r is the radius and h is the height.
For a cylinder with radius r = 4 cm and height h = 10 cm, we substitute the value in:
A = 2 * pi * 4 * (10 + 4) = 2 * pi * 4 * 14
A = 2 * pi * 56 = 112 * pi
Now we compute it numerically using pi approximately 3.14159:
A = 112 * 3.14159 = 351.85808
So the answer is 351.85808.
""",
"""
Answer:
The area A of a rectangle (which is the shape of the classroom floor) is calculated by multiplying the length by the width:
A = length * width
Values for calculation: length = 15 m and width = 10 m.
A = 15 * 10
A = 150
So the answer is 150.
""",
"""
Answer:
The area A of a circular sector is calculated using the formula:
A = (theta / 360) * pi * r^2, where theta is the central angle in degrees and r is the radius.
The radius r is 6 cm and the central angle theta is 45 degrees.
Substitute the values into the formula: 
A = (45 / 360) * pi * (6^2)
Evaluate 6^2 = 36
Simplify the fraction 45/360 = 1/8
Complete the calculation: A = (1 / 8) * pi * 36
A = 4.5 * pi
Now we compute it numerically using pi approximately 3.14159:
A = 4.5 * 3.14159 = 14.137155
So the answer is 14.137155.
"""
]