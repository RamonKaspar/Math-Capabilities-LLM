# =============== FEW SHOT EXAMPLES ===============
""" 
Below are all the user prompts for the few shot prompting part (i.e. example questions). 
These examples are the same for all  techniques to ensure consistency and such direct that
comparison is possible.  Currently we have 5 example questions for each of the 3 datasets.

NOTE: When adding new examples or if you change them, make sure to adopt all the corresponding
system prompts!!
"""

FEW_SHOT_EXAMPLES_ARITHMETIC = [
    "What is the next term in 1, 4, 9, 16, 25?",
    "What is 3*4?",
    "How many hours are between 3pm and 7pm?",
    "What is 2+2?",
    "What is the tens digit of the number 839291?"
]

# These prompts where taken from: https://github.com/reasoning-machines/pal/blob/main/pal/prompt/math_prompts.py
FEW_SHOT_EXAMPLES_WORDPROBLEMS = [
    "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
    "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
    "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
    "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
    "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?"
]


FEW_SHOT_EXAMPLES_GEOMETRY = [
    "A cube has a volume of 27 cubic cm. What is the length of each side of the cube?",
    "A trapezoid has bases of 5 cm and 7 cm, and a height of 4 cm. Calculate its area.",
    "Calculate the total surface area of a cylinder with radius 4 cm and height 10 cm.",
    "A classroom is 15 m long and 10 m wide. How many square meters of carpet are needed to cover the entire floor?",
    "A circular sector has a radius of 6 cm and a central angle of 45 degrees. Calculate the area of the sector."
]

def get_few_shot_examples(dataset: str) -> list[str]:
    if dataset == 'arithmetic':
        return FEW_SHOT_EXAMPLES_ARITHMETIC
    elif dataset == 'wordProblems':
        return FEW_SHOT_EXAMPLES_WORDPROBLEMS
    elif dataset == 'geometry':
        return FEW_SHOT_EXAMPLES_GEOMETRY
    else:
        raise ValueError("Unsupported dataset type")

# ============ SYSTEM PROMPTS ===============
""" 
Below are all the system prompts. In order to properly compare the methods, 
we used the same prompts for every technique!
"""

SYSTEM_ARITHMETIC = 'You are a helpful assistant that can solve arithmetic problems.'
SYSTEM_WORDPROBLEMS = 'You are a helpful assistant that can solve math word problems.'
SYSTEM_GEOMETRY = 'You are a helpful assistant that can solve geometry problems.'

def get_system_prompt(dataset: str) -> str:
    if dataset == 'arithmetic':
        return SYSTEM_ARITHMETIC
    elif dataset == 'wordProblems':
        return SYSTEM_WORDPROBLEMS
    elif dataset == 'geometry':
        return SYSTEM_GEOMETRY
    else:
        raise ValueError("Unsupported dataset type")