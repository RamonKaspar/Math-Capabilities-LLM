from .TechniqueInterface import TechniqueInterface

import re

from .util import extract_number

class DeclarativeSymPy(TechniqueInterface):
    
    def __init__(self, name: str, few_shot_prompting: bool, dataset: str, service: str, model: str, temperature: float, max_token: int):
        super().__init__(name, few_shot_prompting, dataset, service, model, temperature, max_token)
    
    def query(self, question: str) -> tuple[float, str, int, int]:
        response, prompt_tokens, completion_tokens = self.get_llm_response(question)
        # Extract the answer from the response
        eq_list = re.findall(r'\[\[.*?\]\]', response)
        if len(eq_list) > 0:   
            # Implementation see below         
            tmp = reformat_equations_from_peano(eq_list)
            return get_final_using_sympy(tmp), response, prompt_tokens, completion_tokens
        else:
            return extract_number(response), None, prompt_tokens, completion_tokens # Try our best

    def get_chat_introduction(self) -> str:
        return CHAT_INTRODUCTION
    
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



# =============== IMPLEMENTATION ===============
# Taken from offical paper github repo: https://github.com/joyheyueya/declarative-math-word-problem

from sympy import solve, sympify, Symbol
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
import string


def reformat_equations_from_peano(eq_list: str) -> str:
    """
    Reformats equations from a Peano format string by extracting equations or answers.
    E.g. from "eq a = 1, eq b = 2, answer c" to "a = 1, b = 2, c = ?"
    """
    try:
        result = ''
        for eq in eq_list:
            if 'eq' in eq:
                if len(result) == 0:
                    result += eq[eq.index('eq') + 2:-2]
                else:
                    result += ', ' + eq[eq.index('eq') + 2:-2]
            elif 'answer' in eq:
                if len(result) == 0:
                    result += eq[eq.index('answer') + 6:-2].strip() + ' = ?'
                else:
                    result += ', ' + eq[eq.index('answer') + 6:-2].strip() + ' = ?' 
        return result
    except Exception as e:
        return None

def get_final_using_sympy(equations):
    """
    Solves mathematics equations provided in a string format using SymPy.

    Args:
        equations (str): String containing equations separated by commas.

    Returns:
        float or None: The result of the equations solved, or None if an exception occurs.
    """
    try:
        transformations = (standard_transformations + (implicit_multiplication_application,) + (convert_xor,))
        if str(equations) is None or str(equations) == 'nan':
            return None
        equation_list = equations.split(',')
        for eq in equation_list:
            for c in range(len(eq)):
                if c < len(eq) - 2:
                    if eq[c].isalpha() and eq[c+1].isalpha() and eq[c+2].isalpha():
                        return None

        goal_var = None
        goal_expression_list = []
            
        if equation_list[-1].split('=')[0].strip().isalpha() or len(equation_list[-1].split('=')[0].strip()) == 2:
            goal_var = equation_list[-1].split('=')[0].strip()
        elif '=' in equation_list[-1]:
            for l in list(string.ascii_lowercase) + list(string.ascii_uppercase):
                if l not in equation_list[-1]:
                    goal_var = l
                    break
            if goal_var is not None:
                goal_expression = goal_var + ' - (' + equation_list[-1].split('=')[0].strip() + ')'
                goal_expression = parse_expr(goal_expression, transformations=transformations)
                goal_expression = sympify(goal_expression)
                try:
                    return float(solve(goal_expression)[0])
                except Exception as e:
                    pass
                goal_expression_list.append(goal_expression)
            else:
                return None

        if len(equation_list) == 1:
            try:
                goal_expression = parse_expr(equation_list[0].split('=')[0], transformations=transformations)
                return float(sympify(goal_expression))
            except Exception as e:
                return None

        if goal_var == None:
            return None

        for i in range(len(equation_list) - 1):
            sub_eqs = equation_list[i]  
            if '?' not in sub_eqs:
                try:    
                    sub_eqs_split = sub_eqs.split('=')
                    sub_eqs = sub_eqs_split[0].strip() + ' - (' + sub_eqs_split[1].strip() + ')'
                    sub_eqs = parse_expr(sub_eqs, transformations=transformations)
                    sub_eqs = sympify(sub_eqs)
                except Exception as e:
                    return None
                goal_expression_list.append(sub_eqs)

                try:
                    try:
                        return float(solve(goal_expression_list)[Symbol(goal_var)])
                    except Exception as e:
                        return float(solve(goal_expression_list)[0][Symbol(goal_var)])
                except Exception as e:
                    pass

        return None
    except Exception as e:
        return None



# ======== PROMPTS ================

# I added the sixth rule to the chat introduction 
CHAT_INTRODUCTION ="""Let's solve mathematical word problems in a careful, formal manner. The solution will follow the Peano format: 
1- Each sentence in the solution either introduces a new variable or states a new equation. 
2- The last sentence gives the goal: which variable will contain the answer to the problem. 
3- Each equation only uses previously introduced variables. 
4- Each quantity is only named by one variable.
5- Use all the numbers in the question.
6- Use only python SymPy syntax for all operations (e.g. **0.5 instead of sqrt, and ceiling/floor instead of round(x,0)).
"""

FEW_SHOT_SOLUTIONS_ARITHMETIC = [
"""
Let n be the position in the sequence of squares of natural numbers [[var n]]. We're given [[eq n = 6]].
Let a_1, a_2, a_3, a_4, a_5 be the initial terms of the sequence, with values [[eq a_1 = 1]], [[eq a_2 = 4]], [[eq a_3 = 9]], [[eq a_4 = 16]], [[eq a_5 = 25]].
Let a_n be the term in the sequence at position n [[var a_n]]. We derive [[eq a_n = n^2]] based on the pattern established by the initial terms.
The answer is the value of a_n [[answer a_n]].
""",
"""
Let a be the first factor [[var a]]. We're given [[eq a = 3]].
Let b be the second factor [[var b]]. We're given [[eq b = 4]].
Let p be the product of a and b [[var p]]. We have [[eq p = a * b]].
The answer is the value of p [[answer p]].
""",
"""
Let t_1 be the start time in hours [[var t_1]]. We're given [[eq t_1 = 15]] (since 3 PM is 15 hours from midnight).
Let t_2 be the end time in hours [[var t_2]]. We're given [[eq t_2 = 19]] (since 7 PM is 19 hours from midnight).
Let d be the difference in hours between t_1 and t_2 [[var d]]. We have [[eq d = t_2 - t_1]].
The answer is the value of d [[answer d]].
""",
"""
Let a be the first addend [[var a]]. We're given [[eq a = 2]].
Let b be the second addend [[var b]]. We're given [[eq b = 2]].
Let s be the sum of a and b [[var s]]. We have [[eq s = a + b]].
The answer is the value of s [[answer s]].
""",
"""
Let n be the number [[var n]]. We're given [[eq n = 839291]].
Let d be the tens digit of n [[var d]]. We have [[eq d = (n // 10) % 10]].
The answer is the value of d [[answer d]].
"""
]

# Prompts are taken from here: https://github.com/joyheyueya/declarative-math-word-problem/blob/main/prompts/declarative_eight_shot.py
FEW_SHOT_SOLUTIONS_WORDPROBLEMS = [
"""
Let a be the amount of money Olivia started with [[var a]]. We're given [[eq a = 23]]. 
Let b be the number of bagels she bought [[var b]]. We're given [[eq b = 5]].
Let c be how much each bagel costs [[var c]]. We're given [[eq c = 3]].
Let d be the total amount of money she spent on bagels [[var d]]. We have [[eq d = b * c]].
Let e be the amount of money she has left [[var e]]. We have [[eq e = a - d]].
The answer is the value of e [[answer e]].
""",
"""
Let a be the number of golf balls Michael started with [[var a]]. We're given [[eq a = 58]]. 
Let b be the number of golf balls he lost on tuesday [[var b]]. We're given [[eq b = 23]].
Let c be the number of golf balls he lost on wednesday [[var c]]. We're given [[eq c = 2]].
Let d be the number of golf balls he had left [[var d]]. We have [[eq d = a - b - c]].
The answer is the value of d [[answer d]].
""",
"""
Let a be the number of computers in the room [[var a]]. We're given [[eq a = 9]]. 
Let b be the number of computers installed each day [[var b]]. We're given [[eq b = 5]].
Let c be the number of days computers were installed [[var c]]. Since computers were installed from monday to thursday, we know that [[eq c = 4]].
Let d be the number of computers installed [[var d]]. We have [[eq d = b * c]].
Let e be the total number of computers in the room now [[var e]]. We have [[eq e = a + d]].
The answer is the value of e [[answer e]].
""",
"""
Let a be the number of toys Shawn started [[var a]]. We're given [[eq a = 5]]. 
Let b be the number of toys Shawn got for Christmas from his dad [[var b]]. We're given [[eq b = 2]].
Let c be the number of toys Shawn got for Christmas from his mom [[var c]]. We're given [[eq c = 2]].
Let d be the number of toys Shawn has now [[var d]]. We have [[eq d = a + b + c]].
The answer is the value of d [[answer d]].
""",
"""
Let a be the number of lollipops Jason had [[var a]]. We're given [[eq a = 20]]. 
Let b be the number of lollipops Jason gave Denny [[var b]].
Let c be the number of lollipops Jason has now [[var c]]. We have [[eq c = a - b]]. We're given that [[eq c = 12]].
The answer is the value of b [[answer b]].
"""
]

FEW_SHOT_SOLUTIONS_GEOMETRY = [
"""
Let a be the volume of the cube [[var a]]. We're given [[eq a = 27]].
Let s be the length of each side of the cube [[var s]].
Let s^3 be the expression for the volume of the cube, hence [[eq s^3 = a]].
The answer is the value of s [[answer s]].
""",
"""
Let a be the length of the first base of the trapezoid [[var a]]. We're given [[eq a = 5]].
Let b be the length of the second base of the trapezoid [[var b]]. We're given [[eq b = 7]].
Let h be the height of the trapezoid [[var h]]. We're given [[eq h = 4]].
Let A be the area of the trapezoid [[var A]]. We have [[eq A = (1/2) * (a + b) * h]].
The answer is the value of A [[answer A]].""",
"""
Let r be the radius of the cylinder [[var r]]. We're given [[eq r = 4]].
Let h be the height of the cylinder [[var h]]. We're given [[eq h = 10]].
Let A_s be the lateral surface area of the cylinder [[var A_s]]. We have [[eq A_s = 2 * pi * r * h]].
Let A_b be the area of one circular base of the cylinder [[var A_b]]. We have [[eq A_b = pi * r^2]].
Let A be the total surface area of the cylinder [[var A]]. We have [[eq A = 2 * A_b + A_s]].
The answer is the value of A [[answer A]].
""",
"""
Let l be the length of the classroom [[var l]]. We're given [[eq l = 15]].
Let w be the width of the classroom [[var w]]. We're given [[eq w = 10]].
Let A be the area of the classroom floor that needs carpet [[var A]]. We have [[eq A = l * w]].
The answer is the value of A [[answer A]].
""",
"""
Let r be the radius of the circular sector [[var r]]. We're given [[eq r = 6]].
Let t be the central angle of the circular sector in radians [[var t]]. Since 1 degree equals 𝜋/180 radians, we have [[eq t = 45 * (pi/180)]].
Let A be the area of the circular sector [[var A]]. We have [[eq A =(1/2) * r^2 * t]].
The answer is the value of A [[answer A]].
"""
]