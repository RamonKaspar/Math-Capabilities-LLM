from typing import List
from openai.types.chat import ChatCompletionMessageParam


def create_prompt_gpt35(system_prompt: str, introduction: str, question_prelude: str, question: str, few_shot_examples: List[str], few_shot_answers: List[str]) -> ChatCompletionMessageParam:
    """
    This function creates the prompt for the GPT-3.5 and GPT-4 model. It adds the system prompt, 
    the introduction, the question prelude, the question and the few shot examples and answers to the conversation.

    Args:
        system_prompt (str): The system prompt that is added once at the beginning of the conversation
        introduction (str): The introduction, i.e. how to solve it
        question_prelude (str): Introduces the question, e.g. "Question: " or "Answer the following question in Python: "
        question (str): The question that has to be answered
        few_shot_examples (List[str]): List of few shot examples
        few_shot_answers (List[str]): List of answers corresponding to the few shot examples

    Returns:
        ChatCompletionMessageParam: Prompt for the GPT-3.5 and GPT-4 model
        NOTE: For other models, you (may) have to build the prompts differently
    """
    assert len(few_shot_examples) == len(few_shot_answers), "Few shot examples and answers should be of the same length"
    messages = []
    # Add system prompt
    messages.append({"role": "system", "content": system_prompt})
    # Add introduction and first example as user prompt
    messages.append({"role": "user", "content": introduction + "\n" + "Here is one example how to do it:" + "\n\n" + question_prelude + few_shot_examples[0] + "\n\n" + few_shot_answers[0] + "\n\n" + "Now it's your turn."})
    # Add remaining four examples and answers to the conversation
    for i in range(1, len(few_shot_examples)):        
        messages.append({"role": "user", "content": question_prelude + " " + few_shot_examples[i]})
        messages.append({"role": "assistant", "content": few_shot_answers[i]})
    # Prepare the final question that has to be answeted
    messages.append({"role": "user", "content": question_prelude + question})
    return messages
    


import func_timeout

# Code is taken from here: https://github.com/XuZhao0/Model-Selection-Reasoning/blob/main/src/tool.py
def execute_solution_function(code_string: str):
    """
    Executes Python code that is expected to define and return the result of a function named 'solution'.
    The function compiles the code in a restricted environment and attempts to call 'solution()'.
    
    Args:
        code (str): A string of Python code which should contain a function definition called 'solution'.
    
    Returns:
        The return value of the 'solution()' function if successfully called. Returns None if the function
        does not exist, if an error occurs during its execution, or if the code does not adhere to safety constraints.
    """
    def execute(x, code_return):
        try:
            exec(x)
            locals_ = locals()
            # 
            solution = locals_.get('solution', None)
            if solution is not None:
                return solution()
            else:
                executed_code = 'import math\n' + 'import datetime\n' + \
                    '\n'.join([xx[4:]
                                for xx in x.strip().split('\n')[1:-1]])
                exec(executed_code)
                locals_ = locals()
                return locals_.get(code_return, None)

        except Exception as exp:
            print('Executing code error', exp)
            return None

    # === find code snippets between def solution(): and return ===
    try:
        code_list = code_string.strip().split('\n')

        new_code_list = []
        all_codes = []
        code_return = 'ans'

        for i in range(len(code_list)):
            if code_list[i].strip() == 'def solution():':
                new_code_list.append(code_list[i])
                for j in range(i+1, len(code_list)):
                    if code_list[j].startswith('    '):
                        new_code_list.append(code_list[j])
                    if 'return ' in code_list[j]:
                        code_return = code_list[j].split('return ')[1].strip()
                all_codes.append('\n'.join(new_code_list))
                new_code_list = []
        new_code = all_codes[-1]
        ans = func_timeout.func_timeout(
            3, execute, args=(new_code, code_return,))
        ans = ans if ans is not None else ans
    except func_timeout.FunctionTimedOut:
        ans = None
    try:
        ans = float(ans) if ans is not None else ans
    except:
        ans = None
    return ans