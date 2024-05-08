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