from llm_inference.llm_factory import get_llm_service
from .util import create_prompt_gpt35
from .shared_prompts import get_few_shot_examples, get_system_prompt

from abc import abstractmethod

class TechniqueInterface:
    """
    Abstract base class for implementing different query techniques on large language models (LLMs).
    
    This class provides a structured way to define and utilize various techniques for querying LLMs
    based on specific kinds of mathematical query datasets like arithmetic, word problems, or geometry.
    
    Attributes:
        name (str): Identifier for the technique.
        dataset (str): Category of dataset ('arithmetic', 'wordProblems', or 'geometry') which influences the prompting strategy.
        service (str): API service identifier used for LLM communication.
        model (str): Model identifier specifying the LLM to be queried.
        temperature (float): Decides the randomness in the generation of model responses, affecting creativity.
        max_token (int): Upper limit on the response size measured in tokens.
        client: An API client configured to communicate with the specified LLM service.

    Methods:
        get_llm_response(question: str) -> tuple[float, str, int, int]:
            Directly handles sending the formalized query to the LLM and receiving the response.
        query(question: str) -> tuple[float, str, int, int]:
            Processes the question to adapt it for the LLM querying, utilizing specific technique characteristics.
        query_with_detailed_response(question: str) -> dict:
            Wraps the query method to provide detailed response information including metadata.
        get_chat_introduction() -> str:
            Abstract method for returning an introduction text to the conversation.
        get_question_prelude() -> str:
            Abstract method for returning preparatory text to precede the questions, framing it contextually.
        get_few_shot_solutions() -> list[str]:
            Abstract method for retrieving pre-determined solutions for configured few-shot prompts.
    """

    def __init__(self, name: str, dataset: str, service: str, model: str, temperature: float, max_token: int):
        """
        Initializes attributes and configures the LLM client for querying.
        """
        assert dataset in ['arithmetic', 'wordProblems', 'geometry'], "Invalid dataset type is specified."
        self.name = name
        self.service = service
        self.model = model
        self.temperature = temperature
        self.max_token = max_token
        self.dataset = dataset
        self.client = get_llm_service(service, model, temperature, max_token)
    
    def get_llm_response(self, question) -> tuple[str, str, int, int]:
        """
        Sends a specified question to the configured LLM service and returns the raw response.

        This method is common to all techniques and provides the basic functionality to
        interact with the LLM, handling prompt creation and response retrieval.
        
        NOTE: If your LLM service provider differs from Azure or OpenAI, you may need to customize or override
        this method to accommodate your provider's specific API requirements and response formats. This might involve
        altering how prompts are constructed or how responses are parsed.
        
        Parameters:
            question (str): The math question to send to the model.
        
        Returns:
            tuple[str, str, int, int]: A tuple containing the raw answer, reasoning (if any), 
            prompt tokens, and completion tokens.
        """
        # NOTE: If you sure that this function works for your provider, comment this check out.
        if self.service not in ["azure", "openai"]:
            raise ValueError("You may have to adapt the function to your service provider.")
        
        messages = create_prompt_gpt35(
            system_prompt = get_system_prompt(self.dataset), 
            introduction = self.get_chat_introduction(), 
            question_prelude = self.get_question_prelude(), 
            question = question, 
            few_shot_examples = get_few_shot_examples(self.dataset), 
            few_shot_answers = self.get_few_shot_solutions()
        )
        return self.client.make_request(messages)
    
    @abstractmethod
    def query(self, question: str) -> tuple[float, str, int, int]:
        """
        Abstract method that must be implemented by each technique class. This method should
        utilize the response from get_llm_response and apply specific post-processing to derive the
        final result according to the technique's purpose, like executing code or extracting numeric values.        
        Parameters:
            question (str): The math question to send to the model.
        
        Returns:
            tuple[float, str, int, int]: A tuple containing the float answer, reasoning, prompt tokens, and completion tokens.
        """
        raise NotImplementedError
    
    def query_with_detailed_response(self, question: str) -> dict:
        """
        Executes a query using the implemented `query` method, adding detailed response information.
        
        Parameters:
            question (str): The math question to send to the model.
        
        Returns:
            dict: A dictionary containing detailed query and response metadata, useful for analysis and comparison.
        """
        answer, reasoning, prompt_tokens, completion_tokens = self.query(question)  
        data = {
            "technique": self.name,
            "question": question,
            "answer": answer,
            "reasoning": reasoning,
            "model": self.model,
            "temperature": self.temperature,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        return data
    
    @abstractmethod
    def get_chat_introduction(self) -> str:
        """ Returns a customized introduction for the chat session. """
        raise NotImplementedError
    
    @abstractmethod
    def get_question_prelude(self) -> str:
        """ Returns a formatted prelude to the questions, i.e. "Question: ". """
        raise NotImplementedError
    
    @abstractmethod
    def get_few_shot_solutions(self) -> list[str]:
        """
        Returns the few shot solutions for the dataset.
        
        NOTE: Any changes to few-shot examples must be synchronized with updates to this method.
        """
        raise NotImplementedError