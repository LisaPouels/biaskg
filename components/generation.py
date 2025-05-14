from neo4j_graphrag.llm import OpenAILLM
import os
import mlflow
from components.gemini_llm import GeminiLLM
from time import sleep

def initialize_llm(model: str):
    if model == "gemini-2.0-flash":
        sleep(30)
        mlflow.gemini.autolog()
        return GeminiLLM(model_name=model), 15
    elif model == "gpt-4.1-nano":
        mlflow.openai.autolog()
        return OpenAILLM(model_name=model, model_params={"temperature": 0}), None
    else:
        mlflow.openai.autolog()
        return OpenAILLM(
            model_name=model,
            model_params={"temperature": 0, "max_tokens": 1000},
            base_url=os.getenv("BASE_URL_OLLAMA")
        ), None
    
def build_query_prompt(context, question, ans0, ans1, ans2):
    return f"{context} {question} Answer with one of the following options: ({ans0}, {ans1}, {ans2}). Start with the correct answer option so it can be checked, optionally follow it with an explanation."
