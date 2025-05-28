from neo4j_graphrag.llm import OpenAILLM
import os
import mlflow
from components.gemini_llm import GeminiLLM
from time import sleep
from components.prompt_perturbation import character, word, sentence

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

def prompt_perturbation(prompt, perturbation_type, level=0.05):
    if perturbation_type is None or perturbation_type == "original":
        return prompt
    
    character_tool = character.CharacterPerturb(sentence=prompt, level=level)
    word_tool = word.WordPerturb(sentence=prompt, level=level)
    sentence_tool = sentence.SentencePerturb(sentence=prompt)
    if perturbation_type == "character_replacement":
        perturbed_prompt = character_tool.character_replacement()
    elif perturbation_type == "character_deletion":
        perturbed_prompt = character_tool.character_deletion()
    elif perturbation_type == "character_insertion":
        perturbed_prompt = character_tool.character_insertion()
    elif perturbation_type == "character_swap":
        perturbed_prompt = character_tool.character_swap()
    elif perturbation_type == "keyboard_typos":
        perturbed_prompt = character_tool.keyboard_typos()
    elif perturbation_type == "optical_character_recognition":
        perturbed_prompt = character_tool.optical_character_recognition()
    elif perturbation_type == "synonym_replacement":
        perturbed_prompt = word_tool.synonym_replacement()
    elif perturbation_type == "word_insertion":
        perturbed_prompt = word_tool.word_insertion()
    elif perturbation_type == "word_swap":
        perturbed_prompt = word_tool.word_swap()
    elif perturbation_type == "word_deletion":
        perturbed_prompt = word_tool.word_deletion()
    elif perturbation_type == "insert_punctuation":
        perturbed_prompt = word_tool.insert_punctuation()
    elif perturbation_type == "word_split":
        perturbed_prompt = word_tool.word_split()
    elif perturbation_type == "back_translation_hugging_face":  
        perturbed_prompt = sentence_tool.back_translation_hugging_face()
    # elif perturbation_type == "back_translation_google":
    #     perturbed_prompt = sentence_tool.back_translation_google()
    elif perturbation_type == "paraphrase":
        perturbed_prompt = sentence_tool.paraphrase()
    elif perturbation_type == "formalization":
        perturbed_prompt = sentence_tool.formal()
    elif perturbation_type == "casualization":  
        perturbed_prompt = sentence_tool.casual()
    elif perturbation_type == "passive_voice":
        perturbed_prompt = sentence_tool.passive()
    elif perturbation_type == "active_voice":
        perturbed_prompt = sentence_tool.active()
    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}")
    return perturbed_prompt

def build_query_prompt(context, question, ans0, ans1, ans2, perturbation_type=None):
    if perturbation_type:
        context = prompt_perturbation(context, perturbation_type)
    return f"{context} {question} Answer with one of the following options: ({ans0}, {ans1}, {ans2}). Start with the correct answer option so it can be checked, optionally follow it with an explanation."
