from vertexai.generative_models import GenerationConfig
import google.generativeai as genai
from neo4j_graphrag.llm import LLMInterface, LLMResponse
import os
from typing import Optional, Union, List, Any
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.types import LLMMessage


class GeminiLLM(LLMInterface):

    def invoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """
        Invoke the LLM with the given input.
        """
        # Configure the model
        genai.configure(api_key=os.getenv("GENAI_API_KEY"))
        
        # Create a GenerationConfig object
        generation_config = GenerationConfig(
            temperature=0.0,
            # max_output_tokens=512,
            # top_k=40,
            # top_p=0.95,
            # stop_sequences=["\n"],
            # candidate_count=1,
        )

        # Set the model name
        llm = genai.GenerativeModel(self.model_name)
        
        # Generate a response using the model
        response = llm.generate_content(input)
        
        # Return the response
        return LLMResponse(content=response.text)
    
    async def ainvoke(self, input: str) -> LLMResponse:
        return self.invoke(input)