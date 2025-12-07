from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from pydantic import BaseModel, Field
from typing import Literal 

# We will get customer feedback add it to model, then if the response is postive or negative it will be sent to another models and response will be generated to that user.


# Model for Summarizing
llm = HuggingFaceEndpoint(
    repo_id= "moonshotai/Kimi-K2-Thinking",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Model for Quiz
llm2 = HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-20b',
    task='text-generation'
)
model2 = ChatHuggingFace(llm=llm2)


class feedback(BaseModel):
    sentiment:Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser = PydanticOutputParser(pydantic_object=feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n{format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser

print(classifier_chain.invoke({'feedback': 'This smartphone is really fast! But I will not buy again '}))