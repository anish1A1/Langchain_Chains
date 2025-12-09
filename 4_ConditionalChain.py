from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal 

# runnable Branch is used for chain that needs to be used as if else statement

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
    sentiment:Literal['positive', 'negative'] = Field(description='Sentiment of the feedback')

parser = PydanticOutputParser(pydantic_object=feedback)

prompt1 = PromptTemplate(
    template=(
        "Classify the sentiment of the following feedback as 'positive' or 'negative'.\n"
        "You MUST respond ONLY in valid JSON. No explanation.\n\n"
        "Feedback: {feedback}\n\n"
        "Return output in this JSON format:\n"
        "{format_instruction}\n\n"
        "ONLY return JSON. No extra words."
    ),
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)


classifier_chain = prompt1 | model2 | parser

# print(classifier_chain.invoke({'feedback': 'This smartphone is really fast! But I will not buy again '}))


# Second phase 

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)


# We write in this way.
# branch_chain = RunnableBranch(
#     (condition1, chain1),
#     (condition2, chain2),
#     default chain
# )

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model2),
    (lambda x: x.sentiment == "negative", prompt3 | model2),
    RunnableLambda(lambda x: "Could not find sentiment")
)


chain = classifier_chain | branch_chain

output = chain.invoke({'feedback': 'This smartphone is really fast! But I will buy again '})

print(output.content)

