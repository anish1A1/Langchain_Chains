from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


llm = HuggingFaceEndpoint(
    repo_id= "moonshotai/Kimi-K2-Thinking",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

template1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)


chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'Software Engineering UnEmployment Crisis'})

print(result)

print(chain.get_graph().draw_ascii())