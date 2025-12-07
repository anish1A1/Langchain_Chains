from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


llm = HuggingFaceEndpoint(
    repo_id= "moonshotai/Kimi-K2-Thinking",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()
prompt = PromptTemplate(
    template='Generate 2 interesting facts about {topic}',
    input_variables=['topic']
)

chain = prompt | model | parser
# print(chain)

result = chain.invoke({'topic':'Pokemon'})
print(result)


print(chain.get_graph().draw_ascii())