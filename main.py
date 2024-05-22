import os

from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

load_dotenv()

llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

first_prompt = PromptTemplate(
    input_variables=["language", "task"],
    template="Write a very short {language} function that will {task}"
)

second_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a test for de following {language} code:\n{code}"
)

first_chain = LLMChain(llm=llm, prompt=first_prompt, output_key="code")

second_chain = LLMChain(llm=llm, prompt=second_prompt, output_key="test")

sequence_chain = SequentialChain(
    chains=[first_chain, second_chain],
    input_variables=["language", "task"],
    output_variables=["code", "test"]
)

result = sequence_chain(
    {"language": "python", "task": "return a list of numbers"})

print("GENERATED CODE:\n")
print(result["code"])
print("GENERATED TESTE:\n")
print(result["test"])
