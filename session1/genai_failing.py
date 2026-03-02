from dotenv import load_dotenv
load_dotenv()

import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI

# 1. Standard Generative AI Call
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

print("--- Traditional GenAI ---")
question = "What is 34534 multiplied by 2342, and then divided by 4?"
response = llm.invoke(question)

print(f"Question: {question}")
print(f"Response: {response}")
print(f"Answer: {response.content}\n")
# Point out to the class that the LLM is just guessing the math based on text patterns.