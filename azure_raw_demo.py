"""
This is just a demo file, created in order to run see how the AzureAOPENAI without langchain
"""

import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
)
 
# In raw client: pass deployment name as model=
response = client.chat.completions.create(
    model=os.getenv("AZURE_CHAT_DEPLOYMENT", "interns-gpt-4.1"),  # ← deployment name
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Say hello and confirm you're connected via Azure OpenAI."},
    ],
    temperature=0.2,
    max_tokens=100,
)
 
print("Response:", response.choices[0].message.content)
print("\n Azure OpenAI connection is working!")