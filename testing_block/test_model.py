import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
OPENAI_API_KEY = os.getenv("GPT_5_API_KEY")
#OPENAI_API_KEY = os.getenv("GPT_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

models = client.models.list()
for m in models.data:
    print(m.id)



