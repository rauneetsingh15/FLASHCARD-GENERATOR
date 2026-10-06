import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto"
)

notes = """
Photosynthesis is the process by which green plants
make food using sunlight, carbon dioxide and water.
Chlorophyll absorbs sunlight.
Oxygen is released as a by-product.
"""

prompt = f"""
Create 5 study flashcards from these notes.

Rules:
- Give each flashcard a question and answer.
- Keep the answers short and easy.
- Use only information from the notes.
- Format exactly like this:

Q1: question
A1: answer

Q2: question
A2: answer

Notes:
{notes}
"""

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print(response.choices[0].message.content)
