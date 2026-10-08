import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

client = None

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)


def ask_gemini(question, learning_context=""):

    if client is None:
        return {
            "success": False,
            "answer": "Gemini API key is not configured."
        }

    prompt = f"""
You are EduNova AI, a personalized learning assistant.

Explain things clearly and simply.

User question:
{question}
"""

    if learning_context.strip():
        prompt += f"""

Use this uploaded learning material when relevant:

{learning_context}
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        return {
            "success": True,
            "answer": response.text.strip()
        }

    except Exception as error:
        print("Gemini Error:", error)

        return {
          
        }