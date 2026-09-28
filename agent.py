import os
import requests

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing in .env")

client = genai.Client(
    api_key=GEMINI_API_KEY
)

MODEL_NAME = "gemini-3.5-flash-lite"


# ==========================================
# WEB SEARCH KEYWORDS
# ==========================================

WEB_KEYWORDS = [
    "latest",
    "today",
    "current",
    "recent",
    "news",
    "this week",
    "this month",
    "price",
    "stock",
    "weather",
    "update"
]


def needs_web_search(query):

    query_lower = query.lower()

    return any(
        word in query_lower
        for word in WEB_KEYWORDS
    )


# ==========================================
# FAST WEB SEARCH
# ==========================================

def search_web(query):

    try:

        response = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={
                "User-Agent": "Mozilla/5.0"
            },
            timeout=3
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        results = []

        for result in soup.select(".result")[:2]:

            title_element = result.select_one(
                ".result__title"
            )

            snippet_element = result.select_one(
                ".result__snippet"
            )

            if not title_element:
                continue

            results.append({
                "title": title_element.get_text(
                    " ",
                    strip=True
                ),

                "snippet":
                    snippet_element.get_text(
                        " ",
                        strip=True
                    )
                    if snippet_element
                    else ""
            })

        return results

    except Exception:

        return []


# ==========================================
# GEMINI CONFIG
# ==========================================

FAST_CONFIG = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(
        thinking_level="minimal"
    ),
    max_output_tokens=500
)


# ==========================================
# TEXT AGENT
# ==========================================

def run_agent(user_query):

    # --------------------------------------
    # FAST MODE
    # --------------------------------------

    if not needs_web_search(user_query):

        prompt = f"""
Answer this question directly.

Question:
{user_query}

Keep the answer concise and useful.
Use simple language.
Do not invent facts.
"""

        try:

            response = client.models.generate_content(

                model=MODEL_NAME,

                contents=prompt,

                config=FAST_CONFIG
            )

            return {
                "answer": response.text,
                "sources": []
            }

        except Exception as e:

            return {
                "answer": "",
                "sources": [],
                "error": str(e)
            }


    # --------------------------------------
    # WEB MODE
    # --------------------------------------

    search_results = search_web(
        user_query
    )

    context = ""

    for result in search_results:

        context += f"""
Title:
{result["title"]}

Information:
{result["snippet"]}

"""


    prompt = f"""
Answer the user's question.

Question:
{user_query}

Recent web information:
{context}

Give a concise answer.
Use the web information when useful.
Do not invent facts.
"""

    try:

        response = client.models.generate_content(

            model=MODEL_NAME,

            contents=prompt,

            config=FAST_CONFIG
        )

        return {
            "answer": response.text,
            "sources": search_results
        }

    except Exception as e:

        return {
            "answer": "",
            "sources": search_results,
            "error": str(e)
        }


# ==========================================
# VOICE AGENT
# ==========================================

def run_voice_agent(
    audio_bytes,
    mime_type="audio/webm"
):

    prompt = """
Listen to the user's voice.

Understand what the user is asking.

Answer the question directly.

Keep the answer concise and useful.

If the speech is unclear,
say that the audio was unclear.
"""

    try:

        audio_part = types.Part.from_bytes(
            data=audio_bytes,
            mime_type=mime_type
        )

        response = client.models.generate_content(

            model=MODEL_NAME,

            contents=[
                prompt,
                audio_part
            ],

            config=FAST_CONFIG
        )

        return {
            "answer": response.text,
            "sources": []
        }

    except Exception as e:

        return {
            "answer": "",
            "sources": [],
            "error": str(e)
        }