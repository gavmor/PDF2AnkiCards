import os
import llama_index
from llama_index.core.llms import ChatMessage
from llama_index.llms.gemini import Gemini
import time
from tqdm import tqdm
from google.generativeai.types import HarmCategory, HarmBlockThreshold


client = Gemini(
    api_key=os.getenv("GEMINI_API_KEY"),
    model="models/gemini-1.5-flash",
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE
    }
)


def inference_text(text):
    return f"Derive, from the following passage, a set of Anki (`question;answer`) flashcards:\n\n{text}"


def infer_flashcards(text):
    return client.chat([
        ChatMessage(role="system", content="You are an Anki flashcard generator. Questions are to precede answers, joined with a semicolon (;) in the following format:\n\nquestion;answer. You are only interested in facts inferred from the text, but assume the reader does NOT have access to the text. Extract isolated propositions that stand on their own. Each passage is but one of many."),
        ChatMessage(role="user", content=inference_text("The sky is blue.")),
        ChatMessage(
            role="assistant", content="What color is the sky?;Blue.\nWhat, in the text, is blue?;The sky."),
        ChatMessage(role="user", content=inference_text(text)),
    ]).message.content
