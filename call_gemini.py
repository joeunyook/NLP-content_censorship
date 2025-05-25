# utils_llm.py
import os
import google.generativeai as genai

def setup_gemini():
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-pro")
    return model

def get_context_from_gemini(model, tweet_text, prompt=None):
    default_prompt = (
        "Explain the socio-political or cultural context that could help determine whether this content "
        "might be offensive or subject to moderation:"
    )
    full_prompt = prompt or default_prompt
    response = model.generate_content(f"{full_prompt}\n\nTweet: {tweet_text}")
    return response.text
