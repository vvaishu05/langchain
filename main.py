"""LangChain Learning - Main File
This file serves as the main entry point for the LangChain course examples. It demonstrates how to use
the ChatGoogleGenerativeAI class to interact with Google's generative AI models"""

import os

import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def google_genai_chatbot():
    """Example of using the ChatGoogleGenerativeAI class to interact"""
    chatbot = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7)
    response = chatbot.invoke("What is the capital of France?")
    print(response.content)


def check_google_api():
    """Check if the Google API key is set and list available models."""
    # Ensure your key is set
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

    print("Available models that support content generation:")
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            # Use the name part after 'models/' in LangChain
            print(f"- {m.name.replace('models/', '')}")


def main():
    """Main function to run the chatbot and check Google API."""
    # print("Hello from langchain-course!")
    google_genai_chatbot()
    # check_google_api()


if __name__ == "__main__":
    main()
