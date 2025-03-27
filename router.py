from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from symptoms import retrieve_and_answer
from query import queryAnalysis
from imageAgent import imgClassifier
import os  # Add this line at the top

def routerAgent(img, prompt, chatHistory):
    base = '''From the given prompt, determine the task:
    - If it is related to an image, return "image".
    - If it describes symptoms, return "symptom".
    - Otherwise, return "query".
    Respond only with one of these words: image, symptom, or query.'''

    Agent = ChatGoogleGenerativeAI(
        model='gemini-1.5-flash',
        temperature=0,
        api_key=os.getenv("AIzaSyDtB4bETfNDyvpzA_NnBKMrr56rdiOE8bQ"),
        max_tokens=None,
        timeout=30,
        max_retries=2
    )
    
    role = ChatPromptTemplate.from_messages([
        ('system', base),
        ('user', "{input}")
    ])

    chain = role | Agent
    response = chain.invoke({'input': prompt})
    output = response.content.strip().lower()

    if img:
        return imgClassifier(img, prompt)
    elif output == 'query':
        return queryAnalysis(prompt)
    elif output == 'symptom':
        return retrieve_and_answer(prompt, chatHistory)
    else:
        return "Invalid input"
