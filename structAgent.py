from symptoms import answer_generation

def structAgent(prompt, structured_data, chatHistory):
    response = answer_generation(f"Structured Data: {structured_data}\nUser Query: {prompt}", chatHistory)
    return response
