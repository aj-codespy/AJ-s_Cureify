from router import routerAgent
from structAgent import structAgent

def mainAgent(prompt, img=None):
    chatHistory = []
    output = routerAgent(img, prompt, chatHistory)
    
    if 'unstructured' in output.lower():
        return structAgent(prompt, output, chatHistory)
    
    return output
