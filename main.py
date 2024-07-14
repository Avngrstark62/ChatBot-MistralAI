import time
from langchain_huggingface import HuggingFaceEndpoint


sec_key = 'my_secret_key'
repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1"
# repo_id="openai-community/gpt2-medium"
llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,token=sec_key)


from langchain import PromptTemplate, LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory


template = """The following is a conversation between a user and an AI assistant.
{history}
User: {user_input}
AI: """

prompt = PromptTemplate(
    template=template,
    input_variables=["history" ,"user_input"],
)
prompt


llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
)
llm_chain({'history':"History" ,'user_input':
           "What is Machine Learning?"})


def generate_response(input_dict):
    # Generate response using the HuggingFaceEndpoint model
    response = llm_chain(input_dict)['text']
    # Extract the AI's response part
    ai_response = response.split("AI:")[0].split("User:")[0].strip()
    return ai_response


history = ""


while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
    # Generate the complete prompt
    input_dict = {"history": history, "user_input": user_input}
    # Generate the AI response
    ai_response = generate_response(input_dict)
    # Add the user input and AI response to memory
    history += f"User: {user_input}\nAI: {ai_response}\n"
    # Print the response
    print('ChatBot:', ai_response)
    time.sleep(1)  # Wait for 1 second before sending the next request