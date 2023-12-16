### Step 2: Simple Web Chat Interface with Gradio and LangChain

![Web Chat Interfact](../images/gradio-simple-chat.png)

A simple conversation chat with LLama 2 LLM and Langchain, and interact with it uisng Python code is built on [step-1](../step1/README.md). In this tutorial, we build a web interface to interace with the LLM with Gradio.

###  What is Gradio?
Official explanation from Gradio. Gradio is one of the best ways to share your machine learning model, API, or data science workflow with others is to create an interactive app that allows your users or colleagues to try out the demo in their browsers. Gradio allows you to build demos and share them, all in Python. And usually in just a few lines of code! So let’s get started.

## [Simple Chatbot Interface](Gradio-Chat-Interface-Manual.ipynb)
This [sample](Gradio-Chat-Interface-Manual.ipynb) leverages the code developed in [step 1: running llama locally](../step1/Running-llama-locally.ipynb) and use Gradio to create simple chatbot interface that user can ask question to the LLM and get the response back.

In the simple LLM chat built in step 1, every query would be treated as an entirely independent input wihtout considering past interactions. To enable the conversation to be more coherent like conversation chat, a conversation memory is added to the prompt template.

The output of a LLM model is depending on the set objectives and expectations. Without it, the respond back from the LLM model might not be what you expected based on the input your provide so crafting effective prompts with adequate context is crucial. It guides the model in producing responses that are precisely aligned with the provided context.

The following prompt template tell the LLM model that it should play a large language model developer role and help engineer to develop LLM applicaiton with Langchain framework. The conversation or chat history place holder is added in the prompt template so it can be part of the prompt send to the model.

```python
template = """Assistant is a large language model developer.
Assistant able to help engineer to develop large language model application with langchain framework.

Human: what is langchain framework?
AI: langchain framework is a framework to develop large language model application.

{history}
Human: {input}
AI:"""
```

It is every simple to create a simple chat interface with Gradio by just one API call and a callback function. The chat interface callback function has the history parameter that provides chat history from the interface. The following predict callback function reformat the chat history into the format understand by the LLM model.

```python
def predict(message, history):
    # history is in array of array format [["human message", "ai response"], ["human message", "ai response"]..]
    # Reformat the history messages into the prompt format. Just a simple converation started with either
    # Human: or AI: prefix.
    history_messages = ""
    for human_message, ai_response in history:
        history_messages = history_messages + "\nHuman: " + human_message + "\nAI: " + ai_response

    resp = llm_chain.invoke({"input": message, "history": history_messages})
    return resp["text"].strip()
```

