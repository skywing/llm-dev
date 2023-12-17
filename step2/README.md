# Chat Interface with Gradio and LangChain

![Web Chat Interface](../images/gradio-simple-chat.png)

A simple conversation chat with LLama 2 LLM and Langchain, and interact with it uisng Python code is built on [step-1](../step1/README.md). In this tutorial, we build a web interface to interace with the LLM with Gradio.

- [Example 1: Simple Chat Interface](#example1)
- [Example 2: Simplify Chat History with Conversation Memory](#example2)
- [Example 3: Additional Gradio UI Elements to Configure LLM Model](#example3)
- [Example 4: Streaming chat message with OpenAI and Gradio UI](#example4)

###  What is Gradio?
Official explanation from Gradio. Gradio is one of the best ways to share your machine learning model, API, or data science workflow with others is to create an interactive app that allows your users or colleagues to try out the demo in their browsers. Gradio allows you to build demos and share them, all in Python. And usually in just a few lines of code! So let’s get started.

## <a name="example1">Example 1:</a> [Simple Chatbot Interface](Gradio-Chat-Interface-Manual.ipynb)
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

### The important of ending the input query correctly
One thing I learn from the LLama 2 model is that if you provide an input such as a questions without punctuation like period or question mark in the end of the input, the model might think the input is not complete and will try to predict and complete the sentence for you as part of the response. 

The following is one of the example input that is not a complete sentence or end with punctuation and you can see the model provide 2 responses (with AI: prefix). The first response is trying to predict and complete the initial input and the second is the response back to the combine human input and first response.
> Human: hello there
>
> AI: ! I'm an engineer and I'm interested in developing a large language model application using the LangChain framework. Can you help me get started?
>
> AI: Hello! Of course, I'd be happy to help! Developing a large language model application with LangChain can be a complex task, but it's definitely doable with the right guidance. 

Compare to the following end the first input with the "!" vs above without the "!".
> Human: hello there!
> 
> AI:  That's great! Can you tell me more about your project? What kind of language model are you trying to build, and what kind of tasks do you want it to be able to perform?
...

### The use of "STOP" word in Model Configuration
Depends on how you provide the input, the LLM model will sometime return multiple responses including and the responses might include human input it try to predict based on your initial quesitons. Like the model is trying to have a conversation by itself.


> Human: hello there.
>
> AI:  I am an engineer and I want to build a large language model using the LangChain framework. Can you help me?
>
> AI: Hello! Of course, I'd be happy to help! Building a large language model with LangChain can be a complex task, but it's definitely possible with the right guidance. What kind of language model are you interested in building? Are there any specific use cases or applications you have in mind?

The following human response to the previous AI reponse is part of the conversation generated by the LLM model.

> Human: Great! I want to build a language model that can generate creative writing, like poetry or short stories. I think it could be really cool if the model could generate new and interesting content on its own.
>
> AI: That's an exciting use case! Generating creative writing is definitely possible with LangChain....
>
> Human: Okay, that makes sense. But how do I actually go about building this
>
> Human: what is this?

To avoid responses back from the model like above in the UI. We set the "stop" parameter to prefix value of "Human" so the model will stop continue to generate responses when it encouter the prefix value we set on "stop" parameter when initialize the model like below.

```python
llm = LlamaCpp(
    model_path="PATH_TO_YOUR_MODEL",
    .
    .
    .
    stop=["Human"] #uncomment this and the model might continue generate response considered as part of the human input
)
```

## <a name="example2">Example 2:</a> [Simplify Code with Conversation Memory](Gradio-Chat-Interface-Memory.ipynb)

In previous example, the predict callback function includes code to reformat conversation history understand by the model. Langchain framework include the support of conversation memory in its library. 

The prompt is rewritten to include the use of message place holder that will store the chat history in the prompt message. System meesage is where you set the context for the LLM model.
```python
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
    You are a helpful AI is a large language model developer. 
    
    AI able to help engineer to develop large language model application with 
    langchain framework.
    
    Current conversation:"
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}"),
])
```

The following code initialize the conversation buffer memory with the window size value set to 20, and memory_key value set to the variable name provided in the message place holder in the prompt above.

```python
conversation_buffer = ConversationBufferWindowMemory(
    window_size=20, 
    memory_key="chat_history",
    return_messages=True)
```

To add the memory support ot the chain, the code includes the conversation buffer in the chain object initialization as shown below:

```python
llm_chain = LLMChain(
    llm=llm, 
    prompt=prompt, 
    memory=conversation_buffer,
    .
    .
)
```

The predict callback function used by the Gradio is rewritten and much more simplify by removing the code that reformat historical chat messages. 

*The LLama model I use in this case will include the "AI:" prefix in the response. It doesn't look good that is included in the chat messages so I add the code to strip "AI:" from the model response text*

```python
def predict(message, history):
    resp = llm_chain.invoke({"input": message})
    response_text = resp["text"].strip().replace("AI:", "")
    return response_text
```

## <a name="example3">Example 3:</a> [Additional Gradio UI Elements to Configure LLM Model](Gradio-Chat-UI-Elements.ipynb)

You can build complex UI in Gradio easily as the framework come with many libraries to allow you control its layout and many pre-built components are readily to be used such as the chat interface. 

![Gradio UI elements](../images/gradio-ui-elements.png)

The following example demonstrate how to use various out of the box components to get LLM model configuration values from user and update the it accordingly as shown in above demo.

```python
# use according to and group all teh LLM model configuration inputs
with gr.Accordion("LLM Model Configuration"):
    with gr.Group():
        with gr.Row():
            with gr.Column():
                n_gpu_layers = gr.Number(value=2, label="GPU Layers", minimum=1, maximum=4)
            with gr.Column():
                n_batch = gr.Number(label="Batch", minimum=256, maximum=2048, value=256, step=256)
            with gr.Column():
                n_ctx = gr.Number(label="Context", minimum=512, maximum=4096, value=2048, step=512)
        slider = gr.Slider(0, 1, step=0.05, label="Temperature", value=0.75)
        with gr.Row():
            is_verbose = gr.Checkbox(label="Verbose", value=True, info="verbose output")
    update_btn = gr.Button(value="Update")

chatui = gr.ChatInterface(
    predict,
    retry_btn=None,
    undo_btn=None,
    clear_btn=None,
    submit_btn="Send")

update_btn.click(
    fn=reconfigure_llm_model,
    inputs=[
        slider,
        n_gpu_layers,
        n_batch,
        n_ctx,
        is_verbose
    ],
    outputs=None
)
```

## <a name="example4">Example 4:</a> [ Streaming chat message with OpenAI and Gradio UI](Gradio-Chat-Interface-Streaming.ipynb)

The following is the example of using OpenAI API to stream the chat message directly to the Gradio chat interface.

![OpenAI Streaming with Gradio Interface](../images/gradio-openai-streaming-chat.gif)

