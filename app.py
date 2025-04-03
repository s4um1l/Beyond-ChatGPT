# You can find this code for Chainlit python streaming here (https://docs.chainlit.io/concepts/streaming/python)

# OpenAI Chat completion
import os
from openai import AsyncOpenAI  # importing openai for API usage
import chainlit as cl  # importing chainlit for our app
from chainlit.prompt import Prompt, PromptMessage  # importing prompt tools
from chainlit.playground.providers import ChatOpenAI  # importing ChatOpenAI tools
from dotenv import load_dotenv

load_dotenv()

# ChatOpenAI Templates
system_template = """You are a versatile, knowledgeable assistant with strong capabilities in:
1. Explaining technical concepts in simple terms
2. Summarizing and extracting key information
3. Creative writing and storytelling
4. Problem-solving and logical reasoning
5. Adapting your tone and style to different contexts

Always maintain a helpful, pleasant tone while providing comprehensive responses.
"""

# User template
user_template = """{input}
Think through your response step by step.
"""

# 1. Technical concept explanation template
explanation_template = """For explaining technical concepts to beginners:
- Start with a simple, jargon-free definition
- Use a relatable real-world analogy or metaphor
- Break down complex ideas into simpler components
- Provide concrete examples that illustrate the concept
- Explain practical benefits or applications
- End with a memorable summary comparison

Apply this approach to explain this concept: {input}"""

# 2. Summarization template
summary_template = """For summarization tasks:
- Read the full text carefully first
- Identify only the most important points (usually 3-7 key points)
- Use concise, clear language
- Organize points logically (chronological, importance, or topic-based)
- Use bullet points for clarity and scannability
- Ensure no critical information is lost
- Avoid adding your own interpretations or opinions

Summarize this text: {input}"""

# 3. Creative writing template
creative_template = """For creative writing tasks:
- Create a complete story with beginning, middle, and end
- Develop a clear central character with a goal or challenge
- Establish a vivid setting with sensory details
- Include an interesting complication or twist
- Resolve the story in a satisfying way
- Use descriptive language efficiently
- Stick precisely to the required word count
- Incorporate the specific theme or elements requested

Write a creative story with these requirements: {input}"""

# 4. Problem-solving template
problem_solving_template = """For math or logical problems:
- Read the problem carefully to identify what's being asked
- List all given information and constraints
- Break down the problem into smaller steps
- Show your work for each step with clear explanations dont use latex
- Check your solution against the original constraints
- Present the final answer clearly
- Verify the answer with a different approach if possible

Solve this problem step by step: {input}"""

# 5. Tone transformation template
tone_template = """For changing the tone of text:
- Identify the target tone (formal, casual, enthusiastic, etc.)
- Note key characteristics of that tone (vocabulary level, sentence structure, expressions)
- Preserve all important information from the original
- Replace informal phrases with more formal alternatives (or vice versa)
- Adjust sentence structure to match the desired tone
- Revise for consistency in tone throughout
- Ensure the message remains clear despite tone changes

Transform this text to the specified tone: {input}"""


@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():
    settings = {
        "model": "gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 1000,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    cl.user_session.set("settings", settings)


@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: cl.Message):
    settings = cl.user_session.get("settings").copy()

    client = AsyncOpenAI()

    print(message.content)
 # Detect task type and select appropriate template
    if any(term in message.content.lower() for term in ["explain", "concept", "explain this", "explain the concept"]):
        template_to_use = explanation_template
        # For explanations, lower temperature for clarity
        settings["temperature"] = 0.1
        
    elif any(term in message.content.lower() for term in ["summary", "summarize", "key points"]):
        template_to_use = summary_template
        # For summaries, low temperature for factual accuracy
        settings["temperature"] = 0.1
        
    elif any(term in message.content.lower() for term in ["story", "creative", "imaginative"]):
        template_to_use = creative_template
        # For creative writing, higher temperature
        settings["temperature"] = 0.7
        settings["max_tokens"] = 300  # Ensure enough space for creativity
        
    elif any(term in message.content.lower() for term in ["problem", "solve", "math", "how many"]):
        template_to_use = problem_solving_template
        # For math problems, zero temperature for accuracy
        settings["temperature"] = 0
        
    elif any(term in message.content.lower() for term in ["tone", "formal", "professional", "rewrite"]):
        template_to_use = tone_template
        # Moderate temperature for tone transformation
        settings["temperature"] = 0.3
        
    else:
        # Default template if no specific type is detected
        template_to_use = user_template
    
    # Create prompt with the selected template
    prompt = Prompt(
        provider=ChatOpenAI.id,
        messages=[
            PromptMessage(
                role="system",
                template=system_template,
                formatted=system_template,
            ),
            PromptMessage(
                role="user",
                template=template_to_use,
                formatted=template_to_use.format(input=message.content),
            ),
        ],
        inputs={"input": message.content},
        settings=settings,
    )

    print([m.to_openai() for m in prompt.messages])

    msg = cl.Message(content="")

    # Call OpenAI
    async for stream_resp in await client.chat.completions.create(
        messages=[m.to_openai() for m in prompt.messages], stream=True, **settings
    ):
        token = stream_resp.choices[0].delta.content
        if not token:
            token = ""
        await msg.stream_token(token)

    # Update the prompt object with the completion
    prompt.completion = msg.content
    msg.prompt = prompt

    # Send and close the message stream
    await msg.send()
