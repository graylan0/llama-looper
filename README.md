# Llama Looper
![image](https://user-images.githubusercontent.com/34530588/230510137-df3f346c-201a-4ac0-8055-20a411051f5b.png)
Looping the llama, into a superllama? 

https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/llama_cpp.py 
I was inspired by this project and @techcoderx. To make a Llama Loop.


# How to install

### Req's
Activate git lfs
`git lfs install`

Git clone the model
`git clone https://huggingface.co/eachadea/ggml-vicuna-7b-4bit`

Change directory
`cd C:\Users\Shadow\ggml-vicuna-7b-4bit`   
 
Install Pips
```
pip install llama-cpp-python
pip install discord
pip install nest_asyncio
pip install asyncio
```

create llama.py
```
import discord
from discord.ext import commands
import nest_asyncio
import asyncio
from llama_cpp import Llama

intents = discord.Intents.default()
intents.typing = False
intents.presences = False

bot = commands.Bot(command_prefix='!', intents=intents)

llm = Llama(model_path="C:\\Users\\Shadow\\ggml-vicuna-7b-4bit\\ggml-vicuna-7b-4bit-rev1.bin")

async def llama_generate(prompt, max_tokens=200):
    full_prompt = f":: {prompt}\nQ: "  # Add a default format for the prompt
    full_prompt += "Please provide a clear and concise description of the problem you're facing, along with any relevant code snippets or error messages.\n\n"
    full_prompt += "**Problem Description**\n\n"
    full_prompt += "Please describe the problem you're facing in a few sentences.\n\n"
    full_prompt += "**Code Snippet**\n\n"
    full_prompt += "Please provide the relevant code snippet(s) that are causing the issue.\n\n```python\n# Your code here\n```\n\n"
    full_prompt += "**Error Message**\n\n"
    full_prompt += "Please provide the error message you're seeing, if any.\n\n"
    full_prompt += "Q: "  # Add another "Q:" prefix to indicate that the bot should generate a response
    output = llm(full_prompt, max_tokens=max_tokens)
    return output

def generate_chunks(prompt, chunk_size=10):
    words = prompt.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

async def send_chunks(ctx, prompt_chunks):
    for chunk in prompt_chunks:
        llama_response = await llama_generate(chunk)
        llama_response = str(llama_response['choices'][0]['text'])  # Extract the text from the Llama model output

        response_parts = [llama_response[i:i + 10] for i in range(0, len(llama_response), 10)]

        initial_msg = await ctx.send(response_parts[0])

        for i, response_part in enumerate(response_parts[1:], start=1):
            await asyncio.sleep(0.5)
            await initial_msg.edit(content="".join(response_parts[:i + 1]))


@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name} (ID: {bot.user.id})')

@bot.command()
async def trideque(ctx, *, user_input):
    await ctx.send('Processing your input, please wait...')
    prompt_chunks = generate_chunks(user_input)
    await send_chunks(ctx, prompt_chunks)  # Make sure to pass 'ctx' as an argument here

# Replace 'your_bot_token' with your actual bot token from the Discord Developer Portal
nest_asyncio.apply()
bot.run('botokenhere')
```
 # Edit the model path to the correct path in your ENV
 `llm = Llama(model_path="C:\\Users\\Shadow\\ggml-vicuna-7b-4bit\\ggml-vicuna-7b-4bit-rev1.bin")`
 # Bot Token
 
Replace `bot.run('botokenhere')` "botokenhere" with a bottoken from discord developer's portal.

# Demo:
Looping Llamas:
![image](https://user-images.githubusercontent.com/34530588/230567353-3f3795aa-c1a6-4916-9819-9275c836dd1b.png)


https://github.com/abetlen and https://github.com/techcoderx . To build the looped llama

Add infinate loop generating llama memes.

![image](https://user-images.githubusercontent.com/34530588/230510166-7f64c0d5-8854-4689-a79e-cf99fbcc93fc.png)


and even more. (it just keeps looping llamas!)
![image](https://user-images.githubusercontent.com/34530588/230510671-ec46297d-d0de-4dfe-bca3-ce596ad098ee.png)
