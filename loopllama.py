import discord
from discord.ext import commands
import nest_asyncio
import asyncio
from llama_cpp import Llama

intents = discord.Intents.default()
intents.typing = False
intents.presences = False

bot = commands.Bot(command_prefix='!', intents=intents)

llm = Llama(model_path="C:\\Users\\Shadow\\research\\gpt-llama-falcon-intercommunication\\ggml-vic7b-uncensored-q5_1.bin")

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
bot.run('bottoken')
