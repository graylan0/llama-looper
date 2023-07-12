import discord
from discord.ext import commands
import nest_asyncio
import asyncio
from llama_cpp import Llama
import requests
import json
import pickle
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from collections import deque
import random
import time
from textblob import TextBlob  # For sentiment analysis
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

# Initialize the Trideque matrix, AI memory, and vector database
class Node:
    def __init__(self, message):
        self.message = message
        self.responses = deque()

class UserCharacter:
    def __init__(self, name):
        self.name = name
        self.observations = []

trideque_matrix = Node("root")
ai_memory = []
vector_database = {}
prompt_history = []
user_character = UserCharacter("User")

# Define the maximum length, minimum relevance, and minimum novelty
MAX_LENGTH = 1000
MIN_RELEVANCE = 0.5
MIN_NOVELTY = 0.5

# Load a pre-trained Word2Vec model
word2vec_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

class GPTController:
    def __init__(self, trideque_matrix, ai_memory, vector_database):
        self.trideque_matrix = trideque_matrix
        self.ai_memory = ai_memory
        self.vector_database = vector_database

    def update_trideque_matrix(self, node, message, response, observation):
        # Update the Trideque matrix with the new message and response
        new_node = Node(message)
        node.responses.append(new_node)

        # Convert the message and response into vectors and store them in the vector database
        message_vector = word2vec_model[message]
        response_vector = word2vec_model[response]
        self.vector_database[message] = message_vector
        self.vector_database[response] = response_vector

        # Add the message, response, and observation to the AI memory
        self.ai_memory.append((message, response, observation))

        # If the AI memory has more than 1000 items, save it to a .txt file and clear the memory
        if len(self.ai_memory) > 1000:
            with open('ai_memory.txt', 'wb') as f:
                pickle.dump(self.ai_memory, f)
            self.ai_memory.clear()

        return new_node

    # Add other methods for generating prompts, evaluating prompts and responses, etc.

# Discord bot setup
intents = discord.Intents.default()
intents.typing = False
intents.presences = False

bot = commands.Bot(command_prefix='!', intents=intents)

llm = Llama(model_path="C:\\Users\\Shadow\\research\\gpt-llama-falcon-intercommunication\\ggml-vic7b-uncensored-q5_1.bin")

async def llama_generate(prompt, max_tokens=200):
    full_prompt = f":: {prompt}\nQ: "  # Add a default format for the prompt
    full_prompt += "I'm here to help you. Please tell me more about what you're experiencing.\n\n"
    full_prompt += "**Your Feelings**\n\n"
    full_prompt += "Can you describe how you're feeling right now?\n\n"
    full_prompt += "**Your Situation**\n\n"
    full_prompt += "Can you tell me more about what's happening?\n\n"
    full_prompt += "**Your Needs**\n\n"
    full_prompt += "What do you need right now?\n\n"
    full_prompt += "Q: "  # Add another "Q:" prefix to indicate that the bot should generate a response

    # Add observation and action tokens
    full_prompt += "\n[Observation]\n"
    full_prompt += "The user seems to be in need of assistance. They have provided some information about their feelings, situation, and needs.\n"
    full_prompt += "[Action]\n"
    full_prompt += "Generate a helpful and empathetic response based on the user's input.\n"

    # Ensure the full prompt doesn't exceed the maximum sequence length
    if len(full_prompt) > n_ctx:
        full_prompt = full_prompt[:n_ctx]

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

# Create an instance of GPTController
controller = GPTController(trideque_matrix, ai_memory, vector_database)

# Use a loop to simulate the AI receiving messages and generating responses
current_node = trideque_matrix
while True:
    # For the purpose of this example, we'll just use the current time as the message and response
    message = str(time.time())
    response = str(time.time())
    observation = "The user seems to be interacting with the system at " + str(datetime.now())
    user_character.observations.append(observation)
    current_node = controller.update_trideque_matrix(current_node, message, response, observation)

    # Sleep for 1 second before the next iteration
    time.sleep(1)
