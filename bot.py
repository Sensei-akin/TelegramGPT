import os
import telebot
from telebot import types
from dotenv import load_dotenv
load_dotenv()

BOT_TOKEN = os.getenv('BOT_TOKEN')
bot = telebot.TeleBot(BOT_TOKEN)


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
import os,argparse
from dotenv import load_dotenv
from os import getenv
from util import get_answer

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


def main(message):
    
    """
    The main function that runs the program.

    This function takes a query from the command line arguments, uses OpenAI's GPT-3 language model to search for similar
    text in a pre-indexed dataset, and generates an answer based on the search results.

    Parameters: None
    Returns: None
    """
    query = message.text

    # Initialize OpenAIEmbeddings object with API key and chunk size
    embed = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'), chunk_size=1000)

    # Load pre-indexed dataset using FAISS
    index = FAISS.load_local('index v2', embed)

    # Use FAISS to search for similar text in the dataset
    search = index.similarity_search(query, k=5)

    # Generate a prompt for the GPT-3 language model to use for generating an answer
    prompt = f"The text below is a group conversation in English language and some part is in pidgin english. From the text answer this question: {query}?,\
    In order to answer the question, here are some conditions to follow:\
        Ensure the answers provided remains in the context of the question.\
        Based on the text, what are some of the mentions in the conversations,\
        make a list of the answers and you must provide links if they appear in the conversation where necessary.\
        Ignore adding sources to the answer"
    answer = get_answer(search,prompt)
    final_ans = answer['output_text']
    bot.send_message(message.chat.id, "Here's your answer!")
    bot.send_message(message.chat.id, final_ans, parse_mode="Markdown")
    bot.send_message(message.chat.id, "If you have more questions, you can always start all queries with /ask")
    # print(answer['output_text'])
    return 


def create_start_button():
    markup = types.InlineKeyboardMarkup()
    start_button = types.InlineKeyboardButton("Start", callback_data="start")
    markup.add(start_button)
    return markup


@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    bot.reply_to(message, """This bot was created to answer frequently asked questions in the Community group.
Start all queries with /ask.
Feedback and /thanks: @sensei_akin""")
                 

@bot.message_handler(content_types=['new_chat_members'])
def new_member(message):
    for member in message.new_chat_members:
        if member.username == bot.get_me().username:
            send_welcome(message)
        else:
            welcome_text = f"Welcome {member.first_name}! Click the button below to start."
            bot.send_message(message.chat.id, welcome_text, reply_markup=create_start_button())

@bot.callback_query_handler(func=lambda call: call.data == "start")
def handle_start_button(call):
    send_welcome(call.message)

@bot.message_handler(commands=['ask'])
def sign_handler(message):
    text = """I am your friendly FAQbot, I am here to answer your questions. You can ask in English or Pidgin. I am able to understand.\nPlease ensure your question is clear. *Note that any answer provided is based on the conversation history and may or may not be accurate*"""
    sent_msg = bot.send_message(message.chat.id, text, parse_mode="Markdown")
    bot.register_next_step_handler(sent_msg, main)

print('start')
bot.polling()