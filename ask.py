from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
import os,argparse
from dotenv import load_dotenv
from os import getenv
from util import get_answer

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


def main():
    
    """
    The main function that runs the program.

    This function takes a query from the command line arguments, uses OpenAI's GPT-3 language model to search for similar
    text in a pre-indexed dataset, and generates an answer based on the search results.

    Parameters: None
    Returns: None
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('query', type=str, help='The query to search for.')
    args = parser.parse_args()
    query = args.query

    # Initialize OpenAIEmbeddings object with API key and chunk size
    embed = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'), chunk_size=1000)

    # Load pre-indexed dataset using FAISS
    index = FAISS.load_local('index v2', embed)

    # Use FAISS to search for similar text in the dataset
    search = index.similarity_search(query, k=5)

    # Generate a prompt for the GPT-3 language model to use for generating an answer
    prompt = f"The text below is a group conversation in English language and some part is in pidgin english. From the text answer this question: \
    {query}: Based on the text, what are some of the mentions in the conversations, \
    make a list of the answers and you must provide links if they appear in the conversation where necessary.\
    Ignore adding source to the answer"
    answer = get_answer(search,prompt)
    print(answer['output_text'])
    return answer['output_text']



if __name__ == "__main__":
    main()