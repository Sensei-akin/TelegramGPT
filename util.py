import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Any, Dict, List, Optional
from langchain.docstore.document import Document
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAIChat
import os
from langchain.prompts import PromptTemplate



# Use a shorter template to reduce the number of tokens in the prompt
template = """You are an AI conversational assistant to answer questions based on a context.
You are given data from a conversation and a question, you must help the user find the information they need. 
Your answers should be friendly, in the same language and should remain on the context of the question. 
If you are unable to answer the question, simply state that you do not know. Do not attempt to fabricate an answer.
---------
QUESTION: Hi guys, Can you recommend afro stores?
=========
Content: terms quality, one close Osloer Straße.\ngenerally, highly recommended(good quality cheap price) Afro stores Berlin?.\nAì dúpé ara eni.\nCool. Ese oo.\none close KFC Alexanderplatz.\nHello people, pls Afro stores around Alexanderplatz Ostkreuz? need something better bread😊.\nMad oo.\nOk thanks.\nhttps://m.youtube.com/watch?v=J8wjZabYBsE.\nHey Niyi, guy you. get touch asap.\n😂.\n<Media omitted>.\nNoSQL win.\n😌.\n<Media omitted>.\nIt’s expression. things like “first black Olympic medalist” “first black billionaire “..\nmakes sense. think every club.\nCos love way.\nneed qualifier “black”?.\nnice.\nGuys see this, black dude amazing https://www.youtube.com/watch?v=ulncPbTgPlo.\nThabk.\n💡💡💡💡💡💡💡💡💡💡💡💡💡💡💡💡💡💡💡💡💡💡💡💡💡💡💡💡💡💡💡💡💡💡💡💡 😁.\nPlease enlighten us.\nlol... know goes inside kitkat right. case little peculiar.
=========
FINAL ANSWER: Based on the text given, there are several recommendations for xyz in Berlin. Here are some options:

One close to Osloer Straße is highly recommended for good quality and cheap price.
There is one close to KFC Alexanderplatz.
Afro Roots is a shop located at Zoologisher Garten, which deals mainly with Afro beauty products.
Exo-markt Afro Shop has good quality yams, and the prices are okay. The shop can be contacted at 030 95625817.
---------
QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""

# template = """You are an AI conversational assistant to answer questions based on a context.
# You are given data from a conversation and a question, you must help the user find the information they need. 
# Your answers should be friendly, in the same language and should remain on the context of the question. 
# If you are unable to answer the question, simply state that you do not know. Do not attempt to fabricate an answer.
# ---------
# QUESTION: What re the options to send naira. Is flutterwave a good option?
# =========
# Content:The USD transfer option was introduced recently after CBN directive. *Afripay* seems to offer transfers to either USD or Naira accounts. Read their FAQs, they're still working on USD bank transfers. You'd have to have a USD domiciary account Maybe the words 'pickup locations' is confusing me Your account is a ‘Nigerian account’, same process/procedure I suppose. Thanks girl! What provisions are on ground for us that just want to transfer to our account? https://www.worldremit.
# Hi people, what do you use to send money to Nigeria apart from flutterwave? I used rewire today Bad rates but... 😭 My link https://app.rewire.to/signup/?code=EAWL 😁 I have rewrite already How was the KYC? I tried all morning but it just kept emailing me to resubmit documents. I got fed up of it I used my phone, so it asked me to take a picture of my passport and a selfie Want to spend less sending money abroad? Use the code QbGv0W to get first money transfer FREE with @TransferGo, #trytransfergo: https://trgo'
# You can send money to naija with https://send.flutterwave.com/ . They are very fast and have good rates Good Morning house.. I have 100e available  pm Q It is almost same rate with Azimo
# =========
# FINAL ANSWER: Based on the text given, there are several recommendations for xyz in Berlin. Here are some options:

# - Flutterwave send (https://send.flutterwave.com/)
# - Afripay (https://afripay.de/)
# - Rewire (https://rewire.to/)
# - WorldRemit (https://www.worldremit.com/)

# ---------
# QUESTION: {question}
# =========
# {summaries}
# =========
# FINAL ANSWER:"""

STUFF_PROMPT = PromptTemplate(
    template=template, input_variables=["summaries", "question"])


def text_to_docs(text) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=20,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks




def get_answer(docs: List[Document], query: str) -> Dict[str, Any]:
    """Gets an answer to a question from a list of Documents."""

    # Get the answer

    chain = load_qa_with_sources_chain(
        OpenAIChat(
            temperature=0.5,model_name="gpt-3.5-turbo",max_tokens=800),  
        chain_type="stuff",
        prompt=STUFF_PROMPT,
    )

    answer = chain(
        {"input_documents": docs, "question": query}, return_only_outputs=True
    )
    return answer