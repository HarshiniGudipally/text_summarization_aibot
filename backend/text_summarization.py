from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from transformers import pipeline
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
from typing import List, Callable
from nltk.tokenize import sent_tokenize

load_dotenv()

import os

from transformers import PegasusTokenizer, PegasusForConditionalGeneration, BartTokenizer, BartForConditionalGeneration


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

pegasus_model_name = "google/pegasus-large"
pegasus_tokenizer = PegasusTokenizer.from_pretrained(pegasus_model_name)
pegasus_model = PegasusForConditionalGeneration.from_pretrained(pegasus_model_name)

bart_model_name = "facebook/bart-large-cnn"
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)


def calculate_max_tokens(text: str, length: str, tokenizer_func: Callable) -> int:
    proportion = determine_summary_length(text, length)
    total_tokens = len(tokenizer_func(text))
    return max(30, int(total_tokens * proportion))  # Ensure at least 30 tokens

def ensure_complete_ending(summary: str) -> str:
    sentences = sent_tokenize(summary)
    return ' '.join(sentences)

def determine_summary_length(text: str, length: str) -> float:
    word_count = len(text.split())
    if length == "short":
        return 0.2  # 20% of original length
    elif length == "medium":
        return 0.4  # 40% of original length
    else:  # long
        return 0.6  # 60% of original length

def chunk_text(text: str, max_chunk_length: int, tokenizer_func: Callable) -> List[str]:
    sentences = sent_tokenize(text)
    chunks, current_chunk, current_length = [], [], 0
    
    for sentence in sentences:
        sentence_length = len(tokenizer_func(sentence))
        if current_length + sentence_length > max_chunk_length:
            chunks.append(' '.join(current_chunk))
            current_chunk, current_length = [sentence], sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def iterative_summarization(text: str, max_length: int, summarize_func: Callable[[str, int], str]) -> str:
    chunks = chunk_text(text, 1000, lambda x: x.split())  # Simple tokenization for chunking
    summaries = [summarize_func(chunk, max_length // len(chunks)) for chunk in chunks]
    combined_summary = ' '.join(summaries)
    return summarize_func(combined_summary, max_length)

def summarize_text_pegasus(text: str, max_length: int) -> str:
    try:
        inputs = pegasus_tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
        summary_ids = pegasus_model.generate(
            inputs["input_ids"], 
            max_length=max_length,
            min_length=min(30, max_length - 10),  # Allow for shorter summaries
            length_penalty=1.0,  # Reduce penalty for shorter sequences
            num_beams=4, 
            early_stopping=True
        )
        return pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        logging.error(f"bart summarization error: {str(e)}")
        raise Exception(f"bart summarization failed: {str(e)}")



def summarize_text_bart(text: str, max_length: int) -> str:
    try:

        inputs = bart_tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
        summary_ids = bart_model.generate(
            inputs["input_ids"], 
            max_length=max_length, 
            min_length=max(30, max_length // 2), 
            length_penalty=2.0, 
            num_beams=4, 
            early_stopping=True
        )
        return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        logging.error(f"bart summarization error: {str(e)}")
        raise Exception(f"bart summarization failed: {str(e)}")

def summarize_text_openai(text: str, max_tokens: int) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text concisely."},
                {"role": "user", "content": f"Summarize the following text in about {max_tokens} tokens:\n\n{text}"}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI summarization error: {str(e)}")
        raise Exception(f"OpenAI summarization failed: {str(e)}")

def summarize_text(text: str, length: str, method: str) -> str:
    try:
        if method == "google-pegasus":
            max_tokens = calculate_max_tokens(text, length, pegasus_tokenizer.encode)
            summary = iterative_summarization(text, max_tokens, summarize_text_pegasus)
        elif method == "meta-bart":
            max_tokens = calculate_max_tokens(text, length, bart_tokenizer.encode)
            summary = iterative_summarization(text, max_tokens, summarize_text_bart)
        elif method == "openai":
                max_tokens = calculate_max_tokens(text, length, lambda x: x.split())
                summary = summarize_text_openai(text, max_tokens)
        else:
            raise ValueError("Invalid summarization method. Choose 'pegasus', 'bart', or 'openai'.")
    except Exception as e:
        logging.error(f"summarization error: {str(e)}")
        raise Exception(f"summarization failed: {str(e)}")
    return ensure_complete_ending(summary)
