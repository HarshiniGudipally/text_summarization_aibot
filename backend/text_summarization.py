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
load_dotenv()

import nltk
import os

from transformers import PegasusTokenizer, PegasusForConditionalGeneration


def download_nltk_data(package):
    try:
        nltk.data.find(f'tokenizers/{package}')
        print(f"{package} is already downloaded.")
    except LookupError:
        print(f"Downloading {package}...")
        nltk.download(package, quiet=True)
        print(f"{package} has been downloaded.")

# Example usage
download_nltk_data('punkt')
download_nltk_data('punkt_tab')

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

model_name = "google/pegasus-large"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

def determine_summary_length(text, length):
    total_sentences = len(text.split('.'))
    if length == "short":
        return max(1, int(total_sentences * 0.2))
    elif length == "medium":
        return max(1, int(total_sentences * 0.4))
    else:
        return max(1, int(total_sentences * 0.6))

def sumy_summarize(text, sentence_count):
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        stemmer = Stemmer("english")
        summarizer = LsaSummarizer(stemmer)
        summarizer.stop_words = get_stop_words("english")
        summary = summarizer(parser.document, sentence_count)
        return " ".join([str(sentence) for sentence in summary])
    except Exception as e:
        logging.error(f"Sumy summarization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Sumy summarization failed: {str(e)}")

def huggingface_summarize_bart(text, max_length):
    try:
        print(max_length)
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(text, max_length=max_length, min_length=min(30, max_length), do_sample=False)
        return summary[0]['summary_text']
    
    except Exception as e:
        logging.error(f"Huggingface summarization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Huggingface summarization failed: {str(e)}")



def huggingface_summarize(text, max_length):
    try:

        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt", max_length=16384, truncation=True)

        # Generate summary
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min(30, max_length),
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        # Decode and return the summary
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        logging.error(f"Huggingface summarization error: {str(e)}")
        raise Exception(f"Huggingface summarization failed: {str(e)}")


def openai_summarize(text, max_tokens):
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
        logging.error(f"Openai summarization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Openai summarization failed: {str(e)}")

def summarize_text(text, length, method):
    sentence_count = determine_summary_length(text, length)
    max_tokens = sentence_count * 20  # Rough estimate for OpenAI and HuggingFace

    if method == "sumy":
        return sumy_summarize(text, sentence_count)
    elif method == "huggingface":
        return huggingface_summarize(text, max_tokens)
    elif method == "openai":
        return openai_summarize(text, max_tokens)
    else:
        raise ValueError("Invalid summarization method")
