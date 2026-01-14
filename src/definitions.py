import json
import os
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv
import time

# Use this if you would like to generate alternate definitions for the taxonomy.

load_dotenv('.en')


def configure_gemini():
    api_key = os.getenv('API_KEY')
    if not api_key:
        raise ValueError(
            "Google API key not found. Please set it in a .env file.")
    genai.configure(api_key=api_key)


def process_pair(pair):
    text = pair.strip().split("\t")

    return (text[-2], text[-1]) if len(text) >= 3 else (text[0], text[1])


def load_file(filepath):
    try:
        with open(filepath, 'r') as f:
            return f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")


def generate_definitions(words_to_define):
    model = genai.GenerativeModel('gemini-2.5-pro')

    definitions_dict = {}
    print(f"Generating definitions for {len(words_to_define)} words...")

    for word in tqdm(words_to_define, desc="Generating definitions"):
        try:
            prompt = (
                f"Provide a concise, one sentence to three sentence definition for the word: '{word}'. "
                "Use modern and partial historical context since this is mainly for taxonomy expansion of the words given new word so consider the task as context. Do not include the word itself in the response. Just the definition."
            )

            response = model.generate_content(prompt)

            definition = response.text.strip()

            definitions_dict[word] = [definition]

            print(definition)
            print("Waiting for 3 seconds to avoid rate limit..")
            save_to_json(definitions_dict, filename='../data/food/dic.json')
            time.sleep(3)

        except Exception as e:
            print(f"\nCould not get definition for '{word}'. Error: {e}")
            definitions_dict[word] = ["Error: Could not retrieve definition."]

    return definitions_dict


def create_words(dataset):
    taxonomy_file = os.path.join(f"../data/{dataset}/{dataset}_raw_en.taxo")
    taxonomy = load_file(taxonomy_file)

    concept_set = set([])

    for pair in taxonomy:
        child, parent = process_pair(pair)
        concept_set.add(child)
        concept_set.add(parent)

    concepts = sorted(concept_set)

    return concepts


def save_to_txt(word_list, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for word in word_list:
            f.write(word + '\n')
    print(f"\nSuccessfully saved {len(word_list)} words to '{filename}'")


def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"\nSuccessfully saved definitions to {filename}")
