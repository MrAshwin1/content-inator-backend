from flask import Flask, request, jsonify
import re
import openai
import os
from flask_cors import CORS
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)  # add this line to enable CORS

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv('API_KEY')

# Define a function to paraphrase a given sentence
def paraphrase(sentence):
    # Remove any extra spaces and newlines from the sentence
    sentence = re.sub('\s+', ' ', sentence).strip()
    # Set the max_tokens parameter to be equal to or 10 tokens more than the length of the input sentence
    max_tokens = len(sentence.split()) + 10
    # Use OpenAI's GPT-3 to generate a new sentence that conveys the same meaning
    response = openai.Completion.create(
      engine="davinci",
      max_tokens=max_tokens,
      n=1,
      stop=None,
      temperature=0.7,
      top_p=1,
      prompt=f"Paraphrase the following sentence: {sentence}\nNew sentence:"
    )
    # Get the paraphrased sentence from the API response
    new_sentence = response.choices[0].text.strip()
    return new_sentence

# Define a function to improve the fluency of a given sentence
def improve_fluency(sentence):
    # Remove any extra spaces and newlines from the sentence
    sentence = re.sub('\s+', ' ', sentence).strip()
    # Set the max_tokens parameter to be equal to or 10 tokens more than the length of the input sentence
    max_tokens = len(sentence.split()) + 10
    # Use OpenAI's GPT-3 to generate a new sentence that conveys the same meaning, with improved fluency
    response = openai.Completion.create(
      engine="davinci",
      max_tokens=max_tokens,
      n=1,
      stop=None,
      temperature=0.7,
      top_p=1,
      prompt=f"Improve the fluency of the following sentence: {sentence}\nImproved sentence:"
    )
    # Get the improved sentence from the API response
    new_sentence = response.choices[0].text.strip()
    return new_sentence

@app.route('/paraphrase', methods=['POST'])
def paraphrase_endpoint():
    # Get the input phrase from the request body
    input_phrase = request.json.get('input_phrase')

    # Call the paraphrasing function with the input phrase
    rephrase_output = paraphrase(input_phrase)

    # Return the paraphrased sentence as a JSON response
    return jsonify({
        'rephrase_output': rephrase_output,
    })

@app.route('/improve-fluency', methods=['POST'])
def improve_fluency_endpoint():
    # Get the input phrase from the request body
    input_phrase = request.json.get('input_phrase')

    # Call the fluency-improving function with the input phrase
    augment_output = improve_fluency(input_phrase)

    # Return the fluency-improved sentence as a JSON response
    return jsonify({
        'augment_output': augment_output,
    })

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
