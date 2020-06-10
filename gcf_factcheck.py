# def hello_world(request):
#     """Responds to any HTTP request.
#     Args:
#         request (flask.Request): HTTP request object.
#     Returns:
#         The response text or any set of values that can be turned into a
#         Response object using
#         `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
#     """
#     request_json = request.get_json()
#     if request.args and 'message' in request.args:
#         return request.args.get('message')
#     elif request_json and 'message' in request_json:
#         return request_json['message']
#     else:
#         return f'Hello World!'

  
import os
import telegram
# Import Libraries
import json
import random
import os
import re
import pandas as pd
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import re
from scipy.spatial import distance
import string
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import html

import nltk
nltk.download('stopwords', download_dir = "/tmp/nltk_data")
nltk.download('wordnet', download_dir = "/tmp/nltk_data")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

word_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
# nltk_eng_stop_words = stopwords.words('english')
custom_stop_words = ['of','the','a','to','is','in','on', 'that', 'an', 'by', 'for']

bot = telegram.Bot(token=os.environ["TELEGRAM_TOKEN"])
from google.cloud import storage

client = storage.Client()
bucket = client.get_bucket('infringement-100-iot')

# filename = 'GoogleNews-vectors-negative300.bin'
# filename = 'googlenews_500k.bin'
# filename = 'glove_6b_300d.bin'

model_dest_filename = ''
tokenizer_dest_filename = ''
model_loaded = ''
tokenizer_loaded = ''
MAX_SEQUENCE_LENGTH = 300
MAX_NUM_WORDS = 10000
EMBEDDING_DIM = 300

def copy_from_bucket_function(f_name):    
  blob = bucket.blob(f_name)
  dest_file = '/tmp/'+f_name
  blob.download_to_filename(dest_file)

  return dest_file

model_dest_filename = copy_from_bucket_function('fact_check_model_25_5_20.h5')
tokenizer_dest_filename = copy_from_bucket_function('fact_check_tokenizer_25_5_20.pickle')


def load_function():
  print('load_function')
  global tokenizer_loaded
  global model_loaded

  # loading
  with open(tokenizer_dest_filename, 'rb') as handle:
      tokenizer_loaded = pickle.load(handle)

  # load model
  model_loaded = load_model(model_dest_filename)

  # summarize model.
  print(model_loaded.summary())
  print('load_function complete')

load_function()

def update_model():
  print('update_model')
  global model_dest_filename
  global tokenizer_dest_filename

  model_dest_filename = copy_from_bucket_function('fact_check_model_25_5_20.h5')
  tokenizer_dest_filename = copy_from_bucket_function('fact_check_tokenizer_25_5_20.pickle')

  load_function()
  print('update_model complete')


def decontracted(phrase):
    phrase = html.unescape(phrase)
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"govt", " government", phrase)
    return phrase

# clean the data and get it transformed as needed
def get_cleaned_data(input_data, mode='df', stop_words = None, tokenize_op=True):
    print("get_cleaned_data")
    input_df = ''
    
    if mode != 'df':
        input_df = pd.DataFrame([input_data], columns=['text'])
    else:
        input_df = input_data
        
    #lowercase the text
    input_df['text'] = input_df['text'].str.lower().str.strip()

    input_df['text'] = input_df['text'].apply(lambda elem: decontracted(elem))
    
    #remove @mentions of twitter
    # input_df['text'] = input_df['text'].apply(lambda elem: re.sub(r"@[\w]*", "", elem))

    #replace newline with space
    input_df['text'] = input_df['text'].apply(lambda elem: elem.replace('\n', ' ').strip())

    #remove special characters
    input_df['text'] = input_df['text'].apply(lambda elem: re.sub(r"(#[A-Za-z0-9]+)|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", elem))
    
    #reduce multiple spaces to one space
    input_df['text'] = input_df['text'].replace('\s+', ' ', regex=True)

    # remove numbers
    # input_df['text'] = input_df['text'].apply(lambda elem: re.sub(r"\d+", "", elem))
    
    #remove stopwords
    if stop_words is not None:      
      input_df['text'] = input_df['text'].apply(lambda x: ' '.join([word.strip() for word in x.split() if word not in (stop_words)]))
    
    #stemming, changes the word to root form
    # input_df['text'] = input_df['text'].apply(lambda words: ' '.join([word_stemmer.stem(word) for word in words]))
    
    #lemmatization, same as stemmer, but language corpus is used to fetch the root form, so resulting words make sense
#     more description @ https://www.datacamp.com/community/tutorials/stemming-lemmatization-python
    input_df['text'] = input_df['text'].apply(lambda words: (wordnet_lemmatizer.lemmatize(words)))
    
    if tokenize_op == True:
      input_df['text'] = input_df['text'].apply(lambda row: nltk.word_tokenize(row))

    print("get_cleaned_data complete")

    return input_df



def loaded_model_test_function(tokenizer, model, stop_w=None):
  test_messages_dict = [
                        {'real_fact': False, 'text': 'Governments roadmap to ease covid-10 restriction will be on 3 week review process, the current phases would commence on the following dates: Phase 1 - 18th May Phase 2 - 8th June Phase 3 - 29th June Phase 4 - 20th July Phase 5 - 10th August If coronavirus cases begin to increase, we will revert to restrictions set out in the previous stage'}
                        # {'real_fact': False, 'text': 'The workers who worked between the 1990 and 2020, have the right to receive the benefit of 120000 from Ministry of Labour and Employment, Check if your name is in the list of people who have the rights to withdraw this benefits'},
                        # {'real_fact': False, 'text': 'Entire Mumbai military lockdown for 10 days from Saturday please stock everything Only milk and medicine will be available'}
                        ]
  test_messages_df = pd.DataFrame(test_messages_dict)
  test_messages_df = get_cleaned_data(test_messages_df, tokenize_op=False, stop_words=stop_w)

  sequences = tokenizer.texts_to_sequences(test_messages_df.text)
  data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

  final_test_pred = model.predict_classes(data)
  return final_test_pred, data


def get_match(tokenizer, model, input_msg, stop_w=None):
    print("get_match")
    user_message_dict = [{'text': input_msg}]
    user_message_df = pd.DataFrame(user_message_dict)
    user_message_df = get_cleaned_data(user_message_df, tokenize_op=False, stop_words=stop_w)

    print("texts_to_sequences")
    sequences = tokenizer.texts_to_sequences(user_message_df.text)
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    print("predict_classes")
    final_test_pred = model.predict_classes(data)
    
    print("get_match complete")

    return final_test_pred, data

        
def webhook(request):
    if request.method == "POST":
        update = telegram.Update.de_json(request.get_json(force=True), bot)
        chat_id = update.message.chat.id
        chat_message = update.message.text
        print('chat_id: {}, chat_message {}'.format(chat_id, chat_message))
        if (update.message.text == "/start"):
          bot.sendMessage(chat_id=chat_id, text='Hi, Welcome to the FactCheck! Please paste your forwarded message to check if it is from the Real-Fact corpus')
        elif (update.message.text == "update_global"):
          update_model()
        else:
          pred, data = get_match(tokenizer_loaded, model_loaded, chat_message, stop_w=None)
          if pred[0] == 0:
            resp = 'This claim seems a FAKE. '
          else:
            resp = 'This claim seems a FACT. '

          resp = resp + " However, please verify the same from https://twitter.com/PIBFactCheck for confirmation."

          bot.sendMessage(chat_id=chat_id, text=resp)

          # Reply with the same message, kept for reference
          # bot.sendMessage(chat_id=chat_id, text=update.message.text)

    return "ok"