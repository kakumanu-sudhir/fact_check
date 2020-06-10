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


'''
#Requirements.TXT:

# Function dependencies, for example:
# package>=version
Flask==1.1.1
python-telegram-bot==11.1.0
pandas
nltk
gensim
scipy
google-cloud-storage    

'''


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

# import nltk.corpus
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer 
# from nltk.tokenize import word_tokenize
# import nltk 
# nltk.download('punkt')
# from nltk.tokenize import sent_tokenize, word_tokenize
# nltk.download('wordnet')
# from nltk.stem import WordNetLemmatizer

bot = telegram.Bot(token=os.environ["TELEGRAM_TOKEN"])

from google.cloud import storage 

client = storage.Client()
bucket = client.get_bucket('infringement-100-iot')

# filename = 'GoogleNews-vectors-negative300.bin'
filename = 'googlenews_500k.bin'
# filename = 'glove_6b_300d.bin'

blob = bucket.blob(filename)
dest_file = '/tmp/'+filename
blob.download_to_filename(dest_file)

filename1 = filename + '.vectors.npy'
blob_npy = bucket.blob(filename1)
dest_file1 = '/tmp/'+filename1
blob_npy.download_to_filename(dest_file1)

embeddings_index = {}

# def update_model():
  # embeddings_index = get_model()

def get_model():
  print('get_model')
  
  # model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300', binary=True, limit=500000)
  # wv_from_bin = KeyedVectors.load_word2vec_format(dest_file, binary=True, limit=500000) 
  wv_from_bin = KeyedVectors.load(dest_file) 
  #extracting word vectors from google news vector
  # embeddings_index = {}
  print('get_model complete')
  for word, vector in zip(wv_from_bin.vocab, wv_from_bin.vectors):
      coefs = np.asarray(vector, dtype='float32')
      embeddings_index[word] = coefs
  
  print('get_model complete')
  return embeddings_index

# embeddings_index = get_model()

#Initializing dataset dictionary for provided intents
def load_data(intent_path):
    train_set = {}
    test_set = {}
    decoded_dataset = {}
    stop = stopwords.words('english')

    for fil in os.listdir(intent_path):
        if fil.endswith('.csv'):
          intent_df = pd.read_csv(intent_path + fil, encoding='utf-8-sig', names=['intent_header'])
          # print(intent_df.head(5))
          #lowercase the text
          intent_df['intent_header'] = intent_df['intent_header'].str.lower()

          #remove special characters, we cant remove because $ is to be referenced in the slots
          # intent_df = intent_df.apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
          # remove numbers
          intent_df['intent_header'] = intent_df['intent_header'].apply(lambda elem: re.sub(r"\d+", "", elem))

          #remove stopwords
          intent_df['intent_header'] = intent_df['intent_header'].apply(lambda x: ([word for word in x.split() if word not in (stop)]))

          # intent_df = intent_df.apply(lambda x: word_tokenize(x))
          # #stemming, take the word to root form, but this might leave some important characters, hence apply lemma
          # intent_df = intent_df.apply(lambda x: word_stemmer(x))
          # #lemmatization
          # intent_df = intent_df.apply(lambda x: word_lemmatizer(x))

          # print(intent_df.head(5))
          decoded_dataset[fil[:-4]] = intent_df[intent_df.columns[0]].to_list()

    return decoded_dataset

def trainModel(words, model, num_features):
    #feature vector is initialized as an empty array
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in embeddings_index.keys():
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

# Take the user input as test data and predict using the above classifier. Get the mode of the predicted values.
def intentPredict(user_input, attr):
  # YOUR CODE HERE for the prediction
  print(user_input)
  s1_afv = trainModel(user_input, model= embeddings_index, num_features=300 )
  scores = []
  for intent in decoded_data_intent:
    utterances = decoded_data_intent[intent]
    print(utterances)
    fscore = 0.0
    for uter in utterances:
      s2_afv = trainModel(uter,model= embeddings_index, num_features=300)
      cos = distance.cosine(s1_afv, s2_afv)
      # print(cos)
      if cos is not 'nan':
        fscore += cos
    scores+= [(intent,cos)]
  
  return scores

def test_stub():
  # if embeddings_index is not None:
  #     embeddings_index = get_model()
  embeddings_index = get_model()
  print('test_stub')
  
  user_inputs = ['Find me a chinese restaurant in bangalore', 'can you get a excellent restaurant', 
                'Italian restaurant', 'please find a cheap restaurant ',
                 'i am looking for restaurants', 'good eaters nearby',
                 'when do you close', 'are you open right now', 'place an order']
  
  for user_inp in user_inputs:
    s1_afv = trainModel(user_inp, model= embeddings_index, num_features=300 )
    utterances = [['close'], ['open'], ['open', 'right'], ['time', 'open'], ['would', 'like', 'order'], ['place', 'order'], ['pickup', 'order'], ['would']]
    # utterances = [['$meal_type'], ['find', '$cost', '$rating', '$cuisine', '$shop_type', '$city'], ['$urge', '$cost', '$rating', '$shop_type', '$city', 'eat', '$cuisine'], ['$shop_type', 'near', '$city'], ['nearby', '$shop_type'], ['$urge', '$meal_type', '$cuisine', '$shop_type'], ['$urge', '$shop_type', '$cost', '$rating', '$cuisine'], ['$shop_type', 'rated', '$rating'], ['$urge', 'go', '$shop_type', '$city'], ['$urge', '$meal_type'], ['$urge', '$shop_type', '$meal_type'], ['$shop_type', '$meal_type'], ['$urge', '$meal_type'], ['$urge', '$shop_type'], ['$urge'], ['$cuisine'], ['$cost'], ['$city'], ['$shop_type'], ['$meal_type'], ['$rating'], ['want', '$meal_type', 'near'], ['looking', '$meal_type', '$shop_type', '$city'], ['looking', '$shop_type', '$city'], ['$meal_type', '$shop_type', '$city'], ['$meal_type', '$shop_type', 'near'], ['$meal_type', 'near'], ['$rating', '$cost', '$city'], ['$rating', '$cost'], ['$cost', '$rating', '$shop_type'], ['$cost', '$rating'], ['$rating', '$shop_type', 'near'], ['would', 'like', 'rating', '$rating'], ['get', '$shop_type', 'near'], ['$shop_type', 'near', '$city'], ['$shop_type', 'nearby'], ['$shop_type', 'near'], ['find', '$shop_type', 'near']]
    scores = []
    fscore = 0.0
    for uter in utterances:
      s2_afv = trainModel(uter,model= embeddings_index, num_features=300)
      cos = distance.cosine(s1_afv, s2_afv)
      # print(cos)
      if cos is not 'nan':
        fscore += cos
    scores+= [(user_inp,cos)]

    print(scores)

  return scores
        
def webhook(request):
    if request.method == "POST":
        update = telegram.Update.de_json(request.get_json(force=True), bot)
        chat_id = update.message.chat.id
        print('update.message.text {}'.format(update.message.text))
        if (update.message.text == "/start"):
          bot.sendMessage(chat_id=chat_id, text='Hi, Welcome to the FactCheck! Please paste your forwarded message to check if it is from the Real-Fact corpus')
        else:
          resp = test_stub()
          bot.sendMessage(chat_id=chat_id, text=resp)

          # Reply with the same message, kept for reference
          # bot.sendMessage(chat_id=chat_id, text=update.message.text)
    return "ok"
