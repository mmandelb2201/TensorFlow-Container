import json
import boto3
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from keras.models import load_model

s3 = boto3.resource('s3')

def lambda_handler(event, context):
  deserialized_event = json.loads(json.dumps(event))

  response = s3.get_object(Bucket=deserialized_event['Records'][0]['s3']['bucket']['name'], Key=deserialized_event['Records'][0]['s3']['object']['key'])

  request_dict = json.loads(response['Body'].read().decode())
    
  reqs = request_dict['data']
    
  keys = [pd.DataFrame(req, columns=['key']) for req in reqs]

  keys = map(split_word, keys)

  #load and embed using USE
  module = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

  embedded_keys = module(keys)

  #load multiclassifier
  model = load_model('./model/privacy_text_classifier.h5')

  predictions = []
  for key in embedded_keys:
    predictions.append(model.predict(key))

  




def split_word(word):
  s = word.replace(".", " ").replace("-", " ")
  for i in range(len(s) - 1):
    if s[i].islower() and s[i + 1].isupper():
      s = s[0:i+1] + " " + s[i + 1:]
  
  return s.lower()