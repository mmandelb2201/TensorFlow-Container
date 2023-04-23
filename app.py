import json
import boto3
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from keras.models import load_model

RESPONSE_BUCKET_NAME = os.environ['RESPONSE_BUCKET']

s3 = boto3.resource('s3')

def lambda_handler(event, context):
  deserialized_event = json.loads(json.dumps(event))

  response = s3.get_object(Bucket=deserialized_event['Records'][0]['s3']['bucket']['name'], Key=deserialized_event['Records'][0]['s3']['object']['key'])

  request_dict = json.loads(response['Body'].read().decode())
    
  predictions = request_dict['predictions']
  request_id = request_dict['requestId']
    
  keys = [pd.DataFrame(predictions[i]['req'], columns=['key']) for i in range(len(predictions)) if predictions[i]['prediction'] == 1.0]

  keys = map(split_word, keys)

  #load and embed using USE
  module = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

  embedded_keys = module(keys)

  #load multiclassifier
  model = load_model('./model/privacy_text_classifier.h5')

  predictions = []
  for key in embedded_keys:
    predictions.append(model.predict(key))

  s3.put_object(Bucket=RESPONSE_BUCKET_NAME, Key='report-mc-response-{}.json'.format(request_id), Body=json.dumps(predictions))




def split_word(word):
  s = word.replace(".", " ").replace("-", " ")
  for i in range(len(s) - 1):
    if s[i].islower() and s[i + 1].isupper():
      s = s[0:i+1] + " " + s[i + 1:]
  
  return s.lower()