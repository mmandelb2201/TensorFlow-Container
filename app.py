import json
import boto3
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
from keras.models import load_model

import os

RESPONSE_BUCKET_NAME = os.environ['RESPONSE_BUCKET']

s3 = boto3.client('s3')

def lambda_handler(event, context):
  deserialized_event = json.loads(json.dumps(event))

  response = s3.get_object(Bucket=deserialized_event['Records'][0]['s3']['bucket']['name'], Key=deserialized_event['Records'][0]['s3']['object']['key'])

  request_dict = json.loads(response['Body'].read().decode())
    
  predictions = request_dict['predictions']
  request_id = request_dict['requestId']
    
  kv = [pd.DataFrame(predictions[i]['req'], columns=['key', 'value']) for i in range(len(predictions)) if predictions[i]['prediction'] == 1.0]

  keys = []
  for r in kv:
      k = r['key'].tolist()
      for key in k:
          keys.append(key)


  amt = len(keys)

  print(f"There are {amt} keys")

  keys = map(split_word, keys)
  keys = list(keys)

  #load and embed using USE
  module = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

  embedded_keys = module(keys)

  #load multiclassifier
  model = load_model('./privacy_text_classifier.h5')

  predictions = model.predict(embedded_keys)

  print(f"there are {len(predictions)} predictions")

  classes = []
  for p in predictions:
     classes.append(get_class(p))

  print(classes)

  s3.put_object(Bucket=RESPONSE_BUCKET_NAME, Key='report-mc-response-{}.json'.format(request_id), Body=json.dumps(predictions))

def get_class(prediction):
  i = 0
  c = 0
  m = 0
  for pred in prediction:
    if pred > m:
        m = pred
        c = i
    i += 1
  return c

def split_word(word):
  s = word.replace(".", " ").replace("-", " ")
  for i in range(len(s) - 1):
    if s[i].islower() and s[i + 1].isupper():
      s = s[0:i+1] + " " + s[i + 1:]
  
  return s.lower()