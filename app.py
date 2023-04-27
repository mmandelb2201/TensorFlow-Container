import json
import boto3
import pandas as pd
import numpy as np

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
    
  data = request_dict['data']
  request_id = request_dict['requestId']
    
  total_request_numpy = []

  for i in range(len(data)):
      key_column = []
      value_coulmn = []
      for j in range(len(data[i]['req'])):
        key_column.append(data[i]['req'][j][0])
        value_coulmn.append(data[i]['req'][j][1])
      url_column = np.full((len(key_column), 1), data[i]['url'])
      timestamp_column = np.full((len(key_column), 1), data[i]['timeStamp'])
      request_numpy = np.column_stack((key_column, value_coulmn, url_column, timestamp_column))
      total_request_numpy.append(request_numpy)

  total = []
  for req in total_request_numpy:
      for row in req:
          total.append(row)

  df = pd.DataFrame(total, columns=['key', 'value', 'url', 'timestamp'])

  keys = df['key'].tolist()

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

  kv_dict = []
  for index, row in df.iterrows():
      kv_dict.append(
          {
              'key' : row['key'],
              'value' : row['value'],
              'prediction' : classes[index],
              'url' : row['url'],
              'timestamp' : row['timestamp']
          }
      )

  print(kv_dict[0])

  s3.put_object(Bucket=RESPONSE_BUCKET_NAME, Key='report-mc-response-{}.json'.format(request_id), Body=json.dumps(kv_dict))

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