# imports
import os, json, time, requests, pika
from bson.objectid import ObjectId
from utils import connectToMongo, connectToGCS
import pandas as pd

# connect to MongoDB
db = connectToMongo()
files = db['files']

# # connect to GCS
bucket = connectToGCS()

# set up RabbitMQ
print(os.getenv('RABBITMQ_URI', 'localhost'))
connection = pika.BlockingConnection(pika.ConnectionParameters(host='rabbitmq'))
channel = connection.channel()
channel.queue_declare(queue='extract-headers', durable=True, exclusive=False)

# get file data
def get_headers(id):
    """
    read file and update rows x columns in DB
    """
    try:
        url = "https://dashboard.devscene.co/mock.csv"
        # get array of column names
        df = pd.read_csv(url)
        columns = df.columns.tolist()
        return json.dumps({
            'features': columns,
            'rows': df.shape[0],
            'columns': df.shape[1]
        }), 200
    except Exception as e:
        print('Error getting file header - ', e)


def callback(ch, method, properties, body):
    # get message data
    body = json.loads(body)
    email = body['email']
    file_id = body['id']

    # find record in mongodb with email and file_id
    record = files.find_one({ 'email': email, '_id': file_id})
    print(record.get('name'))
    object_name = record.get('objectName')

    # # get signed url from GCS
    # url = bucket.

    ch.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_consume(queue='extract-headers', on_message_callback=callback, auto_ack=False)
channel.start_consuming()