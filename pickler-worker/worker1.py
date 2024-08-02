# imports
import os, json, time, requests, pika, sys
from bson.objectid import ObjectId
from utils import connectToMongo, connectToGCS
from utils import get_signed_url, save_df_to_gcs
import pandas as pd

# connect to MongoDB
db = connectToMongo()
files = db['files']

# # connect to GCS
bucket = connectToGCS()

# set up RabbitMQ
RABBIT_URI = os.getenv('RABBITMQ_URI', 'localhost')
connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBIT_URI))
channel_one = connection.channel()
channel_one.queue_declare(queue='extract-headers', durable=True, exclusive=False)
channel_two = connection.channel()
channel_two.queue_declare(queue='clean-data', durable=True, exclusive=False)

# get file data
def get_dimensions(url):
    """
    read file and update rows x columns in DB
    """
    try:
        # get array of column names
        df = pd.read_csv(url)
        columns = df.columns.tolist()
        return {
            'features': columns,
            'rows': df.shape[0],
            'columns': df.shape[1]
        }
    except Exception as e:
        print('Error getting file header - ', e)


def callback_one(ch, method, properties, body):
    # get message data
    body = json.loads(body)
    email = body['email']
    file_id = body['id']

    # find record in mongodb with email and file_id
    record = files.find_one({'owner': email, '_id': ObjectId(file_id)})
    if record is None:
        print(f"Record with email {email} and id {file_id} not found")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    object_name = record.get('objectName')
    if not object_name:
        print("Object name not found in the record")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    # get signed URL from GCS
    signed_url = get_signed_url(bucket, object_name)
    if not signed_url:
        print("Failed to generate signed URL")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return
    
    # get file dimensions using the signed URL
    dimensions = get_dimensions(signed_url)
    if dimensions is None:
        print("Failed to get file dimensions")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    # update MongoDB record with file dimensions
    files.update_one(
        {'owner': email, '_id': ObjectId(file_id)},
        {'$set': {
            'features': dimensions['features'],
            'rows': dimensions['rows'],
            'columns': dimensions['columns'],
            'status': 'uploaded',
        }}
    )

    print(f"Updated MongoDB record with dimensions: {dimensions}")

    ch.basic_ack(delivery_tag=method.delivery_tag)


channel_one.basic_consume(queue='extract-headers', on_message_callback=callback_one, auto_ack=False)
channel_one.start_consuming()