# imports
import os, json, time, requests, pika, sys
from bson.objectid import ObjectId
from utils import connectToMongo, connectToGCS
from utils import get_signed_url, save_df_to_gcs
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pickler-ml'))
sys.path.append(ROOT_DIR)

# DO NOT MOVE THIS ABOVE sys.path.append(ROOT_DIR)
from clean import clean_data

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


def callback_two(ch, method, properties, body):
    # get message data
    body = json.loads(body)
    form_data = body['form']
    email = form_data['email']
    file_id = form_data['id']

    remove_features = form_data['remove']
    numerical = form_data['numerical']
    categorical = form_data['categorical']
    encoding = form_data['encoding']
    target = form_data['target']

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

    # clean data using the signed URL
    df = clean_data(signed_url, remove_features, numerical, categorical, encoding, target)

    # Create a new object name for the cleaned data
    cleaned_object_name = f"cleaned_{object_name}"

    # Save cleaned data as CSV file to GCS
    if save_df_to_gcs(bucket, df, cleaned_object_name):
        # Update MongoDB record with status and cleaned file name
        files.update_one(
            {'owner': email, '_id': ObjectId(file_id)},
            {'$set': {
                'status': 'cleaned',
                'cleaned_file_name': cleaned_object_name
            }}
        )
        print(f"Updated MongoDB record with cleaned file name: {cleaned_object_name}")

    ch.basic_ack(delivery_tag=method.delivery_tag)


channel_one.basic_consume(queue='extract-headers', on_message_callback=callback_one, auto_ack=False)
channel_one.start_consuming()
channel_one.basic_consume(queue='clean-data', on_message_callback=callback_two, auto_ack=False)
channel_one.start_consuming()