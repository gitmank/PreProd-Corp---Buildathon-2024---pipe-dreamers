# imports
import os, json, time, requests, pika, sys
from bson.objectid import ObjectId
from utils import connectToMongo, connectToGCS
from utils import get_signed_url, save_df_to_gcs
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pickler-ml'))
sys.path.append(ROOT_DIR)

# DO NOT MOVE THIS ABOVE sys.path.append(ROOT_DIR)
from train import train_data

# connect to MongoDB
db = connectToMongo()
files = db['files']
results = db['models']

# connect to GCS
bucket = connectToGCS()

# set up RabbitMQ
RABBIT_URI = os.getenv('RABBITMQ_URI', 'localhost')
connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBIT_URI))
channel_three = connection.channel()
channel_three.queue_declare(queue='train-data', durable=True, exclusive=False)

def callback_three(ch, method, properties, body):
    # get message data
    body = json.loads(body)
    form_data = body['form']
    email = form_data['email']
    file_id = form_data['id']
    target = form_data['target']
    models = form_data['models']
    config = form_data['config']

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

    # Read the data directly from the signed URL
    df = pd.read_csv(signed_url)

    # Train models
    training_results = train_data(df, '/tmp', target, models, config, False)

    # Prepare the results for MongoDB
    results_list = []
    for model_name, metrics in training_results.items():
        gcs_object_name = f"{file_id}_{model_name}.pkl"

        # Save model to GCS
        model_path = os.path.join(ROOT_DIR, gcs_object_name)
        save_df_to_gcs(model_path, bucket, gcs_object_name)

        # Add to results list
        results_list.append({
            'model_name': model_name,
            'metrics': metrics,
            'object_name': gcs_object_name
        })

    # Save all model results to MongoDB in a single entry
    result_entry = {
        'owner': email,
        'file_id': file_id,
        'results': results_list
    }
    results.insert_one(result_entry)

    ch.basic_ack(delivery_tag=method.delivery_tag)

channel_three.basic_consume(queue='train-data', on_message_callback=callback_three, auto_ack=False)
channel_three.start_consuming()
