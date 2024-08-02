import os
from pymongo import MongoClient
from google.cloud import storage

def connectToMongo():
    try:
        client = MongoClient(os.getenv('MONGO_URI', 'mongodb://localhost:27017/alerts-db'))
        client.admin.command('ping')
        print("Successfully connected to MongoDB!")
        return client['pickler-db']
    except Exception as e:
        print('Error connecting to MongoDB - ', e)
        return None

def verify_jwt(token=None):
    try:
        JWT_SECRET = os.getenv('JWT_SECRET', '')
        if(JWT_SECRET == ''):
            print('JWT_SECRET not set in environment')
            return False
        if token is None:
            return False
        result = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        return result
    except Exception as e:
        print('Error checking API key - ', e)
        return False
    
def connectToGCS():
    try:
        client = storage.Client()
        bucket_name = os.getenv('GCS_BUCKET', 'pickler-bucket-test')
        bucket = client.bucket(bucket_name)
        print("Successfully connected to GCS!")
        return bucket
    except Exception as e:
        print('Error connecting to GCS - ', e)
        return None

# def connectToRedis():
#     try:
#         client = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=os.getenv('REDIS_PORT', 6379), decode_responses=True)
#         client.ping()
#         print("Successfully connected to Redis!")
#         return client
#     except Exception as e:
#         print('Error connecting to Redis - ', e)
#         return None