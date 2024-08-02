# imports
import os, json, time, requests
from bson.objectid import ObjectId
from utils import connectToMongo
import pandas as pd

# connect to MongoDB
db = connectToMongo()
files = db['files']

# get file data
def update_headers(id):
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
        