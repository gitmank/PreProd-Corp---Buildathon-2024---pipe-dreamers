# imports
import os, json, time, requests
from flask import Flask, request
from flask_cors import CORS
from bson.objectid import ObjectId
from utils import connectToMongo, verify_jwt
import pandas as pd

# create a Flask app
app = Flask(__name__)
CORS(app, origins='*')

# connect to MongoDB
db = connectToMongo()
files = db['files']

# auth middleware
def auth_middleware(func):
    """
    Middleware to check if the request is authenticated
    decodes JWT token and sets the email in the request object
    """
    def wrapper(*args, **kwargs):
        token = request.headers.get('Authorization', '').split(' ').pop()
        result = verify_jwt(token)
        if result is False:
            return 'unauthenticated', 401
        else:
            email = result['email']
            request.email = email
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper

# ping route
@app.route('/')
@auth_middleware
def ping():
    """
    simple test to check if user is authenticated
    returns the email of the authenticated user
    """
    return request.email, 200

# get file data
@app.route('/files/stats/<id>', methods=['GET'])
# @auth_middleware
def get_file_data(id):
    """
    GET file columns
    returns JSON body with file header as array
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
        print('Error getting columns - ', e)
        return 'internal server error', 500
    

# run app
if __name__ == '__main__':
    app.run(port=os.getenv('FLASK_PORT', 6060), debug=True)