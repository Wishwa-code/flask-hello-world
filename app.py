from flask import Flask
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

app = Flask(__name__)


CORS(app)

# Retrieve MongoDB URL from environment variables
mongo_url = "mongodb+srv://vishva2017087:ckGzmJoKMoXkeMuQ@cluster0.i62acyf.mongodb.net/test"
if not mongo_url:
    raise EnvironmentError("MONGO_URL not found in environment variables.")

try:
    client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
    client.server_info()  # Attempt to connect and force a server call
    db = client.get_default_database()  # Get the default database
    app.logger.info("Successfully connected to MongoDB")
except ServerSelectionTimeoutError as e:
    app.logger.error("Database connection failed.", exc_info=True)
    raise e

@app.route('/')
def hello_world():
    return 'Hello, World!'
