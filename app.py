from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
import hashlib
from flask_jwt_extended import JWTManager, create_access_token
import os
from dotenv import load_dotenv
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from bson import json_util
import pandas as pd
import warnings
from math import sqrt
import pandas as pd
from sklearn.metrics import mean_squared_error
import datetime
from bson.objectid import ObjectId  
import json
import http.client
import urllib.parse


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


def evaluate_arima_model(X, arima_order):
        # prepare training dataset
        train_size = int(len(X) * 0.66)
        train, test = X[0:train_size], X[train_size:]
        history = [x for x in train]

        # make predictions
        predictions = list()
        for t in range(len(test)):
            model = ARIMA(history, order=arima_order)  # Indentation fixed here
            model_fit = model.fit()
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            history.append(test[t])

        # calculate out of sample error
        #rmse = sqrt(mean_squared_error(test, predictions))
        mse = mean_squared_error(test, predictions)
        return mse

# ... previous code ...
    
def evaluate_models(dataset, p_values, d_values, q_values):

    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    # ... other parts of the function ...

    for p in p_values:
        for d in d_values:  # Indentation fixed
            for q in q_values:
                # Code to be executed inside the loops
                order = (p, d, q)
                print(p,d,q)
                try:
                    #rmse = evaluate_arima_model(dataset, order)
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                        print('ARIMA%s MSE=%.3f' % (order,mse))
                    # ... rest of your code ...
                except:
                    continue
    #print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
    return order

def get_data(country, product_id):
    # Build the query filter
    filter = {"country": country, "productCode": product_id}

    # Find matching documents and create the data array
    data = []
    collection = db["imports"]
    cursor = collection.find(filter)

    for document in cursor:  # Iterate over documents
        for record in document['imports']:  # Iterate over the "imports" array
            data.append({
                'date': datetime.date(record['year'], record['month'], 1),
                'sales': record['sales']
            })

        if "order" in document:
            is_empty = document["order"]["tuple_is_empty"]
            if not is_empty:
                order = tuple(document["order"]["my_tuple"])
            else:
                # Handle the case where the order tuple is empty if needed
                pass

    if document and "order" in document:
        is_empty = document["order"]["tuple_is_empty"]  # Output: (2, 0, 1)
        if is_empty:
            my_tuple = document["order"]["my_tuple"]
            print(my_tuple)
            order = tuple(my_tuple)
            print(order)
        else:
            series = pd.DataFrame(data)

            # Convert 'date' column to DatetimeIndex
            series['date'] = pd.to_datetime(series['date'])

            # Set the 'date' column as the index
            series.set_index('date', inplace=True)

            series.index = series.index.to_period('M')
            p_values = [0, 1, 2, 4, 6, 8, 10]
            d_values = range(0, 3)
            q_values = range(0, 3)

            warnings.filterwarnings("ignore")
            order = evaluate_models(series.values, p_values, d_values, q_values)
            #order = (5,0,2)
            new_tuple = order
            if order is None:
                order = (0,0,0)
                collection.update_one(filter,
                    {"$set": {"order.my_tuple": list(new_tuple),
                    "order.tuple_is_empty": True
                    }})
            else:
                new_tuple = order
                collection.update_one(filter,
                    {"$set": {"order.my_tuple": list(new_tuple),
                    "order.tuple_is_empty": True
                    }})
    else:
        print("Document not found or 'order' field missing")


    return (data, order)

def get_historical_data(country, product_id):
    filter = {"country": country, "productCode": product_id}
    projection = {"imports": 1, "_id": 1}  
    collection = db["imports"]
    cursor = collection.find(filter, projection)  

    historical_data = []
    for document in cursor:
        for record in document.get("imports", []):  
            historical_data.append({
                '_id': ObjectId(),  # Generate a new ObjectId for each entry
                'country': country,
                'product_id': product_id,
                'year': record.get('year', None),
                'month': datetime.date(record.get('year', 1), record.get('month', 1), 1).strftime('%B'),
                'sales': record['sales']
            })
    return historical_data

def predict(data,order):

    # Create the DataFrame
    series = pd.DataFrame(data)

    # Convert 'date' column to DatetimeIndex
    series['date'] = pd.to_datetime(series['date'])

    # Set the 'date' column as the index
    series.set_index('date', inplace=True)

    series.index = series.index.to_period('M')

    # Fit the model
    model = ARIMA(series, order=(8, 0, 1))
    model_fit = model.fit()

    # Generate predictions for the next 5 occurrences
    forecast = model_fit.forecast(steps=5)
    return forecast

def get_countries_with_product(product_name):
    # Build the filter to find documents with the given product name
    filter = {"productCode": product_name}

    # Only retrieve necessary fields for efficiency
    projection = {"country": 1, "_id": 0}

    # Find matching documents and extract unique country names
    collection = db["imports"]
    cursor = collection.find(filter, projection)
    countries = set(doc["country"] for doc in cursor)

    return countries

def calculate_growth_rate(data):
    """Calculates the growth rate of an array of numerical values."""
    if len(data) < 2:
        return None  # Cannot calculate growth rate with less than two data points

    initial_value = data[0]
    final_value = data[-1]  
    growth_rate = ((final_value - initial_value) / initial_value) * 100
    return growth_rate

def analyze_countries_product(countries, product_id, product_name="desicated_coconut"):
    results = []  # List to store results for each country

    for country in countries:
        data, order = get_data(country, product_id)

        if data:
            if order is None:
                #print(f"No suitable ARIMA order found for {country}. Using default (8, 0, 1).")
                order = (8, 0, 1)

            try:
                forecast = predict(data, order)
                forecast_values = forecast.values
                growth_rate = calculate_growth_rate(forecast_values)
            except Exception as e:  # Catch any exceptions during prediction
                print(f"Error generating forecast for {country}: {e}")
                forecast_values = []  # Empty list for forecast
                growth_rate = None 

            # Append results to the list
            results.append({
                'country': country,
                'forecast': forecast_values.tolist(),  # Convert NumPy array to list
                'growth_rate': f"{growth_rate:.2f}%" if growth_rate is not None else "N/A"
            })

        else:
            print(f"No data found for {country}")

    return results

def combine_data(historical_data, predicted_sales):
    # Combine historical data with predictions
    combined_data = []
    for data in historical_data:
        # Convert ObjectId to string
        data['_id'] = str(data['_id'])
        combined_data.append(data)

    # Add predictions to the combined data
    for idx, data in enumerate(combined_data):
        data['prediction'] = predicted_sales[idx] if idx < len(predicted_sales) else None

    return combined_data

def get_bookmarks_list(user_ID):
    bookmarks_collection=db["newsbookmarks"]

    # Query for bookmarks with matching userId (no conversion to ObjectId)
    user_bookmarks = list(bookmarks_collection.find(
        {"userId": user_ID},  
        {"_id": 0}            
    ))

    return user_bookmarks  


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')  # No need to encode password
    if not email or not password:
        return jsonify({"error": "Missing email or password"}), 400

    user_collection = db['users']
    user = user_collection.find_one({"email": email, "password": password})  # Check if user exists with provided email and password
    if user:
        access_token = create_access_token(identity=str(user['_id']))
        return jsonify(access_token=access_token, name=user['name'], identity=str(user['_id'])), 200
    else:
        return jsonify({"error": "Invalid email or password"}), 401

@app.route('/fetch_chart_data', methods=['GET'])
def fetch_chart_data():
    country = request.args.get('country')
    product_id = request.args.get('product_id')
    
    data, order = get_data(country, product_id)
    historical_data = get_historical_data(country, product_id)
    
    predicted_sales = predict(data,order)
    #print(predicted_sales)   
    #print(type(predicted_sales))
            
    # Combine historical data with predictions
    combined_data = combine_data(historical_data, predicted_sales)
    
    # Convert combined_data to JSON using json_util
    json_data = json_util.dumps(combined_data)
    #print("json_data",json_data)
    
    return json_data, 200, {'Content-Type': 'application/json'}

@app.route('/fetch_analyzed_data', methods=['GET'])
def fetch_analyzed_data():
    product_id = request.args.get('product_id')
    
    countries = get_countries_with_product(product_id)
    results = analyze_countries_product(countries, product_id)
    
    # Convert combined_data to JSON using json_util
    results = json_util.dumps(results)
    print("results",results)
    
    return results, 200, {'Content-Type': 'application/json'}

@app.route('/fetch_transaction_data', methods=['GET'])
def fetch_transaction_data():
    collection=db["escrow"]
    
    # Fetch all documents from the collection
    all_transactions = list(collection.find())

    # Convert MongoDB documents (including ObjectId) to JSON
    transactions_json = json.loads(json_util.dumps(all_transactions))
    
    return jsonify(transactions_json)
    
@app.route('/update_transaction_status', methods=['PUT'])
def update_transaction_status():
    try:
        transaction_data = request.get_json()
        userId = transaction_data['userId']
        new_status = transaction_data['newStatus']

        collection = db["escrow"]
        result = collection.update_one(
            {"userId": userId},
            {"$set": {"status": new_status}}
        )

        if result.modified_count > 0:
            return jsonify({"message": "Transaction status updated successfully"})
        else:
            return jsonify({"message": "Transaction not found"}), 404

    except Exception as e:
        return jsonify({"message": f"Internal server error: {str(e)}"}), 500
    
@app.route('/create_transaction', methods=['POST'])
def create_transaction():
    try:
        transaction_data = request.get_json()

        # Validate the data (ensure all required fields are present, etc.)
        # ...

        # Insert the transaction into the database
        collection = db["escrow"]
        result = collection.insert_one(transaction_data)
        transaction_data['_id'] = str(result.inserted_id)

        # Return the created transaction (including the generated _id)
        return jsonify({"_id": str(result.inserted_id), **transaction_data}), 201  # 201 Created
    except Exception as e:
        return jsonify({"message": f"Internal server error: {str(e)}"}), 500
    
    
@app.route('/api/bookmark', methods=['POST'])
def bookmark_article():
    data = request.get_json()
    user_id = data.get('userId')
    article = data.get('article')  # Get the entire article document
    print(user_id)
    
    bookmarks_collection=db["newsbookmarks"]

    if user_id is None or article is None:
        return jsonify({"error": "Missing user ID or article data"}), 400

    article_url = article.get('url')  # Use 'url' as the unique identifier

    existing_bookmark = bookmarks_collection.find_one({"userId": user_id, "article.url": article_url})

    if existing_bookmark:
        # Remove bookmark if it exists
        bookmarks_collection.delete_one(existing_bookmark)
        message = "Bookmark removed"
    else:
        # Store the entire article document
        article['isBookmarked'] = True
        bookmarks_collection.insert_one({"userId": user_id, "article": article})
        message = "Bookmark added"

    user_bookmarks = list(bookmarks_collection.find({"userId": user_id}, {"_id": 0}))
    return jsonify({"message": message, "bookmarks": user_bookmarks})

@app.route('/api/news', methods=['GET'])
def get_news():
    
    all_articles = []
    seen_urls = set()
    
    categories = request.args.get('categories', )  
    sort = request.args.get('sort', )
    user_id = request.args.get('userId')
    print(user_id)

    
    # Assuming the Mediastack API limit is 100
    limit = 100
    
    page = 0
    
    conn = http.client.HTTPConnection('api.mediastack.com')
    params = urllib.parse.urlencode({
            'access_key': '7593bf851849fbeef498e22bbab3f33b',
            'languages': 'en',
            'countries': categories,
            'categories': 'business',
            'keywords': sort,
            'limit': limit,
            'offset': page * limit
        })
    conn.request('GET', '/v1/news?{}'.format(params))
    res = conn.getresponse()
    data = res.read()
    page += 1
    
    user_bookmarks = get_bookmarks_list(user_id)
        
    try:
            response_data = json.loads(data.decode('utf-8'))
            

            # Check if 'data' key exists and is a list
            if 'data' not in response_data or not isinstance(response_data['data'], list):
                return jsonify({"error": "Unexpected API response format"}), 500
            
            # Check if a user is logged in 
            user_id = request.args.get('userId')

            user_bookmarks = get_bookmarks_list(user_id) if user_id else []
            bookmarked_urls = set()

            # Iterate over bookmarks and handle potential missing 'url' fields
            for bookmark in user_bookmarks:
                if 'article' in bookmark and 'url' in bookmark['article']:
                    bookmarked_urls.add(bookmark['article']['url'])
            # Filter for articles with images and add isBookmarked field
            # Filter and de-duplicate articles within the page
            for article in response_data['data']:
                if article.get('image') is not None and article['url'] not in seen_urls and article['url'] not in bookmarked_urls:
                    all_articles.append({**article, 'isBookmarked': False})
                    seen_urls.add(article['url'])
                    if len(all_articles) >= 20:
                        break  # Stop fetching if we have 
            

    except json.JSONDecodeError:
            return jsonify({"error": "Invalid API response"}), 500

    # Trim if we have more than 20 unique articles
    response_data['data'] = all_articles[:20]
    
    return jsonify(response_data)        
        
    
    
@app.route('/api/get_bookmarks', methods=['GET'])
def get_bookmarks():
    user_id = request.args.get('userId')  # Get userId as a string

    if not user_id:
        return jsonify({"error": "Missing userId parameter"}), 400

    # Query for bookmarks with matching userId (no conversion to ObjectId)
    user_bookmarks = get_bookmarks_list(user_id)


    return jsonify(user_bookmarks)  

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    phone = data.get('phone')
    password = data.get('password')
    if not name or not email or not phone or not password:
        return jsonify({"error": "Missing fields"}), 400

    user_collection = db['users']
    if user_collection.find_one({"email": email}):
        return jsonify({"error": "Email already registered"}), 400

    user_id = user_collection.insert_one({
        "name": name,
        "email": email,
        "phone": phone,
        "password": password,  # Store the password as provided
        "role": "user",
        "status": "pending"
    }).inserted_id
    return jsonify({"message": "User registered successfully", "user_id": str(user_id)}), 201

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/ping')
def test_api():
    return jsonify(message="Pong"), 200
