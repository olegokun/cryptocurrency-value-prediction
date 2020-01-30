# -*- coding: utf-8 -*-
"""
Created on Wed Jan  30 09:26:00 2020

@author: Oleg
"""


from flask import Flask, jsonify, render_template
import os
import pandas as pd
from graph import build_graph
from cryptonic.models.model import Model
from cryptonic.markets.coinmarketcap import CoinMarketCap
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
env_path = Path('.')/'crypto.env'
load_dotenv(dotenv_path=env_path)

'''
Main script.
To run this script from Spyder: 
    Choose menu options: Run -> Configuration per ... . 
    When inside, select option 'Execute in an external system terminal'
'''
app = Flask(__name__)

# set up a path to the current directory
cur_dir = os.path.dirname(__file__)
model_path = os.path.join(cur_dir, os.getenv('MODEL_NAME'))

model = None

def get_historic_data():
    """
    Load historic data and perform necessary filtering.
    """
    historic_data = CoinMarketCap.historic(ticker=os.getenv('COIN_TYPE'))
    time_ago = (datetime.now() - timedelta(days=7*int(os.getenv('WEEKS_BACK')))).strftime('%Y-%m-%d')
    model_data = historic_data[historic_data['date'] >= time_ago]
    return model_data

def load_model(model_data):
    """
    Load a model from a file, together with data.
    """
    model = Model(path=model_path,
                  data=model_data,
                  variable='close',
                  predicted_period_size=int(os.getenv('PERIOD_SIZE')),
                  model_type='functional')
    return model


######## Flask routines
@app.route('/')
def home_page():
    return """
Hello, Friends! To see model predictions, add '/predict' to this page URL.
To see a plot of both historic and predicted values, add '/graphs' to this page URL
"""

@app.route('/graphs')
def graphs():
    """
    Plot a graph of historic data and model predictions.
    """
    # Get historic data used to train a model and model predictions
    x = model.data
    y = model.predict(output=1, denormalized=True, return_dict=True)
    # Pre-process predictions
    y = pd.DataFrame(y)
    # Rename the column 'prediction' into 'close' in order to conform with 
    # column naming of the original data
    y.rename(columns={'prediction': 'close'}, inplace=True)
    # Select only two columns that both historic observations and predictions share in common
    # Sort observations from the earliest date to the latest in order to conform with predictions format
    x = x[['date', 'close']].sort_values(by=['date'], ascending=True)
    graph_url = build_graph(x, y);
    return render_template('graphs.html', graph=graph_url)
 
@app.route('/predict')
def predict():
    """
    Endpoint for predicting bitcoin prices.
    """
    quantile_names = ['lower', 'median', 'upper']
    for i in range(0,3):
        # Make prediction by using a given model
        # Select one out of three model outputs
        predictions = model.predict(output=i, denormalized=True, return_dict=True)
        # Convert the result to a DataFrame
        predictions = pd.DataFrame(predictions)
        # Rename a column with predictions
        predictions.rename(columns={'prediction': quantile_names[i]+' prediction'}, inplace=True)
        if i == 0:
            # If this is the first time, just store current predictions
            all_predictions = predictions
        else:
            # Otherwise, append the current predictions to the already collected ones
            all_predictions = pd.merge(all_predictions, predictions, on='date')
    
    r = {
            'success': True,
            'message': 'Endpoint for making predictions.',
            'period_length': os.getenv('PERIOD_SIZE', 7),
            'result': all_predictions.set_index('date').to_dict('index')
            }
    return jsonify(r)

if __name__ == '__main__':
    data = get_historic_data()
    model = load_model(data)
    app.run(debug=True, host='0.0.0.0')
