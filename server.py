# imports
import logging
from flask import Flask, render_template, request
from trainings.A3C_train import A3C_train
from trainings.A3C_train import get_rewards as A3C_rewards
from trainings.ES_train import ES_train
from trainings.ES_train import get_rewards as ES_rewards
from play import start_game
from flask import jsonify
from flask import abort

#logger
logger = logging.getLogger("universe-server")
logger.setLevel(logging.INFO)

app = Flask(__name__)
app.debug = True
logger.info("HTTP server listening on port: 5000") # Default port

@app.route('/')
def index():
    """Display home/configuration page"""
    return render_template('index.html', title='Universe server')

@app.route('/train', methods = ['POST'])
def train():
    """Start training of the network"""
    model = request.form['model']
    env = request.form['env']
    num_processes = int(request.form['workers'])
    if model == "A3C":
        A3C_train(env, num_processes)
    elif model == "ES":
        ES_train(env)

    return 'Training started'


@app.route('/play', methods=['POST'])
def play():
    """Start playing environment"""
    model = request.form['model']
    env = request.form['env']
    start_game(model, env)
    return 'Play started'


@app.route('/data/<model>')
def get_data(model):
    """Return current reward data for the chart"""
    data = {}
    if model == "A3C":
        data = A3C_rewards()
    elif model == "ES":
        data = ES_rewards()
    else:
        abort(400)
    
    return jsonify(data)

