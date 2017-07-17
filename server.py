# imports
from flask import Flask, render_template, request
from trainings.A3C_train import A3C_train
from trainings.ES_train import ES_train
from play import start_game


# create app
app = Flask(__name__)
app.config.from_object(__name__)

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

@app.route('/play', methods = ['POST'] )
def play():
    """Start playing environment"""
    model = request.form['model']
    env = request.form['env']
    start_game(model, env)
