# imports
from flask import Flask, render_template, request
from train import ES_train, A3C_train


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
    if model == "A3C":
        pass:
    elif model == "ES":
        pass:

@app.route('/play', methods = ['POST'] )
def play():
    """Start playing environment"""
    model = request.form['model']
    env = request.form['env']
        if model == "A3C":
        pass:
    elif model == "ES":
        pass: