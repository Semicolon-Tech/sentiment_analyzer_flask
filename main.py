from flask import Flask
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, Semicolon!'


@app.route('/ping')
def hello_world():
    return 'pong'


