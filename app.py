from flask import Flask
app = Flask(__name__)


@app.route('/')
def hello_semicolon():
    return 'Hello, Semicolon!'


@app.route('/ping')
def ping():
    return 'pong'


