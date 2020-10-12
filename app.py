from flask import Flask
app = Flask(__name__)


@app.route('/')
def ping():
    return 'Hello, Semicolon!'


@app.route('/ping')
def ping():
    return 'pong'


if __name__ == '__main__':
    app.run()