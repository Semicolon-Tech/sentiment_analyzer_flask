from functools import wraps

import joblib

from flask_cors import CORS
from flask import Flask, jsonify, request

from marshmallow import Schema, fields, ValidationError

model = joblib.load("model/multinomial_naive_bayes_with_tfidf_vectorizer.joblib")


class PredictSchema(Schema):
    text = fields.String(required=True)


def validate_json(schema_class):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get Request body from JSON
            schema = schema_class()

            try:
                # Validate request body against schema data types
                schema.load(request.json)
                return f(*args, *kwargs)

            except ValidationError as err:
                # Return a nice message if validation fails
                return jsonify(err.messages), 422

        return decorated_function
    return decorator


app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=["POST"])
@validate_json(schema_class=PredictSchema)
def predict_controller():
    # predict
    predicted_sentiment = model.predict([request.json.pop("text")])  # prediction

    # the final response to send back
    response = {"output": "positive" if predicted_sentiment else "negative"}
    return response


@app.route('/ping')
def ping():
    return 'pong'


if __name__ == '__main__':
    app.run()
