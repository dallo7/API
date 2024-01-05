import pandas as pd
import pickle
from flask import Flask, request

app = Flask(__name__)


@app.route("/pred")
def user():
    return "I like"


@app.route("/predict", methods=["POST"])
def create_user():
    data = request.get_json()

    # print(data["product"], data["quote"])

    test = {'quote': data["quote"], 'product': data["product"]}

    new_df = pd.DataFrame(test, index=[8])

    amp = pd.get_dummies(new_df)

    amp = amp.to_dict()

    sch = {'quote': {8: 0},
           'product_gas': {8: 0},
           'product_maize': {8: 0},
           'product_metal': {8: 0},
           'product_oil': {8: 0},
           'product_wheat': {8: 0}}

    test_df = {**sch, **amp}

    pred = pd.DataFrame(test_df, index=[8])

    with open('predicktor.pkl', 'rb') as f:
        predictor_load = pickle.load(f)

    parsed = predictor_load.predict(pred)[0]

    if parsed == 1:
        parsed = "Accepted"
    else:
        parsed = "Rejected"

    payload = f"Your quote for this product is {parsed}"

    return payload


if __name__ == "__main__":
    app.run(debug=True)
