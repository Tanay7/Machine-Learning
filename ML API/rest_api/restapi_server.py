from flask import Flask, request, Response
from flask_httpauth import HTTPBasicAuth
import ML_backend as model
import json


app = Flask(__name__)
auth = HTTPBasicAuth()

USER_DATA = {
    "admin": "SuperSecretPwd"
}


@auth.verify_password
def verify(username, password):
    if not (username and password):
        return False
    return USER_DATA.get(username) == password


@app.route('/PricePrediction/api/v1.0/', methods=['GET'])
@auth.login_required
def house_price_prediction():

    result = {}

    try:

        data = request.json
        instance = model.RandomForestModel("C:\\Users\\Tanay\\Desktop\\Genesis\\1_jan_2019\\rest_api\\Ml_model\\RandomForest.sav", data)
        pred = instance.main_process()

        return json.dumps(pred)

    except Exception as e:

        result['status'] = "Failed"
        result['message'] = "Something went wrong!"
        result['error'] = "Error: {}".format(e)
        result['result'] = {'predictions': []}

        return json.dumps(result)


if __name__ == '__main__':
    app.run(debug=True)
