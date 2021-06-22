from flask.helpers import make_response
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
from flask_cors import CORS,cross_origin
from scipy.stats import mode

app = Flask(__name__)
CORS(app, resources={r"/predict/*": {"origins": "*"}})
random_forest = pickle.load(open('random_forest.pkl', 'rb'))
decision_tree = pickle.load(open('decision_tree.pkl', 'rb'))
xgboost = pickle.load(open('xgboost.pkl', 'rb'))

@app.route('/predict',methods=['POST'])
@cross_origin()
def results():

    data = request.get_json(force=True)
    prediction_rf = random_forest.predict(np.array(list(data.values())).reshape(1, -1))
    prediction_dt = decision_tree.predict(np.array(list(data.values())).reshape(1, -1))
    prediction_xg = xgboost.predict(np.array(list(data.values())).reshape(1, -1))

    final_prediction=mode([prediction_rf,prediction_dt,prediction_xg])[0][0]
    print([prediction_rf[0],prediction_dt[0],prediction_xg[0]])
    print(np.array(list(data.values())).reshape(1, -1))
    return make_response(jsonify({"result":str(final_prediction[0])}))

@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"

if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(debug=True,port=port)