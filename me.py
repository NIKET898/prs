from urllib import request
import sklearn
import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

model = pickle.load(open("LinearRegressionModel.pkl", 'rb'))
abc = pd.read_csv("finaldataset (1).csv")


@app.route('/')
def index():
    # zero = sorted(abc[''].unique())
    one = sorted(abc['City'].astype(str).unique())
    two = sorted(abc['Area'].unique())
    three = sorted(abc['Location'].astype(str).unique())
    four = sorted(abc['bhk'].unique())
    return render_template('index.html', one=one, two=two, three=three, four=four)


def cross_origin():
    pass


@app.route('/predict', methods=['POST'])
def predict():
    City = request.form.get('City')
    Area = (request.form.get('Area'))
    Location = request.form.get('Location')
    bhk = (request.form.get('bhk'))

    prediction = model.predict(pd.DataFrame([[City, Area, Location, bhk]], columns=['City', 'Area', 'Location', 'bhk']))
    #print(prediction)

    return str(np.round(prediction[0], 2))


if __name__ == "__main__":
    app.run(debug=True)
