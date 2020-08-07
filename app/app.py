import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(ROOT_DIR)

from methods.moving_methods import MovingMethods
from methods.autoregression import AutoregressionMethods
from methods.cnn import ConvolutionalNeuralNetwork
from methods.rnn import RecurrentNeuralNetwork

import pandas as pd
from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename, redirect


app = Flask(__name__)
UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'csv'}
config = {"UPLOAD_FOLDER": os.getcwd() + "/" + UPLOAD_FOLDER}
app.config.from_mapping(config)


@app.route("/")
def index():
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            print(secure_filename(file.filename))
            file_name = os.path.join(
                app.config['UPLOAD_FOLDER'], 'dataset.csv')
            file.save(file_name)
            df = pd.read_csv(file_name)
            first_5_rows = df.head()
            return render_template(
                'index.html',
                filename=file_name,
                data=first_5_rows.to_html(),
                all_data=df.to_html())
    return redirect(request.url)


@app.route('/moving_methods')
def moving_methods():
    mv_methods = MovingMethods()

    mv_methods.moving_average_forecast()
    mv_methods.moving_average_errors()

    mv_methods.weighted_average_forecast()
    mv_methods.weighted_average_errors()

    mv_methods.exp_moving_average_forecast()
    mv_methods.exp_moving_average_errors()

    mv_methods.double_exp_moving_average_forecast()
    mv_methods.double_exp_moving_average_errors()

    mv_methods.holt_winters_method()
    mv_methods.holt_winters_method_errors()

    return render_template('moving_methods.html')


@app.route('/autoregression_methods')
def autoregression_methods():
    auto_reg_mehtods = AutoregressionMethods()
    auto_reg_mehtods.ar()
    auto_reg_mehtods.arma()
    auto_reg_mehtods.arima()
    return render_template('autoregression_methods.html')


@app.route('/neural_networks')
def neural_networks():
    cnn = ConvolutionalNeuralNetwork()
    cnn.start()
    rnn = RecurrentNeuralNetwork()
    rnn.start()
    return render_template('neural_networks.html')


@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


if __name__ == "__main__":
    app.run(debug=True)
