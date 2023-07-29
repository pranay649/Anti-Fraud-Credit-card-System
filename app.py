from flask import Flask, render_template, request, send_file
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
import pandas as pd
import pickle
from wtforms.validators import InputRequired
import numpy as np
from flask import send_from_directory
import csv

model = pickle.load(open("./model.pkl", "rb"))
app = Flask(__name__)


@app.route("/")
def home1():
    return render_template("home1.html")


@app.route("/predict", methods=["POST", "GET"])
def predict():
    inputs = [float(x) for x in request.form.values()]
    inputs = np.array([inputs])
    print(inputs)
    # inputs = sc.transform(inputs)
    output = model.predict(inputs)
    if output < 0.5:
        output = 0
    else:
        output = 1

    # Pass the inputs and prediction to the result template
    return render_template("result.html", inputs=inputs[0], prediction=output)


@app.route("/index")
def Predictor():
    return render_template("index.html")


@app.route("/Contact")
def contact():
    return render_template("Contact.html")


@app.route("/about")
def about():
    return render_template("About.html")


@app.route("/OnlineFraud")
def OnlineFraud():
    return render_template("OnlineFraud.html")


@app.route("/Help")
def help():
    return render_template("Help.html")


@app.route("/predictor", methods=["POST", "GET"])
def predictor():
    model = pickle.load(open("./model1.pkl", "rb"))
    input = [float(x) for x in request.form.values()]
    input = np.array([input])
    # inputs = sc.transform(inputs)
    output1 = model.predict(input)
    if output1 < 0.5:
        output1 = 0
    else:
        output1 = 1

    # Pass the inputs and prediction to the result template
    return render_template("result1.html", input=input[0], conclusion=output1)


# Set the secret key for the app
app.config["SECRET_KEY"] = "supersecretkey"

# Set the folder to store uploaded files
app.config["UPLOAD_FOLDER"] = "./uploads"


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")


@app.route("/Upload", methods=["GET", "POST"])
def home():
    form = UploadFileForm()
    prediction_available = False
    filename = None
    result_data = None

    if form.validate_on_submit():
        file = form.file.data
        file_path = os.path.join(
            app.config["UPLOAD_FOLDER"], secure_filename(file.filename)
        )
        file.save(file_path)

        # Load the pre-trained model
        model_path = "./model1.pkl"
        model = load_model_from_pickle(model_path)

        # Load the new dataset
        new_dataset = load_dataset(file_path)

        # Prepare the data for prediction (if needed)
        new_dataset_processed = preprocess_data_for_prediction(new_dataset)

        # Make predictions using the loaded model
        predictions = predict_with_model(model, new_dataset_processed)

        new_dataset["Prediction"] = predictions

        # Save the updated DataFrame with predictions to a new CSV file
        output_file_path = os.path.join(
            app.config["UPLOAD_FOLDER"], "predictions_" + secure_filename(file.filename)
        )
        new_dataset.to_csv(output_file_path, index=False)

        prediction_available = True
        filename = "predictions_" + secure_filename(file.filename)

        # Read the prediction result CSV
        with open(output_file_path, "r") as result_file:
            result_data = list(csv.reader(result_file))

    return render_template(
        "Upload.html",
        form=form,
        prediction_available=prediction_available,
        filename=filename,
        result_data=result_data,
    )


def load_model_from_pickle(pickle_file):
    with open(pickle_file, "rb") as file1:
        model = pickle.load(file1)
    return model


def load_dataset(file_path):
    # You may need to adjust this function based on your dataset format
    return pd.read_csv(file_path)


def preprocess_data_for_prediction(dataset):
    # Perform any necessary data preprocessing to match the format used during training
    # This step may include feature scaling, encoding categorical variables, etc.
    # Make sure to apply the same transformations as during training.
    return dataset


def predict_with_model(model, dataset):
    predictions = model.predict(dataset)
    return predictions


@app.route("/download_result/<filename>", methods=["POST"])
def download_result(filename):
    directory = app.config["UPLOAD_FOLDER"]
    return send_from_directory(directory, filename=filename, as_attachment=True)


if __name__ == "__main__":
    app.config["TEMPLAES_AUTO_RELOAD"] = True
    app.run(debug=True)
