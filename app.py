from flask import Flask, render_template, request
import pickle
from prep import *

app = Flask(__name__)

# Load components
preprocess_pipeline = pickle.load(open("text_preprocessing_pipeline.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
model = pickle.load(open("sentiment_model.pkl", "rb"))

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=None, input_text="")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["input_text"]

    # 1️⃣ Preprocess text
    cleaned_text = preprocess_pipeline.transform([text])

    # 2️⃣ Vectorize
    vectorized = tfidf.transform(cleaned_text)

    # 3️⃣ Predict
    pred = model.predict(vectorized)[0]

    return render_template("index.html",
                           prediction=pred,
                           input_text=text)

if __name__ == "__main__":
    app.run(debug=True)