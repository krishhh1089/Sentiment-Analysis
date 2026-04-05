from flask import Flask, render_template, request
import pickle
from prep import *

app = Flask(__name__)

# Load components
preprocess_pipeline = pickle.load(open("text_preprocessing_pipeline.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
model = pickle.load(open("sentiment_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_text = request.form["input_text"]
    cleaned_text = preprocess_pipeline.transform([user_text])
    tfidf_vector = tfidf.transform(cleaned_text)
    prediction = model.predict(tfidf_vector)[0]

    return render_template("index.html",
                           input_text=user_text,
                           prediction=prediction)
if __name__ == "__main__":
    app.run(debug=True)