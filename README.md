# Sentiment Analysis Web App

A Flask-based web application for sentiment analysis of text using machine learning. The app analyzes input text and predicts whether the sentiment is positive or negative.

## Features

- Text preprocessing with NLTK (tokenization, lemmatization, stopword removal, negation handling)
- TF-IDF vectorization
- Machine learning model (Logistic Regression or Multinomial Naive Bayes)
- Simple web interface for text input and prediction display

## Dataset

The model is trained on the IMDB Dataset, which contains movie reviews labeled as positive or negative.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd sentiment-analysis
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download NLTK data (if not already downloaded):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   nltk.download('averaged_perceptron_tagger')
   ```

## Usage

1. Ensure the trained model files are present:
   - `text_preprocessing_pipeline.pkl`
   - `tfidf.pkl`
   - `sentiment_model.pkl`

2. Run the application:
   ```bash
   python app.py
   ```

3. Open your browser and go to `http://127.0.0.1:5000/`

4. Enter text in the textarea and click "Predict" to see the sentiment analysis result.

## Project Structure

- `app.py`: Main Flask application
- `prep.py`: Text preprocessing utilities
- `sentiment.ipynb`: Jupyter notebook for model training and experimentation
- `IMDB Dataset.csv`: Training dataset
- `templates/index.html`: Web interface template
- `static/style.css`: CSS styling
- `requirements.txt`: Python dependencies
- `.gitignore`: Git ignore file

## Model Training

To train or retrain the model, use the `sentiment.ipynb` notebook. It includes:

- Data loading and exploration
- Text preprocessing pipeline
- Feature extraction with TF-IDF
- Model training (Logistic Regression with hyperparameter tuning)
- Model evaluation and saving

## Dependencies

- Flask: Web framework
- NLTK: Natural language processing
- scikit-learn: Machine learning library
- pandas: Data manipulation
- numpy: Numerical computing
- scipy: Scientific computing

## License

[Add your license here]</content>
<parameter name="filePath">c:\Users\gauta\sentiment analysis\README.md