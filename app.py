import gdown
import os
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

# Google Drive file IDs
clf_id = '11Aw3tQPDgERKRGgV6MAjZV_UdM6QgxkY'
tfidf_id = '1bAT4OUXIEYBeewFnn68R1354AoWCzGYl'

# Generate direct download links
clf_url = f'https://drive.google.com/uc?id={clf_id}'
tfidf_url = f'https://drive.google.com/uc?id={tfidf_id}'

# Download model files if not present
if not os.path.exists("clf.pkl"):
    gdown.download(clf_url, "clf.pkl", quiet=False)
if not os.path.exists("tfidf.pkl"):
    gdown.download(tfidf_url, "tfidf.pkl", quiet=False)

# Load models
clf = joblib.load("clf.pkl")
tfidf = joblib.load("tfidf.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tfidf.transform(data).toarray()
        prediction = clf.predict(vect)
        return render_template('index.html', prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
