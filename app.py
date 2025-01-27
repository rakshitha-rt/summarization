from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)

# Load the summarizer model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return render_template('index.html', original_text=text, summary=summary[0]['summary_text'])

if __name__ == "__main__":
    app.run(debug=True)
