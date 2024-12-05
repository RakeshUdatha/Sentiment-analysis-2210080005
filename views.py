
import pickle
from django.shortcuts import render

# Load the pre-trained model
with open('sentiment/sentiment_model.pkl', 'rb') as f:
    vectorizer, model = pickle.load(f)

def analyze_sentiment(request):
    if request.method == 'POST':
        text = request.POST.get('text')
        if not text:
            return render(request, 'analysis/index.html', {
                'error': 'Please provide some text to analyze.',
            })
        
        # Transform the text and make a prediction
        transformed_text = vectorizer.transform([text])
        prediction = model.predict(transformed_text)
        confidence = model.predict_proba(transformed_text).max()

        return render(request, 'analysis/result.html', {
            'text': text,
            'prediction': prediction[0],
            'confidence': round(confidence * 100, 2),
        })
    
    return render(request, 'analysis/index.html')
