from django.shortcuts import render
import os
import pickle
from django.shortcuts import render
from django.conf import settings
from .models import News
import json
import pandas as pd

# ✅ Load model & vectorizer safely
MODEL_PATH = r"D:\Fake_News_Detection\model"  # <-- tumhara path
model = pickle.load(open(os.path.join(MODEL_PATH, 'fake_news_model.pkl'), 'rb'))
vectorizer = pickle.load(open(os.path.join(MODEL_PATH, 'vectorizer.pkl'), 'rb'))

# CSV file path
CSV_PATH = r"D:\Fake_News_Detection\dataset\fake_real_news.csv"

def dashboard(request):
    news = News.objects.all()
    fake_count = news.filter(result="Fake").count()
    real_count = news.filter(result="Real").count()

    chart_data = {
        'labels': ['Fake', 'Real'],
        'data': [fake_count, real_count]
    }

    return render(request, 'dashboard.html', {
        'news': news,
        'fake': fake_count,
        'real': real_count,
        'chart_data': json.dumps(chart_data)
    })

def predict(request):
    result = ''
    csv_news = []

    # Read CSV to show in dropdown
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        csv_news = df['text'].tolist()

    if request.method == "POST":
        # news either from dropdown or textarea
        text = request.POST.get('news_text') or request.POST.get('news_dropdown')

        vect = vectorizer.transform([text])
        prediction = model.predict(vect)[0]
        result = "Real" if prediction == 1 else "Fake"

        # Save to DB
        News.objects.create(text=text, result=result)

    return render(request, 'predict.html', {
        'result': result,
        'csv_news': csv_news
    })

# 🔥 Safe load of model & vectorizer

'''MODEL_PATH = r"D:\Fake_News_Detection\model"
#MODEL_PATH = os.path.join(settings.BASE_DIR, 'model')

model = pickle.load(open(os.path.join(MODEL_PATH, 'fake_news_model.pkl'), 'rb'))
vectorizer = pickle.load(open(os.path.join(MODEL_PATH, 'vectorizer.pkl'), 'rb'))

# CSV file path
CSV_PATH = r"D:\Fake_News_Detection\dataset\fake_real_news.csv"
# -----------------
# Dashboard view
# -----------------
def dashboard(request):
    news = News.objects.all()
    fake_count = news.filter(result="Fake").count()
    real_count = news.filter(result="Real").count()

    # For chart.js
    chart_data = {
        'labels': ['Fake', 'Real'],
        'data': [fake_count, real_count]
    }

    return render(request, 'dashboard.html', {
        'news': news,
        'fake': fake_count,
        'real': real_count,
        'chart_data': json.dumps(chart_data)
    })

# -----------------
# Predict view
# -----------------
def predict(request):
    result = ''
    if request.method == "POST":
        text = request.POST['news']

        vect = vectorizer.transform([text])
        prediction = model.predict(vect)[0]

        result = "Real" if prediction == 1 else "Fake"

        # Save to DB
        News.objects.create(text=text, result=result)

    return render(request, 'predict.html', {'result': result})'''
