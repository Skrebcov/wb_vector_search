# Используем базовый образ Python
FROM python:3.11.5

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Загрузка необходимых данных NLTK
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

EXPOSE 8000

CMD ["uvicorn", "main:app_run", "--host", "0.0.0.0", "--port", "8000"]