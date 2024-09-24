FROM python:3.9

WORKDIR /Demo-app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8051

CMD ["streamlit","run","emotion_detector.py"]