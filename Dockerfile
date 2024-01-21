FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

ENV FLASK_APP=app.py

EXPOSE 8501

CMD exec python -m streamlit run app.py