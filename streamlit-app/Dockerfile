FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "run_model.py", "--server.port=8501", "--server.address=0.0.0.0"]
