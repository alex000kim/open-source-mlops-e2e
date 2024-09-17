FROM python:3.10
ENV PYTHONPATH=/app
WORKDIR /app
COPY . .
RUN pip install -U pip 
RUN pip install -r requirements.txt
EXPOSE 8080
CMD uvicorn src.app.main:app --host 0.0.0.0 --port 8080