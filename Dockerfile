FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY svm_svmlight.py .

ENTRYPOINT ["python", "svm_svmlight.py"]
