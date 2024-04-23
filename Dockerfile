FROM python:3.9.13-slim
ENV MILVUS_URL http://localhost:19530
ENV SLEEP_FOR 25
RUN mkdir -p /app
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt --no-cache-dir 
EXPOSE 8000
ENTRYPOINT [ "python", "main.py" ]