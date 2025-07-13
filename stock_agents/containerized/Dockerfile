FROM python:3.11-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY stock_agent.py . 
COPY startup.sh .


RUN chmod +x startup.sh

RUN mkdir -p /app/data /app/output

EXPOSE 8000

CMD ["./startup.sh"]