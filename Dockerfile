FROM python:3.11-slim

#system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libjpeg62-turbo \
    zlib1g \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

#install python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

#copy source code
COPY . /app

#default command: show help
CMD ["python", "cli.py", "--help"]