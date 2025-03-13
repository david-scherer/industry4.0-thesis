# pull official base image
FROM python:3.10-slim

WORKDIR /app

COPY . .

# install dependencies
RUN pip install --upgrade pip \
    && apt-get update \
    && apt-get -y install libpq-dev gcc

RUN pip install -r requirements.txt

ENTRYPOINT [ "python", "dave_opcua_client.py" ]


