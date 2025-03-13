FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . /app/scheduler
COPY ./config/config_docker.ini /app/scheduler/config/config.ini

ENV PYTHONPATH "${PYTHONPATH}:/app"
ENV FLASK_APP "/app/scheduler/app.py" 

CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]