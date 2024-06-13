FROM python:3.11

WORKDIR /app

COPY . /app

#RUN pip install poetry && \
#  poetry config virtualenvs.create false && \
#  poetry install --no-interaction --no-ansi --only main

COPY ./requirements.txt /usr/src/app/requirements.txt

# dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && rm -rf /root/.cache/pip

CMD exec uvicorn app.server:app --host 0.0.0.0 --port 8001
