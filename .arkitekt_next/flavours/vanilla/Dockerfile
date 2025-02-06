FROM python:3.11-slim-buster

RUN pip install "arkitekt-next[all]>=0.8.64"
RUN pip install matplotlib

RUN mkdir /app
WORKDIR /app
COPY .arkitekt_next /app/.arkitekt_next
COPY app.py /app/app.py
