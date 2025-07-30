FROM python:3.12-slim

RUN pip install matplotlib
RUN pip install "arkitekt-next[all]>=0.21"

RUN mkdir /app
WORKDIR /app
COPY .arkitekt_next /app/.arkitekt_next
COPY app.py /app/app.py
