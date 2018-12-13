#FROM ubuntu:16.04
#FROM amazon/aws-eb-python:3.4.2-onbuild-3.5.1
FROM python:3

RUN apt-get update -y && \
#    apt-get install -y python-pip python-dev && \
    pip install --upgrade pip

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT [ "python" ]

CMD [ "ml-api.py" ]

# docker build -t my-app-image .
# docker run -it --rm -p 5000:8080 my-app-image