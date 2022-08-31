FROM python:3.9-slim-buster

RUN apt-get update && \
    apt-get install -y \
    # build-essential \
    # git curl python3-dev \
    libatlas3-base \
    # libatlas-base-dev liblapack-dev \
    libxml2 \
    # libxml2-dev \
    libffi6 
    # libffi-dev musl-dev libxslt-dev

# RUN apt-get -yqq update
# RUN apt-get install -yqq python3
# RUN apt-get -yqq install python3-pip

COPY . /app

WORKDIR /app

RUN pip3 install -r requirements.txt
RUN pip3 install .

ENTRYPOINT [ "python3" ]

CMD [ "/app/talon/web/bootstrap.py" ]
