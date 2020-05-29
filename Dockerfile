FROM python:3

WORKDIR /usr/src/app

RUN pip3 install --upgrade pip

RUN pip3 install matplotlib numpy

RUN pip3 install tensorflow 

RUN pip3 install keras

COPY . .
 
ENTRYPOINT ["python"]

CMD ["train.py"]
