FROM python:3.10

COPY requirement.txt /
RUN pip3 install -r /requirement.txt
RUN pip3 install gevent gunicorn
ADD . /flask
WORKDIR /flask

EXPOSE 3000

CMD  gunicorn --worker-class gevent  --bind 0.0.0.0:3005 wsgi:app --log-level debug