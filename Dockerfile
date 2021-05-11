
FROM tensorflow/tensorflow:latest-gpu

LABEL maintainer="zanardi3@illinois.edu"

WORKDIR /prode

COPY . .

RUN python3 setup.py bdist_wheel && pip3 install dist/*.whl

ENTRYPOINT [ "python3" ]

