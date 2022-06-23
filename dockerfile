FROM --platform=linux/x86_64 python:3.8-slim-buster
RUN apt update && apt -y update
RUN apt install -y build-essential git
RUN pip3 install --upgrade --quiet pip
RUN git clone https://github.com/alteryx/nlp_primitives.git
WORKDIR "/nlp_primitives/"
RUN make installdeps-complete
RUN pip3 install ".[test]"
RUN pytest nlp_primitives/tests/test_lsa.py