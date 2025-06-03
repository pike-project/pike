FROM python:3.11-slim AS deps

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

COPY . /app

RUN pip install .

# make everything in the /app directory readable, and
# add execute only if it's a directory or already has execute for someone
RUN chmod -R a+rX /app

# CMD ["python", "scripts/eval_from_generations.py"]

# CMD ["sleep", "infinity"]
# CMD ["cat", "README.md"]
# CMD ["ls"]
# CMD ["pwd"]
