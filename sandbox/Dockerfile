FROM python:3.11-slim

WORKDIR /workspace

COPY . /workspace

RUN pip install --upgrade pip && pip install -r requirements.txt && pip install -e .

# CMD ["python", "app.py"]

# CMD ["sleep", "infinity"]
# CMD ["cat", "README.md"]
CMD ["ls"]
# CMD ["pwd"]
