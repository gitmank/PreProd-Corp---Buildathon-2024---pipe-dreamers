# get image
FROM python:3.10-slim

# set working directory
WORKDIR /worker

# copy requirements file
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .

# run worker
CMD ["python", "worker.py"]