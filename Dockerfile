FROM python:3.9.9-bullseye
RUN mkdir /app
WORKDIR /app
#Copy all files
COPY . /app
# Install dependencies
RUN pip install --upgrade pip
RUN pip install  --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
# Run
CMD ["python","app.py"]