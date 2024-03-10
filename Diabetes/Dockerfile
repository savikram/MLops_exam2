# pull python base image
FROM python:3.10

# copy application files
ADD /Diabetes_API_app /Diabetes_API_app/

# specify working directory
WORKDIR /Diabetes_API_app

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

# expose port for application
EXPOSE 8001

# start fastapi application
CMD ["python", "app/main.py"]