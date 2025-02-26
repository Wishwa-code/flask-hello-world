# README

This is the [Flask](http://flask.pocoo.org/) [quick start](http://flask.pocoo.org/docs/1.0/quickstart/#a-minimal-application) example for [Render](https://render.com).

The app in this repo is deployed at [https://flask.onrender.com](https://flask.onrender.com).

## Deployment Setup

Follow the guide at [here](https://render.com/docs/deploy-flask)

## Development Setup

First ensure you are using this python version to run this application - [python version]

Step 1. Create new enviroment | python -m venv venv | [python] [-m] [-flag create new enviroment] [name for the new enviroment]

Step 2. Activate created new enviroment | .\venv\Scripts\activate  | [.\ : curent directory] [name of the enviroment] [\Scripts: This is a standard folder created inside the virtual environment ] [\activate: This is the activation script that sets up the virtual environment ]

Now your directory should looks like this meaning your inside directory | (venv) : C:\your\current\directory>

Step 3. Install the required libraries | pip install -r requirements.txt | Run the next command if you want to run thte application in debug mode | flask --app app run --debug

Now you will be asked to add flask to your path once flask is installed get the path for the location of the flask executable file for the system. Then add it to the the path. when running flask if flask is not running python from the place where you have specified to use you might to tweak some path variables to make sure its using python from correct location.

Once your all set run the application and it will start the process  with this warning | WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
