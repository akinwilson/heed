# Audio streaming application 

## Overview 
This is a django webserver application designed to stream audio from the frontend to the backend where the serving model performs predictions on this stream, returning to the frontend the probability of spotting the keyword. 


## Usage 
Run the server in development mode with

```
python manage.py runserver 127.0.0.1:8000
```
This will autoload changes to the application code; `python`, but not for frontend code; `js`, `html` and `css`. 

