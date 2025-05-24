#!/bin/bash
#    File Name: run-dock.sh
#      Created: 20201210-0744
#      Purpose: Run docker container image

NAME=talon-web

docker run -p 5505:5505 -it $NAME /app/talon/web/bootstrap.py
