#!/bin/bash

timeout 180 reading_NICER_MR.py $@

#if [[ ($? -eq 139) || ($? -eq 124) ]]; then
if [[ $? -ne 0 ]]; then
    echo "Error: either timedout or crashed."
    reading_NICER_MR_for_fails.py $@
    exit 0 # change this to either 0 or 1
fi

# either put a countdown in python, shell, or in create_eos_posterior.. TIMEOUT: https://stackoverflow.com/questions/687948/timeout-a-command-in-bash-without-unnecessary-delay
# exit status of timeout is 124 https://stackoverflow.com/questions/38534097/bash-if-command-timeout-execute-something-else