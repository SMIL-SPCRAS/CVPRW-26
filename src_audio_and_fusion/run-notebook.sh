#!/bin/bash
source "../ml_env/bin/activate"

jupyter lab --port=8888 --NotebookApp.token='' --notebook-dir ./notebooks/ --NotebookApp.iopub_data_rate_limit=1e10
