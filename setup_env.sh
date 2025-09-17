#!/bin/bash
# setup.sh
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/pip freeze > requirements.txt
echo "Setup complete and requirements.txt updated. To use: source venv/bin/activate"
