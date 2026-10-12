#!/bin/bash
# Launch Streamlit app for Crop Disease Severity & Full Supervision Studio
SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if [ -d "${SOURCE_DIR}/venv" ]; then
    source "${SOURCE_DIR}/venv/bin/activate"
fi
streamlit run "${SOURCE_DIR}/app.py"
