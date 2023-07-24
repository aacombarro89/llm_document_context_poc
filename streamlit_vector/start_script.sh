#!/bin/bash

# Check if the models are already installed
if [ ! -f "/usr/local/lib/python3.11/site-packages/sentence_transformers" ]; then
  # If models are not installed, install them
  # Replace the below command with your actual models installation command
  echo "Installing sentence-transformers models..."
  pip install --no-cache-dir sentence-transformers==2.2.2  # Example command, use your actual model installation command
fi

# Start your application (replace this with the actual command to start your app)
streamlit run ./streamlit_demo.py