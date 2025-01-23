# fwi-predict

Python package for predicting water quality parameters for aquaculture facilities using satellite data and machine learning.

## Setup

### Poetry

Poetry is our package manager - it helps install and manage all the Python libraries we need.

1. Install [Poetry](https://python-poetry.org/docs/#installation) following the official documentation.
2. Configure Poetry to create virtual environments in the project directory:
   ```
   poetry config virtualenvs.in-project true
   ```
3. Install project dependencies:
   ```
   poetry install
   ```
4. To install new packages:
   ```
   poetry add {package name}
   ```
4. To run Python scripts using the virtual environment:
   ```
   poetry shell
   python your_script.py
   ```

### Google Cloud Setup

Google Cloud provides the computing power we need for processing large amounts of satellite data.

1. Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
2. Initialize the SDK and authenticate:
   ```
   gcloud init
   gcloud auth application-default login
   ```
3. Request access to the Google Cloud project (contact sebquaade@gmail.com)

### Google Earth Engine

Google Earth Engine gives us access to satellite imagery and data that we use for our predictions.

1. Create a [Google Earth Engine](https://earthengine.google.com/) account
2. Authenticate your environment:
   ```
   earthengine authenticate
   ```

### Streamlit
Streamlit helps us create a user-friendly web interface for our predictions. Validate your [`streamlit`](https://streamlit.io/) installation by running:
