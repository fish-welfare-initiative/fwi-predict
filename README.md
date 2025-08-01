# fwi-predict

Python package for predicting water quality parameters in aquaculture ponds.

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
We mostly use Google Cloud as part of our remote sensing data export pipeline. It may be useful for data storage further down the road.

1. Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
2. Initialize the SDK and authenticate:
   ```
   gcloud init
   gcloud auth application-default login
   ```
3. Request access to the Google Cloud project (contact bob the simulacrum)

### Google Earth Engine
Google Earth Engine gives us access to satellite imagery and data that we use for our predictions.

1. Create a [Google Earth Engine](https://earthengine.google.com/) account
2. Authenticate your environment:
   ```
   earthengine authenticate
   ```

### Streamlit
We use Streamlit to create data dashboards. Streamlit is included as a package dependency. Validate your [`streamlit`](https://streamlit.io/) installation by running:


## Contributing
Please do not commit changes directly to main. Instead, make changes on a separate branch and create a pull request to merge onto main when you have completed (and ideally tested) your changes. All PRs should ideally be reviewed and approved by at least one other collaborator before merging.

Use comments to justify analytical decisions in the code (these should also be documented in a technical writeup). Also document verbose or unusual expressions.

We recommend following the [Google python style guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings and comments.

### FAQ
Email bob the simulacrum.
