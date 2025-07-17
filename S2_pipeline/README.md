# Sentinel-2 Water Quality Processing Pipeline

This pipeline automates the download, atmospheric correction, band stacking, and cloud upload of Sentinel-2 Level-1C imagery for water quality analysis using ACOLITE and Google Earth Engine (GEE).

## About ACOLITE

This pipeline uses [ACOLITE](https://github.com/acolite/acolite) for atmospheric correction and aquatic product generation.  
ACOLITE is developed at RBINS and supports processing of various satellite sensors for aquatic applications.  
ACOLITE allows simple and fast processing of imagery from various satellites, including Sentinel-2/MSI, Landsat, PlanetScope, and more, for coastal and inland water applications.  
The Dark Spectrum Fitting (DSF) atmospheric correction algorithm works especially well for turbid and productive waters, but can also be applied over clear waters and land with reasonable success.

For more information, installation, and references, please see the [ACOLITE GitHub repository](https://github.com/acolite/acolite) and the [official documentation](https://odnature.naturalsciences.be/remsem/acolite-forum/).

**If you use this pipeline, please also cite ACOLITE and the relevant publications.**

## Features

- **Automated Sentinel-2 product search and download** from Copernicus Data Space Ecosystem.
- **Atmospheric correction** using ACOLITE with customizable settings.
- **Band stacking and scaling** for a canonical set of water quality bands.
- **Automatic NoData band creation** for missing bands.
- **Cloud-Optimized GeoTIFF (COG) output** and metadata manifest generation.
- **Upload to Google Cloud Storage (GCS)** and ingestion to Google Earth Engine.
- **Checkpointing** to avoid reprocessing previously completed products.
- **Supports both historical and near-realtime modes**.

## Requirements

- Python 3.8+
- [ACOLITE](https://github.com/acolite/acolite) (installed and accessible via command line)
- Google Cloud SDK and credentials
- Earth Engine Python API
- Required Python packages: see `environment.yml` or `requirements.txt`

### Install ACOLITE

Clone ACOLITE from GitHub:
```bash
git clone https://github.com/acolite/acolite.git
```
Follow ACOLITE's installation instructions in the repository.

### ACOLITE Path Configuration

After cloning ACOLITE from GitHub, **make sure to update the pipeline scripts to use the correct path to your ACOLITE installation**.  
For example, if you cloned ACOLITE to `/Users/yourname/acolite`, set the path accordingly in your configuration or script:

```python
ACOLITE_PATH = "/Users/yourname/acolite"
```

If you use a different location, **change the path to match where you cloned ACOLITE**.

> **Note:** The pipeline calls ACOLITE via command line.  
> If you move the ACOLITE folder, update the path everywhere it is referenced in your scripts or environment variables.

## Usage

### 1. Prepare Environment

- Set up your `.env` file with Copernicus credentials:
  ```
  COPERNICUS_USERNAME=your_username
  COPERNICUS_PASSWORD=your_password
  ```

- **With Conda:**  
  Create environment and install all dependencies:
  ```bash
  conda env create -f environment.yml
  conda activate s2_wq_pipeline
  ```

- **With pip:**  
  ```bash
  pip install -r requirements.txt
  ```

- Authenticate with Google Earth Engine:
  ```bash
  earthengine authenticate
  ```

### Google Cloud Storage Configuration

Before using Google Cloud Storage API, set your project and credentials with these commands:

```bash
# Login with Google Cloud SDK
gcloud auth application-default login

# List available projects
gcloud projects list

# Set quota project for Application Default Credentials
gcloud auth application-default set-quota-project your-gcp-project-id

# Set project for gcloud CLI
gcloud config set project your-gcp-project-id

# Check current project value
gcloud config get-value project

# Print current access token
gcloud auth application-default print-access-token
```

Or specify the project directly in your Python code:
```python
from google.cloud import storage
storage_client = storage.Client(project='your-gcp-project-id')
```

### 2. Run the Pipeline

#### Near-Realtime Mode (last 7 days, auto cloud cover threshold):

```bash
python 00_S2Pipeline_Final.py --mode near-realtime --cloud-cover 30
```

#### Historical Mode (custom date range):

```bash
python 00_S2Pipeline_Final.py --mode historical --start-date 2024-06-01 --end-date 2024-06-30 --cloud-cover 20
```

### 3. Output

- **Processed GeoTIFFs** and **metadata XML** are saved in the `adjusted_tifs` directory.
- **Logs** are written to `pipeline.log`.
- **Checkpointing** is handled via `pipeline_checkpoint.json` to avoid duplicate processing.
- **Uploaded files** are sent to your configured GCS bucket and ingested into GEE.

## Band List

The pipeline processes and stacks the following canonical bands (if available):

```
    'rhow_443',
    'rhow_492',
    'rhow_560',
    'rhow_665',
    'rhow_704',
    'rhow_740',
    'rhow_783',
    'rhow_833',
    'rhow_865',
    'rhow_1614',
    'rhow_2202',
    'l2_flags',
    'tur_nechad2009_665',
    'spm_nechad2010_665',
    'chl_oc3',
    'ci',
    'chl_ci',
    'chl_ocx',
    'chlor_a',
    'kd490'
```

If a band is missing from the ACOLITE output, a NoData band is created in its place.

## Customization

- **AOI**: The Area of Interest is defined in the script (buffered polygon). You can modify this section to use a different AOI or load from a GeoJSON/WKT.
- **ACOLITE settings**: The pipeline generates a settings file for each product. You can customize the parameters in the `run_acolite_cli` function.
- **Cloud Storage and GEE**: Update the `bucket_name` and `destination_folder` variables as needed.

## Troubleshooting

- Check `pipeline.log` for errors and processing details.
- Ensure all dependencies and credentials are correctly set up.
- If bands are missing in the output, verify your ACOLITE settings and input data.

## References

The Dark Spectrum Fitting (DSF) algorithm and ACOLITE are described in:

- Vanhellemont and Ruddick 2018, [Atmospheric correction of metre-scale optical satellite data for inland and coastal water applications](https://www.sciencedirect.com/science/article/pii/S0034425718303481)
- Vanhellemont 2019a, [Adaptation of the dark spectrum fitting atmospheric correction for aquatic applications of the Landsat and Sentinel-2 archives](https://doi.org/10.1016/j.rse.2019.03.010)
- Vanhellemont 2019b, [Daily metre-scale mapping of water turbidity using CubeSat imagery.](https://doi.org/10.1364/OE.27.0A1372)

### Deploying the Pipeline on Google Cloud Run

You can run this pipeline on [Google Cloud Run](https://cloud.google.com/run) for scalable, serverless execution.  
**Key steps and considerations:**

#### 1. Dockerize the Pipeline

- Create a `Dockerfile` that installs Python, ACOLITE, and all dependencies.
- Copy your pipeline code and ACOLITE installation into the image.
- Set environment variables for paths and credentials (do not hardcode local paths).

Example Dockerfile snippet:
```dockerfile
FROM python:3.10

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy pipeline code and ACOLITE
COPY . /app
COPY /path/to/acolite /app/acolite

# Set environment variables (can be overridden in Cloud Run)
ENV BASE_DIR=/app/s2_wq_data
ENV ACOLITE_PATH=/app/acolite

CMD ["python", "your_pipeline_script.py", "--mode", "historical"]
```

#### 2. Update Path Handling in Code

Replace hardcoded paths with environment variables:
```python
import os
BASE_DIR = os.getenv('BASE_DIR', '/app/s2_wq_data')
ACOLITE_PATH = os.getenv('ACOLITE_PATH', '/app/acolite')
```

#### 3. Set Up Google Cloud Credentials

- Use a service account with the required permissions.
- Set `GOOGLE_APPLICATION_CREDENTIALS` to the path of your service account key file.
- Set `GOOGLE_CLOUD_PROJECT` as an environment variable.

#### 4. Deploy to Cloud Run

Use the following commands to deploy:
```bash
gcloud builds submit --tag gcr.io/your-gcp-project-id/s2-wq-pipeline

gcloud run deploy s2-wq-pipeline \
  --image gcr.io/your-gcp-project-id/s2-wq-pipeline \
  --region YOUR_REGION \
  --set-env-vars BASE_DIR=/app/s2_wq_data,ACOLITE_PATH=/app/acolite,GOOGLE_CLOUD_PROJECT=your-gcp-project-id \
  --service-account your-service-account@your-gcp-project-id.iam.gserviceaccount.com
```

#### 5. Notes

- Cloud Run does not support interactive authentication. Use service account credentials.
- Make sure all required files and dependencies are included in the Docker image.
- Adjust entrypoint and arguments as needed for your workflow.

**See Google Cloud Run documentation for more details:**  
https://cloud.google.com/run/docs/deploying




