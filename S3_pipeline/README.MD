# Sentinel-3 Water Quality Pipeline

## About Product

Sentinel-3 Water Quality Pipeline is an automated workflow for downloading, processing, and publishing Sentinel-3 OLCI water quality products (Chlorophyll-a, TSM) from EUMETSAT. It converts NetCDF files to georeferenced GeoTIFFs, generates metadata manifests, uploads results to Google Cloud Storage (GCS), and ingests data into Google Earth Engine (GEE).

---

## Features

- Search and download Sentinel-3 OLCI products by location and date range
- Extract and convert NetCDF bands (`chl_nn.nc`, `tsm_nn.nc`) to GeoTIFF with georeferencing
- Stack multiple bands into a single GeoTIFF
- Generate XML manifest with metadata and band statistics
- Upload processed files to Google Cloud Storage
- Ingest assets into Google Earth Engine

---

## Requirements

- Python 3.10
- Conda (recommended)
- EUMETSAT Data Store account
- Google Cloud Platform project and credentials
- Earth Engine account

**Main Python dependencies:**  
`eumdac`, `shapely`, `numpy`, `rasterio`, `xarray`, `scipy`, `google-cloud-storage`, `earthengine-api`, `python-dotenv`

---

## Usage

### 1. Prepare Environment

#### Create Conda Environment

```bash
conda env create -f environment.yml
conda activate s3_env
```

#### Configure Environment Variables

Create a `.env` file in your project folder:

```env
EUMETSAT_USERNAME=your_eumetsat_username
EUMETSAT_PASSWORD=your_eumetsat_password
GCS_BUCKET_NAME=your_gcs_bucket
COLLECTION_ID=EO:EUM:DAT:0407
```

#### Google Cloud Storage Configuration

Before using Google Cloud Storage API, set your project and credentials:

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

---

### 2. Run the Pipeline

```bash
# Historical mode (manual date range)
python S3_Pipeline_dual_final.py --mode historical --start-date YYYY-MM-DD --end-date YYYY-MM-DD

# Dynamic mode (recent N days)
python S3_Pipeline_dual_final.py --mode dynamic --days-back 2
```

---

### 3. Output

- GeoTIFF files (stacked bands, georeferenced)
- XML manifest with metadata and band statistics
- ZIP archive of GeoTIFF + manifest
- Upload to GCS and optional GEE ingestion

#### Band List

- `chl_nn` : Chlorophyll-a concentration
- `tsm_nn` : Total Suspended Matter

---

## Cloud Run Preparation

Before deploying to Cloud Run, make sure you:

1. **Containerize the Pipeline**
   - Create a `Dockerfile` for your pipeline.
   - Install all dependencies in the container.
   - Set the entrypoint to run `S3_Pipeline_dual_final.py`.

2. **Set Environment Variables**
   - Use Cloud Run environment variables or Secret Manager for credentials (EUMETSAT, GCS, etc.).

3. **Google Cloud Authentication**
   - Ensure your Cloud Run service has access to GCS and Earth Engine.
   - Use Workload Identity or mount a service account.

4. **Input/Output Handling**
   - Adjust the pipeline to accept input via HTTP request or Cloud Pub/Sub if needed.
   - Output results to GCS or return via HTTP response.

5. **Resource & Timeout**
   - Set appropriate memory and timeout settings for Cloud Run.

6. **Logging & Monitoring**
   - Use Cloud Logging for monitoring pipeline execution.

7. **Testing**
   - Test your pipeline locally with Docker before deploying to Cloud Run.

---

## Deploying the Pipeline on Google Cloud Run

1. Containerize your pipeline (e.g., with Docker).
2. Push the image to Google Container Registry.
3. Deploy to Cloud Run:

```bash
gcloud run deploy s3-pipeline \
  --image gcr.io/your-gcp-project-id/s3-pipeline:latest \
  --platform managed \
  --region your-region \
  --allow-unauthenticated
```