from dotenv import load_dotenv
load_dotenv()

import os
import datetime
import shutil
import eumdac
from shapely.geometry import Polygon
import numpy as np
import rasterio
import xarray as xr
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
import scipy.interpolate
import fnmatch
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
import hashlib
import sys
import zipfile
from google.cloud import storage
import tempfile
import logging
import argparse
import io
import ee
import subprocess
from datetime import datetime

ee.Initialize(project="servir-sea-landcover")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration (Environment Variables) ---
EUMETSAT_USERNAME = os.getenv("EUMETSAT_USERNAME")
EUMETSAT_PASSWORD = os.getenv("EUMETSAT_PASSWORD")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "s3_test1")
COLLECTION_ID = os.getenv("COLLECTION_ID", "EO:EUM:DAT:0407")

# Validate environment variables
if not EUMETSAT_USERNAME or not EUMETSAT_PASSWORD:
    logger.error("EUMETSAT credentials not set in environment variables")
    sys.exit(1)

# Search parameters
CENTER_LON, CENTER_LAT = 100.251297, 5.385047
BUFFER_DEG = 20 / 111.0  # Approx 20 km buffer in degrees

bbox = Polygon([
    (CENTER_LON - BUFFER_DEG, CENTER_LAT - BUFFER_DEG),
    (CENTER_LON + BUFFER_DEG, CENTER_LAT - BUFFER_DEG),
    (CENTER_LON + BUFFER_DEG, CENTER_LAT + BUFFER_DEG),
    (CENTER_LON - BUFFER_DEG, CENTER_LAT + BUFFER_DEG)
])

# Conversion and Manifest Configuration
TARGET_PATTERNS = [
    '*chl_nn.nc',
    '*tsm_nn.nc'
]

# --- Parse Command-Line Arguments ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process Sentinel-3 data with historical or dynamic date range.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["historical", "dynamic"],
        default="dynamic",
        help="Mode to select date range: 'historical' for manual dates, 'dynamic' for recent data (default: dynamic)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date in YYYY-MM-DD format (required for historical mode)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date in YYYY-MM-DD format (required for historical mode)"
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=2,
        help="Number of days to go back for dynamic mode (default: 2)"
    )
    return parser.parse_args()

def parse_date(date_str, default):
    """Parse a date string in YYYY-MM-DD format to a datetime object with UTC timezone."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return dt
    except ValueError as e:
        logger.error(f"Invalid date format: {date_str}. Expected YYYY-MM-DD. Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error parsing date {date_str}: {str(e)}")
        sys.exit(1)

# --- Set Date Range ---
args = parse_arguments()
current_utc = datetime.now(timezone.utc)

if args.mode == "historical":
    if not args.start_date or not args.end_date:
        logger.error("Both --start-date and --end-date are required in historical mode.")
        sys.exit(1)
    dtstart = parse_date(args.start_date, None)
    dtend = parse_date(args.end_date, None)
    if dtstart >= dtend:
        logger.error("Start date must be earlier than end date in historical mode.")
        sys.exit(1)
    logger.info(f"Running in historical mode with date range: {dtstart} to {dtend} (UTC)")
else:
    # Dynamic mode: Use current date minus days_back
    dtend = current_utc
    dtstart = current_utc - timedelta(days=args.days_back)
    logger.info(f"Running in dynamic mode with date range: {dtstart} to {dtend} (UTC)")

# --- Helper Functions ---
def cleanup_file(file_path):
    """Remove a single file."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Successfully removed file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to remove file {file_path}: {str(e)}")

def upload_to_gcs(local_path, bucket_name, destination_blob_name):
    """Upload a file to Google Cloud Storage."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_path)
        logger.info(f"Successfully uploaded {local_path} to {bucket_name}/{destination_blob_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload {local_path} to {bucket_name}/{destination_blob_name}: {str(e)}")
        return False

def download_from_gcs(bucket_name, source_blob_name, local_path):
    """Download a file from Google Cloud Storage."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(local_path)
        logger.info(f"Successfully downloaded {bucket_name}/{source_blob_name} to {local_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {bucket_name}/{source_blob_name} to {local_path}: {str(e)}")
        return False

def delete_from_gcs(bucket_name, blob_name):
    """Delete a file from Google Cloud Storage."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.delete()
        logger.info(f"Successfully deleted {bucket_name}/{blob_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete {bucket_name}/{blob_name}: {str(e)}")
        return False

def rename_sentinel3_file(original_filename):
    """Renames Sentinel-3 filename to format: S3B_OL_2_<start_time>_<stop_time>.tif"""
    basename = os.path.splitext(original_filename)[0]
    parts = basename.split("_")
    if len(parts) >= 9:
        return f"{parts[0]}_{parts[1]}_{parts[7]}_{parts[8]}.tif"
    logger.error("Filename structure not as expected.")
    return None

def interpolate_geo_coordinates(latitude, longitude, target_shape):
    """Interpolate latitude and longitude tie-point grids to match the target data shape."""
    orig_height, orig_width = latitude.shape
    target_height, target_width = target_shape

    x_orig = np.linspace(0, 1, orig_width)
    y_orig = np.linspace(0, 1, orig_height)
    x_target = np.linspace(0, 1, target_width)
    y_target = np.linspace(0, 1, target_height)

    X_orig, Y_orig = np.meshgrid(x_orig, y_orig)
    X_target, Y_target = np.meshgrid(x_target, y_target)

    points = np.stack([Y_orig.ravel(), X_orig.ravel()], axis=1)
    lat_values = latitude.ravel()
    lon_values = longitude.ravel()

    lat_values = np.where(np.isnan(lat_values), np.nanmean(lat_values), lat_values)
    lon_values = np.where(np.isnan(lon_values), np.nanmean(lon_values), lon_values)

    lat_interp = scipy.interpolate.griddata(points, lat_values, (Y_target, X_target), method='linear', fill_value=np.nan)
    lon_interp = scipy.interpolate.griddata(points, lon_values, (Y_target, X_target), method='linear', fill_value=np.nan)

    return lat_interp, lon_interp

def convert_nc_to_geotiff_with_gcp(nc_path, geo_nc_path, output_name, temp_dir):
    """Convert NetCDF to GeoTIFF with GCPs and a defined affine transform."""
    logger.info(f"Converting NC file: {nc_path}")
    try:
        ds = xr.open_dataset(nc_path)
        ds_geo = xr.open_dataset(geo_nc_path)
        
        data_var = list(ds.data_vars)[0]
        logger.info(f"Data variable: {data_var}")
        data = ds[data_var].isel(time=0) if "time" in ds.dims else ds[data_var]
        data = data.fillna(np.nan)
        logger.info(f"Data shape: {data.shape}")
        
        latitude = ds_geo['latitude'].values
        longitude = ds_geo['longitude'].values
        logger.info(f"Geo coordinates shape: {latitude.shape}")
        
        if data.shape != latitude.shape:
            logger.info(f"Interpolating geo coordinates: {latitude.shape} to {data.shape}")
            latitude, longitude = interpolate_geo_coordinates(latitude, longitude, data.shape)
        
        if data.shape != latitude.shape or data.shape != longitude.shape:
            raise ValueError(f"Dimension mismatch after interpolation: data shape {data.shape}, geo shape {latitude.shape}")
        
        height, width = latitude.shape
        gcps = []
        step = max(1, min(height // 20, width // 20))
        for row in range(0, height, step):
            for col in range(0, width, step):
                lon = longitude[row, col]
                lat = latitude[row, col]
                if not np.isnan(lon) and not np.isnan(lat):
                    gcps.append(GroundControlPoint(col=col, row=row, x=lon, y=lat))
        
        if not gcps:
            raise ValueError("No valid GCPs could be created")
        logger.info(f"Created {len(gcps)} valid GCPs for {output_name}")
        
        lon_min = np.nanmin(longitude)
        lon_max = np.nanmax(longitude)
        lat_min = np.nanmin(latitude)
        lat_max = np.nanmax(latitude)

        pixel_size_x = (lon_max - lon_min) / width
        pixel_size_y = (lat_max - lat_min) / height

        transform = rasterio.transform.Affine(
            pixel_size_x, 0, lon_min,
            0, -pixel_size_y, lat_max
        )
        logger.info(f"Computed transform: {transform}")

        output_path = os.path.join(temp_dir, output_name + ".tif")
        crs = CRS.from_epsg(4326)
        logger.info(f"Output path: {output_path}")
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=str(data.dtype),
            crs=crs,
            nodata=np.nan,
            transform=transform,
            gcps=gcps
        ) as dst:
            dst.write(data.values, 1)
        
        logger.info(f"Successfully wrote GeoTIFF: {output_path}")
        return output_path, transform
    
    except Exception as e:
        logger.error(f"Error processing {nc_path}: {str(e)}")
        return None, None
    finally:
        if 'ds' in locals():
            ds.close()
        if 'ds_geo' in locals():
            ds_geo.close()

def compute_md5(file_path):
    """Compute the MD5 checksum of a file."""
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Error computing MD5 for {file_path}: {str(e)}")
        return "N/A"

def pretty_print_xml(root):
    """Pretty-print XML with indentation."""
    try:
        rough_string = ET.tostring(root, "utf-8")
        parser = ET.XMLParser(encoding="utf-8")
        root = ET.fromstring(rough_string, parser=parser)
        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="utf-8", xml_declaration=True).decode()
    except Exception as e:
        logger.error(f"Error pretty-printing XML: {str(e)}")
        return ""

def get_band_statistics(src, band_idx):
    """Extract statistics for a given band."""
    try:
        band_data = src.read(band_idx)
        valid_data = band_data[~np.isnan(band_data)]
        if valid_data.size == 0:
            return {"min": "N/A", "max": "N/A", "mean": "N/A"}
        return {
            "min": str(valid_data.min()),
            "max": str(valid_data.max()),
            "mean": str(valid_data.mean())
        }
    except Exception as e:
        logger.error(f"Error computing statistics for band {band_idx}: {str(e)}")
        return {"min": "N/A", "max": "N/A", "mean": "N/A"}

def extract_image_time(filename):
    """Extract the image acquisition time from the Sentinel-3 filename."""
    try:
        basename = os.path.splitext(filename)[0]
        parts = basename.split("_")
        if len(parts) >= 4:
            start_time_str = parts[2]
            image_time = datetime.strptime(start_time_str, "%Y%m%dT%H%M%S")
            return image_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            logger.error(f"Filename structure not as expected for {filename}")
            return "N/A"
    except Exception as e:
        logger.error(f"Error extracting image time from {filename}: {str(e)}")
        return "N/A"

def create_manifest(tif_path, temp_dir):
    """Create an XML manifest file for a given GeoTIFF with metadata."""
    logger.info(f"Attempting to process: {tif_path}")
    
    if not os.path.exists(tif_path):
        logger.error(f"Error: File does not exist at {tif_path}")
        return None

    try:
        with rasterio.open(tif_path) as src:
            logger.info(f"Successfully opened GeoTIFF: {tif_path}")
            logger.info(f"GeoTIFF metadata - width: {src.width}, height: {src.height}, count: {src.count}")

            manifest_root = ET.Element("manifest")

            dataset = ET.SubElement(manifest_root, "dataset")
            base_id = os.path.basename(tif_path).replace(".tif", "")
            ET.SubElement(dataset, "id").text = base_id
            ET.SubElement(dataset, "name").text = f"Sentinel-3A OLCI {os.path.basename(tif_path)}"
            ET.SubElement(dataset, "description").text = "GeoTIFF derived from Sentinel-3A OLCI data."
            ET.SubElement(dataset, "version").text = "1.0"
            ET.SubElement(dataset, "image_timestamp").text = extract_image_time(os.path.basename(tif_path))
            upload_time = datetime.now(timezone.utc)
            ET.SubElement(dataset, "upload_timestamp").text = upload_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            ET.SubElement(dataset, "provider").text = "SIG"

            file_elem = ET.SubElement(manifest_root, "file")
            ET.SubElement(file_elem, "path").text = os.path.basename(tif_path)
            ET.SubElement(file_elem, "type").text = "GeoTIFF"
            ET.SubElement(file_elem, "band").text = "stacked"
            ET.SubElement(file_elem, "size").text = str(os.path.getsize(tif_path))
            ET.SubElement(file_elem, "checksum", type="MD5").text = compute_md5(tif_path)

            metadata = ET.SubElement(file_elem, "metadata")
            ET.SubElement(metadata, "width").text = str(src.width)
            ET.SubElement(metadata, "height").text = str(src.height)
            ET.SubElement(metadata, "count").text = str(src.count)
            ET.SubElement(metadata, "crs").text = str(src.crs)
            transform = src.transform
            ET.SubElement(metadata, "transform").text = f"{transform.a}, {transform.b}, {transform.c}, {transform.d}, {transform.e}, {transform.f}, 0.00, 0.00, 1.00"
            bounds = src.bounds
            bounds_elem = ET.SubElement(metadata, "bounds")
            ET.SubElement(bounds_elem, "left").text = str(bounds.left)
            ET.SubElement(bounds_elem, "bottom").text = str(bounds.bottom)
            ET.SubElement(bounds_elem, "right").text = str(bounds.right)
            ET.SubElement(bounds_elem, "top").text = str(bounds.top)
            ET.SubElement(metadata, "driver").text = str(src.driver)
            ET.SubElement(metadata, "data_type").text = str(src.dtypes[0])
            ET.SubElement(metadata, "nodata").text = str(src.nodata) if src.nodata is not None else "nan"

            bands = ET.SubElement(metadata, "bands")
            descriptions = src.descriptions if src.descriptions else []
            for i in range(1, src.count + 1):
                band_name = descriptions[i-1] if i <= len(descriptions) and descriptions[i-1] else f"Band_{i}"
                band_elem = ET.SubElement(bands, "band", id=str(i), name=band_name)
                ET.SubElement(band_elem, "description").text = band_name
                stats = get_band_statistics(src, i)
                stats_elem = ET.SubElement(band_elem, "statistics")
                ET.SubElement(stats_elem, "min").text = stats["min"]
                ET.SubElement(stats_elem, "max").text = stats["max"]
                ET.SubElement(stats_elem, "mean").text = stats["mean"]

            manifest_path = os.path.join(temp_dir, os.path.basename(tif_path).replace(".tif", ".xml"))
            xml_content = pretty_print_xml(manifest_root)
            if not xml_content:
                logger.error("Error: XML content is empty after pretty-printing")
                return None

            with open(manifest_path, "wb") as f:
                f.write(xml_content.encode('utf-8'))
            logger.info(f"Created manifest file: {manifest_path}")
            return manifest_path

    except Exception as e:
        logger.error(f"Error processing {tif_path}: {str(e)}", file=sys.stderr)
        return None

def create_compressed_archive(tif_path, manifest_path, temp_dir):
    """Compress the GeoTIFF and manifest into a .zip archive."""
    try:
        archive_path = os.path.join(temp_dir, os.path.basename(tif_path).replace(".tif", ".zip"))
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(tif_path, os.path.basename(tif_path))
            if manifest_path:
                zipf.write(manifest_path, os.path.basename(manifest_path))
        logger.info(f"Created compressed archive: {archive_path}")
        return archive_path
    except Exception as e:
        logger.error(f"Error creating compressed archive for {tif_path}: {str(e)}")
        return None

def parse_manifest(manifest_path):
    """Parse the manifest XML to extract band names and metadata (adapted to your structure)."""
    try:
        if not os.path.exists(manifest_path) or os.path.getsize(manifest_path) == 0:
            print(f"Manifest file {manifest_path} is missing or empty.")
            return None, None

        tree = ET.parse(manifest_path)
        root = tree.getroot()

        # Extract band names from <bands><band name="...">
        band_names = []
        bands = root.findall('.//bands/band')
        for band in bands:
            band_name = band.attrib.get('name')
            if band_name:
                band_names.append(band_name)

        if not band_names:
            print(f"No band names found in {manifest_path}.")
            return None, None

        # Extract metadata
        metadata = {}

        # Get file_name from <dataset><id>
        dataset_id = root.find('.//dataset/id')
        if dataset_id is not None:
            metadata['file_name'] = dataset_id.text

        # Get timestamps from <dataset>
        for tag in ['upload_timestamp', 'image_timestamp']:
            elem = root.find(f'.//dataset/{tag}')
            if elem is not None:
                metadata[tag] = elem.text

        # Get width/height/crs from <metadata>
        for tag in ['width', 'height', 'crs']:
            elem = root.find(f'.//metadata/{tag}')
            if elem is not None:
                metadata[tag] = int(elem.text) if tag in ['width', 'height'] else elem.text

        return band_names, metadata

    except Exception as e:
        print(f"Error parsing manifest: {e}")
        return None, None

def warp_to_epsg4326(input_path, output_path):
    try:
        subprocess.run([
            "gdalwarp",
            "-t_srs", "EPSG:4326",         # Target CRS
            "-r", "bilinear",              # Resampling method
            "-overwrite",                  # Overwrite if exists
            input_path,
            output_path
        ], check=True)
        print(f"Warped file saved: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error running gdalwarp: {e}")
        return None

# Create a unique subfolder based on timestamp
timestamp_id = datetime.now().strftime("%Y%m%dT%H%M%S")
temp_dir = f"/Users/weraphongsuaruang/Gamuda/s3data/{timestamp_id}"
os.makedirs(temp_dir, exist_ok=True)

# Scaling parameters
SCALE_FACTOR = 10000.0  # e.g., 0.001 becomes 10
OFFSET = 0.0            # Adjust if you want to shift negatives

# --- Local Entry Point ---
if __name__ == "__main__":
    #try:
        # --- Initialize EUMETSAT Connection ---
        credentials = (EUMETSAT_USERNAME, EUMETSAT_PASSWORD)
        token = eumdac.AccessToken(credentials)
        datastore = eumdac.DataStore(token)
        logger.info("Successfully connected to EUMETSAT Data Store")

        # --- Search for Products ---
        logger.info(f"Searching for products from {dtstart} to {dtend} (UTC)")
        products = datastore.get_collection(COLLECTION_ID).search(
            geo=bbox.wkt,
            dtstart=dtstart,
            dtend=dtend
        )
        product_list = list(products)
        logger.info(f"Found {len(product_list)} products in the search range.")

        if not product_list:
            logger.info("No products found")
            sys.exit(0)

        # Process only the first product
        product = product_list[0]
        logger.info(f"Processing product: {product}")

        # Use temporary directory for all local operations
        #with tempfile.TemporaryDirectory() as temp_dir:
        #    logger.info(f"Temporary directory: {temp_dir}")

         # Track uploaded raw files for later deletion
        raw_files_in_bucket = []
        geo_nc_path = None
        temp_files = []

        for filename in product.entries:
            if not (filename.endswith('geo_coordinates.nc') or filename.endswith('tsm_nn.nc') or filename.endswith('chl_nn.nc')):
                continue
        
            print(filename)

            dest_path = os.path.join(temp_dir, os.path.basename(filename))
            temp_files.append(dest_path)


            with product.open(filename) as fsrc, open(dest_path, 'wb') as dst_file:
                logger.info(f"Downloading {filename} to {dest_path}...")
                shutil.copyfileobj(fsrc, dst_file)
                logger.info(f"Saved to {dest_path}")


            stack_filename = rename_sentinel3_file(str(product).replace('/', '_'))
            print("stack_filename",stack_filename)
            stack_path = os.path.join(temp_dir, stack_filename)
            stack_base = os.path.splitext(stack_filename)[0]
            logger.info(f"Stack output path: {stack_path}")
        print(temp_files)

        geo_blob_name = [blob for blob in temp_files if blob.endswith('geo_coordinates.nc')][0]
        geo_temp_path = os.path.join(temp_dir, os.path.basename(geo_blob_name))
        
        print(geo_temp_path )
        band_paths = {}
        first_transform = None
        
        for pattern in TARGET_PATTERNS:
            matched_blobs = [blob for blob in temp_files if fnmatch.fnmatch(os.path.basename(blob), pattern)]
            logger.info(f"Pattern {pattern}: Found {len(matched_blobs)} files - {matched_blobs}")
            if matched_blobs:
                blob_name = matched_blobs[0]
                band_name = pattern.replace('*', '').replace('.nc', '')
                local_path = os.path.join(temp_dir, os.path.basename(blob_name))
                
                tif_path, transform = convert_nc_to_geotiff_with_gcp(local_path, geo_temp_path, f"{stack_base}_{band_name}", temp_dir)
   
                if tif_path:
                    band_paths[band_name] = tif_path
                    if first_transform is None:
                        first_transform = transform
                
        print(band_paths)
        
        valid_bands = [path for band_name, path in band_paths.items() if path is not None]
        logger.info(f"Valid bands to stack: {len(valid_bands)} - {valid_bands}")

        if valid_bands:
            with rasterio.open(valid_bands[0]) as sample:
                gcps = sample.gcps[0] if sample.gcps else []
                crs = sample.crs if sample.crs else CRS.from_epsg(4326)
                height = sample.height
                width = sample.width
                dtype = sample.dtypes[0]

            if not gcps:
                logger.warning("No GCPs found in the first band. Output may not be georeferenced correctly.")

            logger.info(f"Stacking {len(valid_bands)} bands into: {stack_path}")
            with rasterio.open(
                stack_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=len(valid_bands),
                dtype='int16',
                crs=CRS.from_epsg(4326),
                transform=first_transform,
                nodata=-9999,
                gcps=gcps
            ) as dst:
                band_order = [p.replace('*', '').replace('.nc', '') for p in TARGET_PATTERNS]
                for idx, band_name in enumerate(band_order, start=1):
                    if band_paths.get(band_name):
                        with rasterio.open(band_paths[band_name]) as src:
                            data = src.read(1).astype(np.float32)
                            data_scaled = np.round((data - OFFSET) * SCALE_FACTOR)
                            data_scaled = np.where(np.isnan(data_scaled), -9999, data_scaled)
                            data_scaled = data_scaled.astype(np.int16)
                            dst.write(data_scaled, idx)
                            dst.set_band_description(idx, band_name)
                            # Optionally tag with scale metadata per band
                            dst.update_tags(idx, SCALE_FACTOR=str(1 / SCALE_FACTOR), ADD_OFFSET=str(OFFSET), UNITS="mg/m³")

            # Path to warped GeoTIFF
            warped_stack_path = stack_path.replace(".tif", "_warped.tif")

            # Run gdalwarp
            warped_result = warp_to_epsg4326(stack_path, warped_stack_path)

            # If successful, continue with warped file for upload
            if warped_result:
                stack_path = warped_result  # Use this for upload/archive/manifest
            
            logger.info(f"Stacked GeoTIFF with {len(valid_bands)} bands saved: {stack_path}")
        
        
            manifest_path = create_manifest(stack_path, temp_dir)
            
            destination_blob_name = f"processed/{os.path.basename(stack_filename)}"
            upload_success = upload_to_gcs(stack_path, BUCKET_NAME, destination_blob_name)
            if upload_success:
                asset_id = f"projects/servir-sea-landcover/assets/S3WQ/{os.path.splitext(os.path.basename(stack_path))[0]}"
                gcs_uri = f"gs://{BUCKET_NAME}/{destination_blob_name}"
                
                band_names, metadata = parse_manifest(manifest_path)
                

                
               
                if band_names and metadata:
                    asset_request = {
  
                        "id": asset_id,
                        "tilesets": [
                            {
                                "sources": [{"uris": [gcs_uri]}]
                            }
                        ],
                        "bands": [
                            {"id": bname, "tilesetBandIndex": i} for i, bname in enumerate(band_names)
                        ],
                        "properties": metadata,  # optional custom props
                    }
                    #dt = int(metadata['image_timestamp'])/ 1000
                    #iso_time = datetime.utcfromtimestamp(seconds).strftime('%Y-%m-%dT%H:%M:%SZ')
                    #print(iso_time)
                    asset_request["startTime"] = metadata['image_timestamp']

                    
                    task = ee.data.startIngestion(ee.data.newTaskId()[0], asset_request)
                    logger.info(f"Started GEE ingestion task: {task['id']}")
                    shutil.rmtree(temp_dir)
                    
        
        
