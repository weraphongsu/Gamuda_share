import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from shapely.wkt import loads
import os
import zipfile
from datetime import datetime, timedelta
import time
import subprocess
import rasterio
import numpy as np
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import shutil
from rasterio.warp import calculate_default_transform, reproject, Resampling
from google.cloud import storage
import json
import logging
import math

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define directories and constants
BASE_DIR = "/Users/weraphongsuaruang/Gamuda/S2Acolite_ProcessingScale"
DOWNLOAD_DIR = os.path.join(BASE_DIR, "s2")
ACOLITE_OUTPUT_DIR = os.path.join(BASE_DIR, "acolite_output")
ADJUSTED_OUTPUT_DIR = os.path.join(BASE_DIR, "adjusted_tifs")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "pipeline_checkpoint.json")

CANONICAL_BANDS = [
    'rhow_443', 'rhow_492', 'rhow_560', 'rhow_665', 'rhow_704', 'rhow_740', 'rhow_783', 'rhow_833',
    'rhow_865', 'rhow_1614', 'rhow_2202', 'l2_flags', 'tur_nechad2009_665', 'spm_nechad2010_665',
    'chl_oc3', 'ci', 'chl_ci', 'chl_ocx', 'chlor_a', 'kd490'
]

NODATA_VALUE = -9999  # Use this for all bands, including l2_flags
NODATA_FLAGS = 65535  # For l2_flags (uint16)

# Checkpoint handling
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"processed_products": []}

def save_checkpoint(processed_products):
    checkpoint = {"processed_products": processed_products}
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    logger.info(f"Saved checkpoint: {processed_products}")

# 1. Authentication
def get_access_token(username: str, password: str) -> str:
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    try:
        response = requests.post(url, data=data)
        response.raise_for_status()
        return response.json()["access_token"]
    except Exception as e:
        logger.error(f"Failed to get access token: {e}")
        raise


# 2. Load AOI
def load_aoi(aoi_input: str):
    logger.info("Loading Area of Interest (AOI)")
    if os.path.isfile(aoi_input) and aoi_input.endswith('.geojson'):
        try:
            gdf = gpd.read_file(aoi_input)
            if len(gdf) == 0:
                raise ValueError("GeoJSON file is empty.")
            aoi_geometry = gdf.geometry.iloc[0]
            logger.info(f"Loaded AOI from GeoJSON: {aoi_input}")
        except Exception as e:
            logger.error(f"Failed to read GeoJSON file: {e}")
            raise
    else:
        try:
            aoi_geometry = loads(aoi_input)
            logger.info("Loaded AOI from WKT string")
        except Exception as e:
            logger.error(f"Failed to parse WKT: {e}")
            raise
    return aoi_geometry.wkt

# 3. Query Sentinel-2 Products
def query_sentinel2(aoi_wkt: str, start_date: str, end_date: str, max_cloud_cover: int):
    data_collection = "SENTINEL-2"
    logger.info(f"Querying Sentinel-2 Level-1C Products: {start_date} to {end_date}, Cloud cover <= {max_cloud_cover}%")
    url = (
        f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
        f"$filter=Collection/Name eq '{data_collection}' and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi_wkt}') and "
        f"ContentDate/Start gt {start_date}T00:00:00.000Z and "
        f"ContentDate/Start lt {end_date}T00:00:00.000Z and "
        f"Attributes/OData.CSC.StringAttribute/any(s:s/Name eq 'productType' and s/Value eq 'S2MSI1C') and "
        f"Attributes/OData.CSC.DoubleAttribute/any(d:d/Name eq 'cloudCover' and d/Value le {max_cloud_cover})&$count=True&$top=1000"
    )
    try:
        json_data = requests.get(url).json()
        logger.info("Query successful")
        if "value" in json_data and len(json_data["value"]) > 0:
            df = pd.DataFrame.from_dict(json_data["value"])
            df["geometry"] = df["GeoFootprint"].apply(lambda geo: shape(geo) if geo else None)
            productDF = gpd.GeoDataFrame(df[df["geometry"].notnull()]).set_geometry("geometry")
            logger.info(f"Found {len(productDF)} Level-1C product(s)")
            return productDF
        else:
            logger.warning(f"No Level-1C products found with cloud cover <= {max_cloud_cover}%.")
            return None
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise

# 4. Download and Extract Single Product
def download_and_extract_product(product, access_token, username, password):
    logger.info(f"Downloading product: {product['Name']}")
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {access_token}"})
    product_id = product["Id"]
    product_name = product["Name"].split(".")[0]
    download_url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    save_path = os.path.join(DOWNLOAD_DIR, f"{product_name}.zip")
    
    try:
        response = session.get(download_url, allow_redirects=False)
        while response.status_code in (301, 302, 303, 307):
            download_url = response.headers["Location"]
            response = session.get(download_url, allow_redirects=False)
        
        file_response = session.get(download_url, allow_redirects=True)
        if file_response.status_code == 401:
            logger.warning("Token expired, refreshing...")
            access_token = get_access_token(username, password)
            session.headers.update({"Authorization": f"Bearer {access_token}"})
            logger.info("Token refreshed successfully")
            file_response = session.get(download_url, allow_redirects=True)
        
        file_response.raise_for_status()
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        with open(save_path, "wb") as file:
            file.write(file_response.content)
        logger.info(f"Downloaded: {save_path}")
        
        logger.info(f"Extracting: {product_name}.zip")
        try:
            with zipfile.ZipFile(save_path, 'r') as zip_ref:
                zip_ref.extractall(DOWNLOAD_DIR)
            logger.info(f"Extracted: {product_name}.zip to {DOWNLOAD_DIR}")
            os.remove(save_path)
            logger.info(f"Deleted zip file: {save_path}")
            return access_token
        except zipfile.BadZipFile:
            logger.error(f"Failed to extract {product_name}.zip: Corrupted or invalid ZIP file.")
            os.remove(save_path)
            return None
        except Exception as e:
            logger.error(f"Failed to extract {product_name}.zip: {e}")
            os.remove(save_path)
            return None
    except Exception as e:
        logger.error(f"Download failed for {product_name}: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return None

# 5. ACOLITE Processing
def run_acolite_cli(product_name, safe_dir, acolite_output_dir):
    logger.info(f"Running ACOLITE for {product_name}")
    product_output_dir = os.path.join(acolite_output_dir, product_name)
    os.makedirs(product_output_dir, exist_ok=True)
    
    settings_file = os.path.join(acolite_output_dir, f'settings_{product_name}.txt')
    settings_content = f"""# ACOLITE Processing Settings
# Input and Output
inputfile={safe_dir}/
output={product_output_dir}/
# Spatial selection
limit=5.218863396361375,100.14971030414435,5.493738415679082,100.3627274844965
# Atmospheric Correction Method
atmospheric_correction=True
atmospheric_correction_method=dark_spectrum
# L2W Output Parameters (surface reflectance + turbidity)
l2w_parameters=rhow_*,tur_nechad2009_665,spm_nechad2010_665,chlor_a,chl_ci,kd490
# L2W options
l2w_mask=False
l2w_mask_water_parameters=True
l2w_export_geotiff=True
# General Settings
verbosity=2
s2_target_res=10
# Optional PNG/Map Outputs
map_l2w=True
map_png=True
map_auto_range=True
"""
    try:
        with open(settings_file, 'w') as f:
            f.write(settings_content)
        logger.info(f"Updated ACOLITE settings file: {settings_file}")
    except Exception as e:
        logger.error(f"Failed to write settings file: {e}")
        return None
    
    acolite_cmd = ['python', '/Users/weraphongsuaruang/Python/acolite/launch_acolite.py', '--cli', '--settings', settings_file]
    try:
        result = subprocess.run(acolite_cmd, capture_output=True, text=True, check=True)
        logger.info("ACOLITE processing successful")
        logger.debug(f"ACOLITE stdout: {result.stdout}")
        return product_output_dir
    except subprocess.CalledProcessError as e:
        logger.error(f"ACOLITE processing failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return None

# 6. Create NoData Band
def create_nodata_band(profile, band_name):
    if profile is None:
        logger.error("Cannot create NoData band without a reference profile")
        return None, None
    width, height = profile['width'], profile['height']
    dtype = np.int16
    nodata = NODATA_VALUE
    nodata_data = np.full((height, width), nodata, dtype=dtype)
    stats = {'min': None, 'max': None, 'mean': None, 'std': None}
    logger.info(f"Created NoData band for {band_name} (dtype: {dtype}, nodata: {nodata})")
    return nodata_data, stats

# 7. Adjust and Stack GeoTIFFs with log10 scaling for turbidity and SPM
def adjust_geotiff_range(input_path, output_path, band_name):
    logger.info(f"Processing GeoTIFF: {input_path} (band: {band_name})")
    try:
        

        with rasterio.open(input_path) as src:
            data = src.read(1)
            profile = src.profile.copy()
            nodata = -9999
            profile.update(nodata=nodata, dtype=rasterio.int16)

            # Mask invalid/negative values
            valid_mask = (data != nodata) & (data >= 0)

            scaled = np.full_like(data, NODATA_VALUE, dtype=np.int16)

            if band_name == 'l2_flags':
                # No scaling for l2_flags, but convert to int16
                scaled[valid_mask] = data[valid_mask].astype(np.int16)
            elif band_name.startswith('rhow_'):
                # Scale reflectance by 10000 (no min/max normalization)
                scaled[valid_mask] = (data[valid_mask] * 10000).astype(np.int16)
            elif band_name == 'tur_nechad2009_665':
                scaled[valid_mask] = (np.clip(data[valid_mask], 0, 100) / 100 * 10000).astype(np.int16)
            elif band_name == 'spm_nechad2010_665':
                scaled[valid_mask] = (np.clip(data[valid_mask], 0, 200) / 200 * 10000).astype(np.int16)
            elif band_name in ['chl_oc3', 'chl_ci', 'chl_ocx', 'chlor_a']:
                scaled[valid_mask] = (np.clip(data[valid_mask], 0, 50) / 50 * 10000).astype(np.int16)
            elif band_name == 'kd490':
                scaled[valid_mask] = (np.clip(data[valid_mask], 0, 5) / 5 * 10000).astype(np.int16)
            else:
                if valid_mask.sum() > 0:
                    dmin = np.min(data[valid_mask])
                    dmax = np.max(data[valid_mask])
                    if dmax > dmin:
                        scaled[valid_mask] = ((data[valid_mask] - dmin) / (dmax - dmin) * 10000).astype(np.int16)
                # else already filled with nodata


        

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(scaled, 1)
            logger.info(f"Saved adjusted file: {output_path} (dtype: {profile['dtype']}, nodata: {nodata})")

            valid_data = scaled[scaled != nodata]
            stats = {
                'min': int(np.min(valid_data)) if valid_data.size > 0 else None,
                'max': int(np.max(valid_data)) if valid_data.size > 0 else None,
                'mean': float(np.mean(valid_data)) if valid_data.size > 0 else None,
                'std': float(np.std(valid_data)) if valid_data.size > 0 else None
            }
            return scaled, profile, stats
    except Exception as e:
        logger.error(f"Failed to process {input_path}: {e}")
        return None, None, None

def dict_to_xml(tag, d):
    elem = ET.Element(tag)
    if isinstance(d, dict):
        for key, val in d.items():
            if isinstance(val, dict):
                child = dict_to_xml(key, val)
                elem.append(child)
            elif isinstance(val, list):
                child = ET.Element(key)
                for sub_item in val:
                    if isinstance(sub_item, dict):
                        sub_child = dict_to_xml('item', sub_item)
                    else:
                        sub_child = ET.Element('item')
                        sub_child.text = str(sub_item) if sub_item is not None else 'null'
                    child.append(sub_child)
                elem.append(child)
            else:
                child = ET.Element(key)
                child.text = str(val) if val is not None else 'null'
                elem.append(child)
    elif isinstance(d, list):
        for item in d:
            child = dict_to_xml('item', item)
            elem.append(child)
    else:
        elem.text = str(d) if d is not None else 'null'
    return elem

# 8. Stack and Reproject GeoTIFFs

def stack_and_reproject_geotiffs(tif_files, adjusted_output_dir, product_name, product_date):
    logger.info(f"Stacking and Reprojecting GeoTIFFs for {product_name}")
    stacked_data = []
    band_info = []
    first_profile = None
    source_prefix = None

    # Map band names to files, handling slight naming variations
    band_mapping = {}
    for tif in tif_files:
        base_name = os.path.basename(tif).split('_L2W_')[-1].replace('.tif', '')
        if base_name.startswith('rhow_'):
            wavelength = int(float(base_name.split('rhow_')[1]))  # Extract numeric part
            canonical = next((b for b in CANONICAL_BANDS if b.startswith('rhow_') and abs(int(float(b.split('rhow_')[1])) - wavelength) <= 5), None)
            if canonical:
                band_mapping[canonical] = tif
        elif base_name in ['l2_flags', 'chl_oc3', 'ci', 'chl_ci', 'chl_ocx', 'chlor_a', 'kd490']:
            band_mapping[base_name] = tif
        elif base_name == 'TUR_Nechad2009_665':
            band_mapping['tur_nechad2009_665'] = tif
        elif base_name == 'SPM_Nechad2010_665':
            band_mapping['spm_nechad2010_665'] = tif

    available_bands = [band for band in CANONICAL_BANDS if band in band_mapping]
    logger.info(f"Available bands ({len(available_bands)}): {', '.join(available_bands)}")
    if len(available_bands) < len(CANONICAL_BANDS):
        missing_bands = [band for band in CANONICAL_BANDS if band not in available_bands]
        logger.warning(f"Expected {len(CANONICAL_BANDS)} bands, found {len(available_bands)}. Missing: {', '.join(missing_bands)}")

    # Process all canonical bands
    for idx, band_name in enumerate(CANONICAL_BANDS, start=1):
        if band_name in band_mapping:
            tif_file = band_mapping[band_name]
            output_filename = f'adjusted_{os.path.basename(tif_file)}'
            output_path = os.path.join(adjusted_output_dir, output_filename)
            
            if source_prefix is None:
                parts = os.path.basename(tif_file).split('_')
                source_prefix = '_'.join(parts[:9])
                if not product_date:
                    date_part = parts[2] + '_' + parts[3] + '_' + parts[4] + '_' + parts[5] + '_' + parts[6] + '_' + parts[7]
                    try:
                        product_date = datetime.strptime(date_part, '%Y_%m_%d_%H_%M_%S').strftime('%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        logger.warning(f"Invalid date format in {tif_file}: {date_part}")
                        product_date = "Unknown"

            data, profile, stats = adjust_geotiff_range(tif_file, output_path, band_name)
            if data is None or profile is None or stats is None:
                logger.warning(f"Failed to process band {band_name}, creating NoData band")
                data, stats = create_nodata_band(first_profile, band_name)
        else:
            logger.info(f"Band {band_name} missing, creating NoData band")
            data, stats = create_nodata_band(first_profile, band_name)
        
        if data is None:
            logger.error(f"Failed to create data for band {band_name}, skipping product")
            return None, None, None
        
        band_info.append({
            'band_id': idx,
            'band_name': band_name,
            'actual_range': {'min': stats['min'], 'max': stats['max']},
            'statistics': stats,
            'dtype': str(data.dtype),
            'nodata': NODATA_VALUE,
            'status': 'processed' if band_name in band_mapping else 'nodata'
        })
        stacked_data.append(data)
        if first_profile is None and profile is not None:
            first_profile = profile

    if len(stacked_data) != len(CANONICAL_BANDS):  # 20 bands
        logger.error(f"Expected {len(CANONICAL_BANDS)} bands, got {len(stacked_data)}. Aborting stack for {product_name}")
        return None, None, None

    try:
        stacked_data = np.stack(stacked_data, axis=0)
        logger.info(f"Stacked {len(CANONICAL_BANDS)} bands")
    except ValueError as e:
        logger.error(f"Failed to stack GeoTIFFs: {e}")
        return None, None, None

    with rasterio.open(tif_files[0]) as src:
        transform = src.transform
        crs = src.crs
        logger.info(f"Source CRS: {crs}")
        if crs != 'EPSG:4326':
            logger.info("Reprojecting to EPSG:4326...")
            dst_crs = 'EPSG:4326'
            height, width = stacked_data.shape[1], stacked_data.shape[2]
            dst_transform, dst_width, dst_height = calculate_default_transform(
                crs, dst_crs, width, height, *src.bounds
            )
            dtype = np.uint16  # Updated to always use uint16
            nodata = 65535    # Uniform nodata
            dst_data = np.full((stacked_data.shape[0], dst_height, dst_width), nodata, dtype=dtype)
            reproject(
                stacked_data,
                dst_data,
                src_transform=transform,
                src_crs=crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest
            )
            stacked_data = dst_data
            first_profile.update(transform=dst_transform, crs=dst_crs, width=dst_width, height=dst_height, dtype=dtype, nodata=nodata)
            logger.info(f"Reprojected to EPSG:4326: new dimensions ({dst_height}, {dst_width})")
        else:
            logger.info("Source CRS is already EPSG:4326, skipping reprojection")
            dtype = np.uint16  # Updated to always use uint16
            first_profile.update(dtype=dtype, nodata=nodata)

    temp_output_path = os.path.join(adjusted_output_dir, f'{source_prefix}_temp.tif')
    cog_profile = first_profile.copy()
    cog_profile.update(count=len(stacked_data), dtype=stacked_data.dtype, nodata=nodata)
    with rasterio.open(temp_output_path, 'w', **cog_profile) as dst:
        dst.write(stacked_data)
        for i, band in enumerate(band_info, start=1):
            dst.set_band_description(i, f"{band['band_id']}_{band['band_name']}")
    logger.info(f"Saved temporary GeoTIFF: {temp_output_path}")
    
    stacked_output_path = os.path.join(adjusted_output_dir, f'{source_prefix}.tif')
    gdal_cmd = [
        'gdal_translate', '-of', 'COG', '-co', 'TILED=YES',
        '-co', 'BLOCKXSIZE=256', '-co', 'BLOCKYSIZE=256',
        '-co', 'OVERVIEW_RESAMPLING=NEAREST', '-co', 'ADD_ALPHA=NO',
        temp_output_path, stacked_output_path
    ]
    try:
        subprocess.run(gdal_cmd, check=True, capture_output=True, text=True)
        logger.info(f"Converted to Cloud-Optimized GeoTIFF: {stacked_output_path}")
        os.remove(temp_output_path)
        logger.info(f"Deleted temporary file: {temp_output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to convert to COG: {e}")
        return None, None, None

    # Add system time in ISO format for GEE compatibility
    current_time = datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'  # 06:14 AM UTC, June 11, 2025
    manifest = {
        'file_name': os.path.basename(stacked_output_path),
        'processing_date': current_time,
        'product_date': product_date,
        'system_start_time': product_date,
        'system_end_time': (datetime.strptime(product_date, '%Y-%m-%d %H:%M:%S') + timedelta(hours=1)).isoformat() + 'Z',
        'crs': str(cog_profile['crs']),
        'width': cog_profile['width'],
        'height': cog_profile['height'],
        'band_count': len(CANONICAL_BANDS),
        'bands': band_info,
        'available_bands': available_bands,
        'missing_bands': [band for band in CANONICAL_BANDS if band not in available_bands]
    }
    
    manifest_file = os.path.join(adjusted_output_dir, f'{source_prefix}.xml')
    try:
        root = dict_to_xml('metadata', manifest)
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        with open(manifest_file, 'w') as f:
            f.write(xml_str)
        logger.info(f"Saved metadata manifest (XML): {manifest_file}")
    except Exception as e:
        logger.error(f"Failed to save manifest: {e}")
        return None, None, None
    
    return stacked_output_path, manifest_file, product_date








# 8. Upload to GCS and Clean Up
def upload_to_gcs_and_cleanup(bucket_name, source_folder, destination_folder, product_name, tif_file, xml_file, safe_dir, acolite_output_dir):

    # logger.info(f"(MUTED) Skipping upload for {product_name}")

    logger.info(f"Uploading files for {product_name} to Google Cloud Storage")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    files_to_upload = [tif_file, xml_file]
    uploaded = []
    
    for file_path in files_to_upload:
        filename = os.path.basename(file_path)
        destination_blob_name = f"{destination_folder}/{filename}"
        blob = bucket.blob(destination_blob_name)
        try:
            blob.upload_from_filename(file_path)
            logger.info(f"Uploaded {filename} to {destination_blob_name}")
            uploaded.append(file_path)
        except Exception as e:
            logger.error(f"Failed to upload {filename}: {e}")
            return False
    
    logger.info(f"Cleaning up source files for {product_name}")
    try:
        if os.path.exists(safe_dir):
            shutil.rmtree(safe_dir, ignore_errors=True)
            logger.info(f"Deleted .SAFE directory: {safe_dir}")
        
        if os.path.exists(acolite_output_dir):
            shutil.rmtree(acolite_output_dir, ignore_errors=True)
            logger.info(f"Deleted ACOLITE output directory: {acolite_output_dir}")
        
        for file_path in files_to_upload:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted local file: {file_path}")
        
        for file in os.listdir(ADJUSTED_OUTPUT_DIR):
            if file.startswith('adjusted_') and file.endswith('.tif'):
                file_path = os.path.join(ADJUSTED_OUTPUT_DIR, file)
                os.remove(file_path)
                logger.info(f"Deleted intermediate file: {file_path}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to clean up source files: {e}")
        return False
    # return True  # Mute upload for testing purposes

# Main Pipeline
def main():
    username = 'wsuaruang@sig-gis.com'
    password = 'yw-~838NbD7~+uc'
    aoi_input = "POLYGON ((100.11616186486486 5.249911864864865, 100.38643213513512 5.249911864864865, 100.38643213513512 5.520182135135135, 100.11616186486486 5.520182135135135, 100.11616186486486 5.249911864864865))"
    year = 2025
    start_date = f"{year}-03-01"
    end_date = f"{year}-06-30"
    max_cloud_cover = 30
    bucket_name = "s2_bucket1"
    destination_folder = f"acolite_{year}"
    
    for directory in [DOWNLOAD_DIR, ACOLITE_OUTPUT_DIR, ADJUSTED_OUTPUT_DIR]:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    checkpoint = load_checkpoint()
    processed_products = checkpoint.get("processed_products", [])
    
    try:
        access_token = get_access_token(username, password)
        logger.info("Authentication successful")
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        return
    
    try:
        footprint = load_aoi(aoi_input)
        logger.info(f"AOI WKT: {footprint}")
    except Exception as e:
        logger.error(f"AOI loading failed: {e}")
        return
    
    try:
        productDF = query_sentinel2(footprint, start_date, end_date, max_cloud_cover)
        if productDF is None:
            logger.error("No products found, exiting pipeline")
            return
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return
    
    for _, product in productDF.iterrows():
        product_name = product["Name"].split(".")[0]
        if product_name in processed_products:
            logger.info(f"Skipping already processed product: {product_name}")
            continue
        
        access_token = download_and_extract_product(product, access_token, username, password)
        if access_token is None:
            logger.error(f"Skipping {product_name} due to download/extraction failure")
            continue
        
        safe_dir = os.path.join(DOWNLOAD_DIR, f"{product_name}.SAFE")
        if not os.path.exists(safe_dir):
            logger.error(f".SAFE directory not found for {product_name}")
            continue
        
        acolite_output = run_acolite_cli(product_name, safe_dir, ACOLITE_OUTPUT_DIR)
        if acolite_output is None:
            logger.error(f"Skipping {product_name} due to ACOLITE failure")
            shutil.rmtree(safe_dir, ignore_errors=True)
            continue
        
        tif_files = [os.path.join(acolite_output, f) for f in os.listdir(acolite_output) if f.endswith('.tif')]
        if not tif_files:
            logger.error(f"No GeoTIFF files found in {acolite_output}")
            shutil.rmtree(safe_dir, ignore_errors=True)
            shutil.rmtree(acolite_output, ignore_errors=True)
            continue
        
        # Extract product date from metadata if available, otherwise infer
        product_date = None
        try:
            with open(os.path.join(safe_dir, 'MTD_MSIL1C.xml'), 'r') as f:
                import xml.etree.ElementTree as ET
                tree = ET.parse(f)
                root = tree.getroot()
                date_str = root.find('.//PRODUCT_START_TIME').text if root.find('.//PRODUCT_START_TIME') is not None else None
                if date_str:
                    product_date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%fZ').strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            logger.warning(f"Failed to extract product date from metadata: {e}")

        tif_file, xml_file, _ = stack_and_reproject_geotiffs(tif_files, ADJUSTED_OUTPUT_DIR, product_name, product_date)
        if tif_file is None or xml_file is None:
            logger.error(f"Skipping {product_name} due to stacking/reprojection failure")
            shutil.rmtree(safe_dir, ignore_errors=True)
            shutil.rmtree(acolite_output, ignore_errors=True)
            continue
        
        success = upload_to_gcs_and_cleanup(
            bucket_name, ADJUSTED_OUTPUT_DIR, destination_folder, product_name,
            tif_file, xml_file, safe_dir, acolite_output
        )
        if success:
            processed_products.append(product_name)
            save_checkpoint(processed_products)
        else:
            logger.error(f"Upload or cleanup failed for {product_name}, retaining source files for retry")
            continue
    
    try:
        os.remove(CHECKPOINT_FILE)
        logger.info("Deleted checkpoint file")
    except Exception as e:
        logger.error(f"Failed to delete checkpoint file: {e}")
    
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    logger.info("Starting Sentinel-2 Processing Pipeline")
    try:
        main()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.info("Pipeline can resume by rerunning the script")