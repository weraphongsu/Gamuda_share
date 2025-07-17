import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, Polygon
from shapely.wkt import loads
import os
import zipfile
from datetime import datetime, timedelta, timezone
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
import argparse
from enum import Enum
import ee
import traceback
import re
from dotenv import load_dotenv
import subprocess
load_dotenv()
username = os.getenv('COPERNICUS_USERNAME')
password = os.getenv('COPERNICUS_PASSWORD')



# Define directories and constants
BASE_DIR = "/Users/weraphongsuaruang/Gamuda/S2_WQ_test"
os.makedirs(BASE_DIR, exist_ok=True)
DOWNLOAD_DIR = os.path.join(BASE_DIR, "s2")
ACOLITE_OUTPUT_DIR = os.path.join(BASE_DIR, "acolite_output")
ADJUSTED_OUTPUT_DIR = os.path.join(BASE_DIR, "adjusted_tifs")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "pipeline_checkpoint.json")

# Set up logging
log_file = os.path.join(BASE_DIR, 'pipeline.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


CANONICAL_BANDS = [
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
def run_acolite_cli(product_name, safe_dir, acolite_output_dir, limit_coords=None):
    logger.info(f"Running ACOLITE for {product_name}")
    product_output_dir = os.path.join(acolite_output_dir, product_name)
    os.makedirs(product_output_dir, exist_ok=True)

    settings_file = os.path.join(acolite_output_dir, f'settings_{product_name}.txt')
    
    # acolite_limit = (
    #     limit_coords
    #     if limit_coords else
    #     "-8.755466801103564, 115.2074921074872, -8.670414997695296, 115.2795234813738"
    # )

    if limit_coords is None:
        logger.error("limit_coords must not be None for ACOLITE processing.")
        raise ValueError("limit_coords must not be None for ACOLITE processing.")

    acolite_limit = limit_coords



    settings_content = f"""# ACOLITE Processing Settings
# Input and Output
inputfile={safe_dir}/
output={product_output_dir}/
# Spatial selection
limit={acolite_limit}
# Atmospheric Correction Method
atmospheric_correction=True
atmospheric_correction_method=dark_spectrum
# L2W Output Parameters (surface reflectance + turbidity)
l2w_parameters=rhow_*,l2_flags,tur_nechad2009_665,spm_nechad2010_665,chlor_a,chl_ci
# L2W options
l2w_turbidity=True
l2w_spm=True
l2w_chlor_a=True
l2w_chl_ci=True
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

            # elif band_name.startswith('rhow_') or band_name.startswith('rhot_') or band_name.startswith('rhos_'):
            # # Scale reflectance by 10000 (no min/max normalization)
            #     scaled[valid_mask] = (data[valid_mask] * 10000).astype(np.int16)


            elif band_name.startswith('rhow_'):
                # Scale reflectance by 10000 (no min/max normalization)
                scaled[valid_mask] = (data[valid_mask] * 10000).astype(np.int16)

            elif band_name == 'tur_nechad2009_665':
                # define nodata for value over 150
                mask = (data >= 0) & (data <= 150)
                scaled[mask] = (data[mask] / 150 * 1000).astype(np.int16)
                scaled[~mask] = NODATA_VALUE
            elif band_name == 'spm_nechad2010_665':
                # define nodata for value over 250
                mask = (data >= 0) & (data <= 250)
                scaled[mask] = (data[mask] / 250 * 1000).astype(np.int16)
                scaled[~mask] = NODATA_VALUE

            # elif band_name == 'tur_nechad2009_665':
            #     scaled [valid_mask] = (np.clip(data[valid_mask], 0, 100) / 100 * 1000).astype(np.int16)
            # elif band_name == 'spm_nechad2010_665':
            #     scaled [valid_mask] = (np.clip(data[valid_mask], 0, 200) / 200 * 1000) .astype(np.int16)

       

            elif band_name in ['chl_oc3', 'chl_ci', 'chl_ocx', 'chlor_a']:
                # scaled[valid_mask] = (data[valid_mask] * 10000).astype(np.int16)
                scaled[valid_mask] = (np.clip(data[valid_mask], 0, 100) / 100 * 10000).astype(np.int16)
            elif band_name == 'kd490':
                scaled[valid_mask] = (data[valid_mask] * 10000).astype(np.int16)
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

ACOLITE_TO_CANONICAL = {
    '443': 'rhow_443',
    '492': 'rhow_492',
    '560': 'rhow_560',
    '559': 'rhow_560',
    '665': 'rhow_665',
    '704': 'rhow_704',
    '740': 'rhow_740',
    '739': 'rhow_740',
    '783': 'rhow_783',
    '780': 'rhow_783',
    '833': 'rhow_833',
    '865': 'rhow_865',
    '864': 'rhow_865',
    '1614': 'rhow_1614',
    '1610': 'rhow_1614',
    '2202': 'rhow_2202',
    '2186': 'rhow_2202',
}



def find_closest_band(canonical_band, tif_files):
    """
    Find the GeoTIFF file whose wavelength is closest to the canonical band.
    Example: canonical_band='rhow_560' will match rhow_559, rhow_561, etc.
    """
    m = re.match(r'(rhow|rhot|rhos)_(\d+)', canonical_band)
    if not m:
        return None
    prefix, target_wl = m.group(1), int(m.group(2))
    candidates = []
    for tif in tif_files:
        base = os.path.basename(tif).split('_L2W_')[-1].replace('.tif', '')
        m2 = re.match(rf'{prefix}_(\d+)', base)
        if m2:
            wl = int(m2.group(1))
            candidates.append((abs(wl - target_wl), wl, tif))
    if candidates:
        # Return tif with closest wavelength
        return sorted(candidates)[0][2]
    return None

def stack_and_reproject_geotiffs(tif_files, adjusted_output_dir, product_name, product_date):
    logger.info(f"Stacking and Reprojecting GeoTIFFs for {product_name}")
    stacked_data = []
    band_info = []
    first_profile = None

    # หา profile และ source_prefix จาก tif แรกที่เจอ (ถ้ามี)
    source_prefix = None
    for tif in tif_files:
        try:
            with rasterio.open(tif) as src:
                first_profile = src.profile.copy()
            if source_prefix is None:
                parts = os.path.basename(tif).split('_')
                source_prefix = '_'.join(parts[:9])
            break
        except Exception:
            continue

    if source_prefix is None:
        source_prefix = product_name

    # >>> Insert here <<<
    if product_date is None:
        try:
            # Try to extract date from source_prefix, e.g. S2B_MSI_2024_01_03_02_07_13_T51NYB
            parts = source_prefix.split('_')
            # Find a pattern that is a date, e.g. 2024_01_03_02_07_13
            for i in range(len(parts) - 6):
                date_part = '_'.join(parts[i:i+6])
                try:
                    product_date = datetime.strptime(date_part, '%Y_%m_%d_%H_%M_%S').strftime('%Y-%m-%d %H:%M:%S')
                    break
                except Exception:
                    continue
            if product_date is None:
                product_date = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            logger.warning(f"Cannot extract product_date from {source_prefix}, using current time")
            product_date = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    # Map band names to files
    band_mapping = {}
    for tif in tif_files:
        base_name = os.path.basename(tif).split('_L2W_')[-1].replace('.tif', '')
        # Map l2_flags, chlor_a, chl_ci, chl_oc3, chl_ocx, ci, kd490 (case-insensitive)
        for b in ['l2_flags', 'chlor_a', 'chl_ci', 'chl_oc3', 'chl_ocx', 'ci', 'kd490']:
            if base_name.lower() == b:
                band_mapping[b] = tif

    # Map reflectance bands by closest wavelength
    for band_name in CANONICAL_BANDS:
        # For rhow/rhot/rhos
        if band_name.startswith(('rhow_', 'rhot_', 'rhos_')) and band_name not in band_mapping:
            tif_file = find_closest_band(band_name, tif_files)
            if tif_file:
                band_mapping[band_name] = tif_file
        # For turbidity (Nechad2009)
        if band_name == 'tur_nechad2009_665' and band_name not in band_mapping:
            # Match any TUR_Nechad2009_xxx
            tif_file = None
            for tif in tif_files:
                base_name = os.path.basename(tif).split('_L2W_')[-1].replace('.tif', '')
                if base_name.startswith('TUR_Nechad2009_'):
                    tif_file = tif
                    break
            # If not found, use closest wavelength
            if not tif_file:
                tif_file = find_closest_band('TUR_Nechad2009_665', tif_files)
            if tif_file:
                band_mapping[band_name] = tif_file
        # For SPM (Nechad2010)
        if band_name == 'spm_nechad2010_665' and band_name not in band_mapping:
            # Match any SPM_Nechad2010_xxx
            tif_file = None
            for tif in tif_files:
                base_name = os.path.basename(tif).split('_L2W_')[-1].replace('.tif', '')
                if base_name.startswith('SPM_Nechad2010_'):
                    tif_file = tif
                    break
            # If not found, use closest wavelength
            if not tif_file:
                tif_file = find_closest_band('SPM_Nechad2010_665', tif_files)
            if tif_file:
                band_mapping[band_name] = tif_file

    available_bands = [band for band in CANONICAL_BANDS if band in band_mapping]
    logger.info(f"Available bands ({len(available_bands)}): {', '.join(available_bands)}")

    # Stack only available bands (do not create NoData band)
    for idx, band_name in enumerate(CANONICAL_BANDS, start=1):
        if band_name in band_mapping:
            tif_file = band_mapping[band_name]
            output_filename = f'adjusted_{os.path.basename(tif_file)}'
            output_path = os.path.join(adjusted_output_dir, output_filename)
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
        # else: ไม่ต้องสร้าง NoData band

    if len(stacked_data) == 0:
        logger.error(f"No bands to stack for {product_name}")
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

    source_prefix = product_name  # Fix: define source_prefix as product_name
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
        'system_time_start': (datetime.strptime(product_date, '%Y-%m-%d %H:%M:%S') + timedelta(hours=1)).isoformat() + 'Z',
        'system_time_end': (datetime.strptime(product_date, '%Y-%m-%d %H:%M:%S') + timedelta(hours=1)).isoformat() + 'Z',
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
    """Upload files to GCS and clean up local files after successful GEE ingestion"""
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
            return False, None, None
    
    if all(uploaded):
        tif_filename = os.path.basename(tif_file)
        xml_filename = os.path.basename(xml_file)

        try:
              # Delete the GeoTIFF and XML files that were just uploaded
            if os.path.exists(tif_file):
                os.remove(tif_file)
            if os.path.exists(xml_file):
                os.remove(xml_file)
              # Delete all adjusted_tifs files related to this product
            for file in os.listdir(ADJUSTED_OUTPUT_DIR):
                if file.startswith('adjusted_') and product_name in file and file.endswith('.tif'):
                    file_path = os.path.join(ADJUSTED_OUTPUT_DIR, file)
                    os.remove(file_path)
                    logger.info(f"Deleted intermediate file: {file_path}")
            # Delete the SAFE folder (downloaded Sentinel-2)
            if os.path.exists(safe_dir):
                shutil.rmtree(safe_dir, ignore_errors=True)
             # Delete the ACOLITE output folder
            if os.path.exists(acolite_output_dir):
                shutil.rmtree(acolite_output_dir, ignore_errors=True)
            # Delete the original Sentinel-2 (.SAFE) file in DOWNLOAD_DIR
            safe_zip = os.path.join(DOWNLOAD_DIR, f"{product_name}.zip")
            if os.path.exists(safe_zip):
                os.remove(safe_zip)
            logger.info(f"Deleted local files and folders for {product_name}")
        except Exception as e:
            logger.error(f"Failed to delete local files: {e}")

        return True, tif_filename, xml_filename
    
    return False, None, None





def parse_manifest(manifest_path, image_id):
    """Parse the manifest XML to extract band names and metadata."""
    try:
        if not os.path.exists(manifest_path):
            logger.error(f"Manifest file {manifest_path} does not exist.")
            return None, None
        if os.path.getsize(manifest_path) == 0:
            logger.error(f"Manifest file {manifest_path} is empty.")
            return None, None

        with open(manifest_path, 'r') as f:
            xml_content = f.read()
        if not xml_content.strip():
            logger.error(f"Manifest file {manifest_path} is empty or contains only whitespace.")
            return None, None

        tree = ET.parse(manifest_path)
        root = tree.getroot()

        band_names = []
        band_metadata = []
        bands_root = root.find('.//bands')
        if bands_root is None:
            logger.error(f"No <bands> section found in {manifest_path}. Raw XML:\n{xml_content}")
            return None, None

        band_elements = bands_root.findall('band') or bands_root.findall('item')
        if not band_elements:
            logger.error(f"No <band> or <item> elements found in {manifest_path}. Raw XML:\n{xml_content}")
            return None, None

        def safe_float(val):
            if val is None or val.strip().lower() == 'null' or val.strip() == '':
                return None
            return float(val)

        for band_elem in band_elements:
            band_name_elem = band_elem.find('band_name')
            if band_name_elem is None or band_name_elem.text is None:
                logger.error(f"Missing or empty band_name in {manifest_path}. Raw XML:\n{xml_content}")
                return None, None
            band_names.append(band_name_elem.text)

            band_info = {
                'band_id': band_elem.find('band_id').text if band_elem.find('band_id') is not None else None,
                'nodata': band_elem.find('nodata').text if band_elem.find('nodata') is not None else None,
                'status': band_elem.find('status').text if band_elem.find('status') is not None else None
            }
            range_elem = band_elem.find('actual_range') or band_elem.find('adjusted_range')
            if range_elem is not None:
                min_elem = range_elem.find('min')
                max_elem = range_elem.find('max')
                band_info['range_min'] = safe_float(min_elem.text) if min_elem is not None else None
                band_info['range_max'] = safe_float(max_elem.text) if max_elem is not None else None
            scaled_range = band_elem.find('scaled_range')
            if scaled_range is not None:
                min_elem = scaled_range.find('min')
                max_elem = scaled_range.find('max')
                band_info['scaled_range_min'] = safe_float(min_elem.text) if min_elem is not None else None
                band_info['scaled_range_max'] = safe_float(max_elem.text) if max_elem is not None else None
            scale_factor = band_elem.find('scale_factor')
            if scale_factor is not None:
                band_info['scale_factor'] = safe_float(scale_factor.text) if scale_factor.text else None
            statistics = band_elem.find('statistics')
            if statistics is not None:
                min_stat = statistics.find('min')
                max_stat = statistics.find('max')
                mean_stat = statistics.find('mean')
                std_stat = statistics.find('std')
                band_info['stats_min'] = safe_float(min_stat.text) if min_stat is not None else None
                band_info['stats_max'] = safe_float(max_stat.text) if max_stat is not None else None
                band_info['stats_mean'] = safe_float(mean_stat.text) if mean_stat is not None else None
                band_info['stats_std'] = safe_float(std_stat.text) if std_stat is not None else None
            band_metadata.append(band_info)

        metadata = {}
        for field in ['file_name', 'processing_date', 'product_date', 'crs', 'width', 'height', 'system_time_start', 'system_time_end']:
            elem = root.find(field)
            if elem is None or elem.text is None:
                logger.error(f"Missing or empty {field} in {manifest_path}. Raw XML:\n{xml_content}")
                return None, None
            metadata[field] = elem.text if field not in ['width', 'height'] else int(elem.text)

        for i, band_info in enumerate(band_metadata):
            for key, value in band_info.items():
                if value is not None:
                    metadata[f"band_{i+1}_{key}"] = value

        parts = image_id.split('_')
        if len(parts) >= 7:
            metadata['system_time_start'] = f"{parts[2]}-{parts[3]}-{parts[4]}T00:00:00"

        return band_names, metadata
    except ET.ParseError as e:
        logger.error(f"XML parsing error in {manifest_path}: {e}. Raw XML:\n{xml_content}")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error parsing {manifest_path}: {e}. Raw XML:\n{xml_content}")
        logger.error(traceback.format_exc())
        return None, None





def iso_to_unix_ms(iso_str):
    dt = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%SZ")
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

def extract_band_names_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    band_names = []
    for band_item in root.find('bands').findall('item'):
        band_name = band_item.find('band_name').text
        band_names.append(band_name)
    return band_names

def ingest_to_gee_api(asset_id, gcs_uri, system_time_start, band_names):
    manifest = {
        "id": asset_id,
        "tilesets": [{
            "sources": [{
                "uris": [gcs_uri]
            }]
        }],
        "bands": [{"id": name, "tilesetBandIndex": i} for i, name in enumerate(band_names)],
        "properties": {},
        "startTime": system_time_start
    }
    task_id = ee.data.startIngestion(ee.data.newTaskId()[0], manifest)
    print(f"Started ingestion task: {task_id}")
    return task_id


def main():

    # Authenticate and initialize Earth Engine
    try:
        ee.Initialize()
    except Exception as e:
        logger.error("Earth Engine authentication required. Please run: earthengine authenticate")
        ee.Authenticate()
        ee.Initialize(project='servir-ee')
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Sentinel-2 Processing Pipeline')
    parser.add_argument('--mode', choices=['historical', 'near-realtime'], required=True,
                       help='Processing mode: historical or near-realtime')
    parser.add_argument('--start-date', required=False, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=False, help='End date (YYYY-MM-DD)')
    parser.add_argument('--cloud-cover', type=int, default=30,
                       help='Maximum cloud cover percentage (default: 30)')
    args = parser.parse_args()

    # Auto-set dates for near-realtime mode
    if args.mode == 'near-realtime':
        today = datetime.utcnow().date()
        args.end_date = today.strftime('%Y-%m-%d')
        args.start_date = (today - timedelta(days=7)).strftime('%Y-%m-%d')
        logger.info(f"Auto-set date range for near-realtime: {args.start_date} to {args.end_date}")
    elif not args.start_date or not args.end_date:
        logger.error("For historical mode, --start-date and --end-date are required.")
        return

    # Create necessary directories
    for directory in [DOWNLOAD_DIR, ACOLITE_OUTPUT_DIR, ADJUSTED_OUTPUT_DIR]:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

    # Load credentials from .env file
    load_dotenv()
    username = os.getenv('COPERNICUS_USERNAME')
    password = os.getenv('COPERNICUS_PASSWORD')
    bucket_name = "sentinel2acolite"
    destination_folder = "test"

    # Authentication
    try:
        access_token = get_access_token(username, password)
        logger.info("Authentication successful")
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        return

    # Load AOI

    ## using Buffer Center Point
    center_lon, center_lat =  100.250516, 5.359031
    buffer_deg = 14 / 111.0

    bbox = Polygon([
    (center_lon - buffer_deg, center_lat - buffer_deg),
    (center_lon + buffer_deg, center_lat - buffer_deg),
    (center_lon + buffer_deg, center_lat + buffer_deg),
    (center_lon - buffer_deg, center_lat + buffer_deg)
    ])

    gdf = gpd.GeoDataFrame({'id': [1]}, geometry=[bbox], crs="EPSG:4326")
    upper_right = (center_lon + buffer_deg, center_lat + buffer_deg)
    lower_left = (center_lon - buffer_deg, center_lat - buffer_deg)
    limit_coords = f'{lower_left[1]}, {lower_left[0]}, {upper_right[1]}, {upper_right[0]}'
    # Convert to WKT for pipeline use
    aoi_wkt = gdf.geometry.iloc[0].wkt

    try:
        aoi_wkt = load_aoi(aoi_wkt)
    except Exception as e:
        logger.error(f"Failed to load AOI: {e}")
        return

    # Query products
    products_df = query_sentinel2(aoi_wkt, args.start_date, args.end_date, args.cloud_cover)
    if products_df is None or len(products_df) == 0:
        logger.error("No products found matching criteria")
        return

    

    # Load checkpoint
    processed_products = load_checkpoint().get("processed_products", [])

    # Process each product
    for _, product in products_df.iterrows():
        product_name = product['Name'].split('.')[0]
        
        # Skip if already processed
        if product_name in processed_products:
            logger.info(f"Skipping already processed product: {product_name}")
            continue

        # Download and process
        access_token = download_and_extract_product(product, access_token, username, password)
        if access_token is None:
            continue

        safe_dir = os.path.join(DOWNLOAD_DIR, f"{product_name}.SAFE")
        acolite_output = run_acolite_cli(product_name, safe_dir, ACOLITE_OUTPUT_DIR,limit_coords)
        if acolite_output is None:
            continue

        # Get list of ACOLITE output GeoTIFFs
        tif_files = []
        for root, _, files in os.walk(acolite_output):
            tif_files.extend([os.path.join(root, f) for f in files if f.endswith('.tif')])

        # ### Stack and reproject
        logger.info(f"Skipping adjust/stack step for {product_name}. Raw ACOLITE outputs are available at {acolite_output}")
        stacked_tif, manifest_xml, product_date = stack_and_reproject_geotiffs(
            tif_files, ADJUSTED_OUTPUT_DIR, product_name, None
        )
        if stacked_tif is None:
            continue

        # read system_time_start และ band_names from manifest_xml before deleting it
        tree = ET.parse(manifest_xml)
        root = tree.getroot()
        system_time_start_iso = root.find('system_time_start').text
        system_time_start = system_time_start_iso 

        band_names, _ = parse_manifest(manifest_xml, product_name)
        if band_names is None:
            logger.error(f"Failed to parse band names from {manifest_xml}")
            continue

        success, tif_filename, xml_filename = upload_to_gcs_and_cleanup(
            bucket_name, ADJUSTED_OUTPUT_DIR, destination_folder,
            product_name, stacked_tif, manifest_xml,
            safe_dir, acolite_output
        )

        if success:
            asset_id = f"projects/servir-ee/assets/sentinel2Acolite/{os.path.splitext(tif_filename)[0]}"
            gcs_uri = f"gs://{bucket_name}/{destination_folder}/{tif_filename}"

            # no more read XML again, use the one we just uploaded
            ingest_to_gee_api(asset_id, gcs_uri, system_time_start, band_names)
            processed_products.append(product_name)
            save_checkpoint(processed_products)
            logger.info(f"Successfully processed and uploaded {product_name} to GCS")
        else:
            logger.error(f"Failed to upload {product_name} to GCS")

   

if __name__ == "__main__":
    main()





