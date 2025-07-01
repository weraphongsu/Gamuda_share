import ee
import xml.etree.ElementTree as ET
from google.cloud import storage
import os
import traceback
import datetime

def parse_manifest(manifest_path, image_id):
    """Parse the manifest XML to extract band names and metadata."""
    try:
        # Check if file exists and is not empty
        if not os.path.exists(manifest_path):
            print(f"Manifest file {manifest_path} does not exist.")
            return None, None
        if os.path.getsize(manifest_path) == 0:
            print(f"Manifest file {manifest_path} is empty.")
            return None, None
        
        # Read XML content
        with open(manifest_path, 'r') as f:
            xml_content = f.read()
        if not xml_content.strip():
            print(f"Manifest file {manifest_path} is empty or contains only whitespace.")
            return None, None
        
        # Parse XML
        tree = ET.parse(manifest_path)
        root = tree.getroot()
        
        # Extract band names and metadata
        band_names = []
        band_metadata = []
        bands_root = root.find('.//bands')
        if bands_root is None:
            print(f"No <bands> section found in {manifest_path}. Raw XML:\n{xml_content}")
            return None, None
        
        # Try <band> or <item> elements
        band_elements = bands_root.findall('band') or bands_root.findall('item')
        if not band_elements:
            print(f"No <band> or <item> elements found in {manifest_path}. Raw XML:\n{xml_content}")
            return None, None
        
        structure = 'band' if bands_root.findall('band') else 'item'
        print(f"Detected XML structure: <bands><{structure}> in {manifest_path}")
        
        def safe_float(val):
            if val is None or val.strip().lower() == 'null' or val.strip() == '':
                return None
            return float(val)

        for band_elem in band_elements:
            band_name_elem = band_elem.find('band_name')
            if band_name_elem is None or band_name_elem.text is None:
                print(f"Missing or empty band_name in {manifest_path}. Raw XML:\n{xml_content}")
                return None, None
            band_names.append(band_name_elem.text)
            
            # Extract band metadata
            band_info = {
                'band_id': band_elem.find('band_id').text if band_elem.find('band_id') is not None else None,
                'nodata': band_elem.find('nodata').text if band_elem.find('nodata') is not None else None,
                'status': band_elem.find('status').text if band_elem.find('status') is not None else None
            }
            # Extract range (actual_range or adjusted_range)
            range_elem = band_elem.find('actual_range') or band_elem.find('adjusted_range')
            if range_elem is not None:
                min_elem = range_elem.find('min')
                max_elem = range_elem.find('max')
                band_info['range_min'] = safe_float(min_elem.text) if min_elem is not None else None
                band_info['range_max'] = safe_float(max_elem.text) if max_elem is not None else None
            # Extract scaled_range
            scaled_range = band_elem.find('scaled_range')
            if scaled_range is not None:
                min_elem = scaled_range.find('min')
                max_elem = scaled_range.find('max')
                band_info['scaled_range_min'] = safe_float(min_elem.text) if min_elem is not None else None
                band_info['scaled_range_max'] = safe_float(max_elem.text) if max_elem is not None else None
            # Extract scale_factor
            scale_factor = band_elem.find('scale_factor')
            if scale_factor is not None:
                band_info['scale_factor'] = safe_float(scale_factor.text) if scale_factor.text else None
            # Extract statistics
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
        
        # Extract general metadata, including system_start_time and system_end_time
        metadata = {}
        for field in ['file_name', 'processing_date', 'product_date', 'crs', 'width', 'height', 'system_start_time', 'system_end_time']:
            elem = root.find(field)
            if elem is None or elem.text is None:
                print(f"Missing or empty {field} in {manifest_path}. Raw XML:\n{xml_content}")
                return None, None
            metadata[field] = elem.text if field not in ['width', 'height'] else int(elem.text)

        # Set GEE system time properties if present
        if 'system_start_time' in metadata:
            metadata['system_time_start'] = metadata['system_start_time']
        if 'system_end_time' in metadata:
            metadata['system_time_end'] = metadata['system_end_time']

        # Optionally remove the original fields if you only want GEE keys
        metadata.pop('system_start_time', None)
        metadata.pop('system_end_time', None)

        # Add band metadata to general metadata for GEE properties
        for i, band_info in enumerate(band_metadata):
            for key, value in band_info.items():
                if value is not None:
                    metadata[f"band_{i+1}_{key}"] = value
        
        # Extract date from filename (assuming format: S2A_MSI_2018_02_01_03_56_46_T47NPF)
        parts = image_id.split('_')
        if len(parts) >= 7:
            # Set as ISO 8601 string, e.g., 2018-02-01T00:00:00
            metadata['system_time_start'] = f"{parts[2]}-{parts[3]}-{parts[4]}T00:00:00"
        
        # Convert system_start_time to ISO 8601 format if present
        system_start_time_elem = root.find('system_start_time')
        if system_start_time_elem is not None and system_start_time_elem.text:
            # Format: '2019-01-14 03:31:01'
            dt = datetime.datetime.strptime(system_start_time_elem.text, "%Y-%m-%d %H:%M:%S")
            # Use ISO 8601 string for GEE property
            metadata['system_time_start'] = dt.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            # fallback: use product_date or filename
            pass
        
        return band_names, metadata
    except ET.ParseError as e:
        print(f"XML parsing error in {manifest_path}: {e}. Raw XML:\n{xml_content}")
        return None, None
    except Exception as e:
        print(f"Unexpected error parsing {manifest_path}: {e}. Raw XML:\n{xml_content}")
        print(traceback.format_exc())
        return None, None

def upload_to_gee(bucket_name, source_folder, asset_collection, project_name):
    """Ingest TIFF files from GCS to GEE Image Collection."""
    # Initialize GEE
    try:
        ee.Initialize(project=project_name)
    except Exception as e:
        print(f"Failed to initialize GEE: {e}")
        return
    
    # Initialize GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # List blobs in the GCS folder
    blobs = bucket.list_blobs(prefix=source_folder)
    
    for blob in blobs:
        if blob.name.endswith('.tif'):
            # Extract file name and image ID
            file_name = os.path.basename(blob.name)
            image_id = os.path.splitext(file_name)[0]
            full_asset_id = f"{asset_collection}/{image_id}"
            
            # Find corresponding manifest XML
            manifest_blob_name = blob.name.replace('.tif', '.xml')
            manifest_blob = bucket.blob(manifest_blob_name)
            
            # Check if manifest exists
            if not manifest_blob.exists():
                print(f"Manifest {manifest_blob_name} not found in bucket for {file_name}. Skipping.")
                continue
            
            # Download manifest temporarily
            temp_manifest_path = f"/tmp/{os.path.basename(manifest_blob_name)}"
            try:
                manifest_blob.download_to_filename(temp_manifest_path)
            except Exception as e:
                print(f"Error downloading manifest {manifest_blob_name}: {e}")
                continue
            
            # Parse band names and metadata from manifest
            band_names, metadata = parse_manifest(temp_manifest_path, image_id)
            if not band_names or not metadata:
                print(f"Failed to parse manifest for {file_name}. Skipping.")
                try:
                    os.remove(temp_manifest_path)
                except Exception as e:
                    print(f"Error removing {temp_manifest_path}: {e}")
                continue
            
            # Construct GEE ingestion request
            asset_request = {
                'id': full_asset_id,
                'tilesets': [
                    {
                        'sources': [
                            {
                                'uris': [f'gs://{bucket_name}/{blob.name}']
                            }
                        ]
                    }
                ],
                'bands': [
                    {'id': band_name, 'tilesetBandIndex': idx}
                    for idx, band_name in enumerate(band_names)
                ],
                'properties': metadata
            }
            
            # Check if asset already exists
            try:
                ee.data.getAsset(full_asset_id)
                print(f"Asset {full_asset_id} already exists, skipping.")
                os.remove(temp_manifest_path)
                continue
            except ee.EEException:
                # Asset does not exist, proceed with ingestion
                pass
            
            # Start ingestion task
            try:
                task = ee.data.startIngestion(ee.data.newTaskId()[0], asset_request)
                print(f"Started ingestion for {image_id} with task ID: {task['id']}")
            except ee.EEException as e:
                print(f"Ingestion failed for {image_id}: {e}")
            except Exception as e:
                print(f"Unexpected error during ingestion for {image_id}: {e}")
            
            # Clean up temporary manifest file
            try:
                os.remove(temp_manifest_path)
            except Exception as e:
                print(f"Error removing {temp_manifest_path}: {e}")

if __name__ == "__main__":
    # Configuration
    bucket_name = "s2_bucket1"
    source_folder = "acolite_testSacled"
    asset_collection = "projects/servir-sea-landcover/assets/S2Acolite_scaled"
    project_name = "servir-sea-landcover"
    
    # Ingest files to GEE
    upload_to_gee(bucket_name, source_folder, asset_collection, project_name)