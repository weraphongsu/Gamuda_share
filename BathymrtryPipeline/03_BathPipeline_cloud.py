import os
import io
import re
import zipfile
import time
import json
import argparse
import tempfile
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Point, Polygon
from rasterio.transform import from_origin
from scipy.interpolate import griddata
from pyproj import Transformer, Proj
from google.cloud import storage
import ee
import shutil


def extract_date_range(header_lines):
    from datetime import datetime
    for line in header_lines:
        if '-' in line and '/' in line:
            try:
                d1_str, d2_str = line.split('-')
                d1 = datetime.strptime(d1_str.strip(), "%d/%m/%Y")
                d2 = datetime.strptime(d2_str.strip(), "%d/%m/%Y")
                return f"{d1.strftime('%Y%m%d')}_{d2.strftime('%Y%m%d')}", d1, d2
            except:
                pass
    return "unknown", None, None

def download_zip_from_gcs(bucket_name, blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    zip_bytes = blob.download_as_bytes()
    return io.BytesIO(zip_bytes)

def process_txt_zip(zip_stream, island_name):
    transformer = Transformer.from_proj(Proj('epsg:3382'), Proj('epsg:4326'))
    all_dfs = []
    date_token = None

    with zipfile.ZipFile(zip_stream) as zf:
        for name in zf.namelist():
            if not name.endswith('.txt'):
                continue
            with zf.open(name) as f:
                content = f.read()
            encodings = ['utf-8', 'utf-16', 'ISO-8859-1']
            for enc in encodings:
                try:
                    text = content.decode(enc)
                    lines = text.splitlines()
                    header_lines = lines[:8]
                    if not date_token:
                        date_token, d1, d2 = extract_date_range(header_lines)
                    df = pd.read_csv(io.StringIO('\n'.join(lines[8:])), sep=r'\s+', names=['Offset','Easting','Northing','Vertical'])
                    df = df.dropna()
                    lon, lat = transformer.transform(df['Easting'].values, df['Northing'].values)
                    df['Latitude'] = lat
                    df['Longitude'] = lon
                    df['SourceFile'] = name
                    all_dfs.append(df)
                    break
                except:
                    continue

    if not all_dfs:
        raise ValueError("No .txt data could be read from zip.")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df, date_token, d1, d2

# def interpolate_and_export(df, aoi_gdf, grid_res, crs_epsg, output_path):
#     gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Easting'], df['Northing']), crs=f"EPSG:{crs_epsg}")
#     gdf_clip = gdf[gdf.within(aoi_gdf.unary_union)]
#     if gdf_clip.empty:
#         raise ValueError("No point within AOI")

#     x, y, z = gdf_clip['Easting'].values, gdf_clip['Northing'].values, gdf_clip['Vertical'].values
#     xi = np.arange(x.min(), x.max(), grid_res)
#     yi = np.arange(y.min(), y.max(), grid_res)
#     xi, yi = np.meshgrid(xi, yi)
#     zi = griddata((x, y), z, (xi, yi), method='linear')
#     zi_flipped = np.flipud(zi)
#     transform = from_origin(xi.min(), yi.max(), grid_res, grid_res)

#     with rasterio.open(
#         output_path,
#         'w', driver='GTiff', height=zi_flipped.shape[0], width=zi_flipped.shape[1],
#         count=1, dtype=zi_flipped.dtype, crs=f"EPSG:{crs_epsg}", transform=transform, nodata=np.nan
#     ) as dst:
#         dst.write(zi_flipped, 1)
#     return output_path

def interpolate_and_export(df, aoi_gdf, grid_res, crs_epsg, output_path):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Easting'], df['Northing']), crs=f"EPSG:{crs_epsg}")
    
    # Reproject AOI if needed
    if aoi_gdf.crs.to_epsg() != crs_epsg:
        aoi_gdf = aoi_gdf.to_crs(epsg=crs_epsg)
    
    # Clip
    gdf_clip = gdf[gdf.within(aoi_gdf.unary_union)]
    if gdf_clip.empty:
        raise ValueError("No point within AOI")

    # Interpolation
    x, y, z = gdf_clip['Easting'].values, gdf_clip['Northing'].values, gdf_clip['Vertical'].values
    xi = np.arange(x.min(), x.max(), grid_res)
    yi = np.arange(y.min(), y.max(), grid_res)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method='linear')
    zi_flipped = np.flipud(zi)
    transform = from_origin(xi.min(), yi.max(), grid_res, grid_res)

    # Write GeoTIFF
    with rasterio.open(
        output_path,
        'w', driver='GTiff', height=zi_flipped.shape[0], width=zi_flipped.shape[1],
        count=1, dtype=zi_flipped.dtype, crs=f"EPSG:{crs_epsg}", transform=transform, nodata=np.nan
    ) as dst:
        dst.write(zi_flipped, 1)
    return output_path



def reproject_tif_to_4326(src_path, dst_path):
    # Quote file paths to handle spaces
    os.system(f'gdalwarp -t_srs EPSG:4326 -r bilinear -overwrite "{src_path}" "{dst_path}"')

def upload_to_gcs(local_path, bucket_name, dest_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_blob_name)
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{dest_blob_name}"

def upload_to_gee(gs_uri, asset_id, start_dt, end_dt, project):
    ee.Initialize(project=project)
    start_sec = int(start_dt.timestamp()) if start_dt else None
    end_sec = int(end_dt.timestamp()) if end_dt else None

    manifest = {
        "name": asset_id,
        "tilesets": [{"sources": [{"uris": [gs_uri]}]}],
        "bands": [{"id": "b1"}],
    }
    if start_sec: manifest["start_time"] = {"seconds": start_sec}
    if end_sec: manifest["end_time"] = {"seconds": end_sec}

    task = ee.data.startIngestion(ee.data.newTaskId()[0], manifest)
    task_id = task['name']

    while True:
        status = ee.data.getOperation(task_id)
        state = status.get('metadata', {}).get('state')
        print(f"Task {task_id}: {state}")
        if state == 'SUCCEEDED':
            break
        elif state in ('FAILED', 'CANCELLED'):
            raise RuntimeError(f"Ingestion failed: {status}")
        time.sleep(15)

# === ENTRY POINT ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_bucket", required=True)
    parser.add_argument("--zip_blob", required=True)
    parser.add_argument("--output_bucket", required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--aoi_name", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--gee_asset_path", required=True)
    args = parser.parse_args()

    temp_dir = tempfile.mkdtemp()
    try:
        zip_stream = download_zip_from_gcs(args.input_bucket, args.zip_blob)
        df, date_token, d1, d2 = process_txt_zip(zip_stream, args.aoi_name)

        # Load AOI
        ee.Initialize(project=args.project)
        aoi_fc = ee.FeatureCollection("projects/servir-ee/assets/penang_mudflat_aoiBuf350")
        aoi = aoi_fc.filter(ee.Filter.eq("Name", args.aoi_name))
        aoi_geojson = aoi.getInfo()
        aoi_geom = aoi_geojson['features'][0]['geometry']
        # aoi_gdf = gpd.GeoDataFrame.from_features([{'geometry': aoi_geom}], crs='EPSG:4326').to_crs(epsg=3382)
        aoi_gdf = gpd.GeoDataFrame.from_features(
        [{'type': 'Feature', 'geometry': aoi_geom, 'properties': {}}],
        crs='EPSG:4326').to_crs(epsg=3382)

        # Get the zip name without extension
        zip_base = os.path.splitext(os.path.basename(args.zip_blob))[0]
        # cretae output paths
        raw_tif_path = os.path.join(
            temp_dir,
            f"{args.aoi_name.replace(' ', '_')}_{zip_base}_{date_token}.tif"
        )
        tif_4326_path = raw_tif_path.replace('.tif', '_4326.tif')

        interpolate_and_export(df, aoi_gdf, grid_res=5, crs_epsg=3382, output_path=raw_tif_path)
        reproject_tif_to_4326(raw_tif_path, tif_4326_path)

        # Upload to output bucket
        blob_name = f"{args.output_prefix}/{os.path.basename(tif_4326_path)}"
        gs_uri = upload_to_gcs(tif_4326_path, args.output_bucket, blob_name)

        # Upload to GEE
        asset_id = f"{args.gee_asset_path}/{os.path.splitext(os.path.basename(tif_4326_path))[0]}"
        upload_to_gee(gs_uri, asset_id, d1, d2, args.project)

        print("Done.")
    finally:
        shutil.rmtree(temp_dir)
