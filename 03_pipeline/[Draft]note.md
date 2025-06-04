*** Using .env for EUMETSAT Credential 
- be used by 
from dotenv import load_dotenv
load_dotenv()



*** google cloud config

gcloud auth application-default login

gcloud projects list  

gcloud auth application-default set-quota-project servir-sea-landcover 

gcloud config set project servir-sea-landcover   

gcloud config get-value project 

gcloud auth application-default print-access-token 

---------------------------------------------------------------------------------------------------

*** run script Dual mode 

1. Run in Dynamic Mode (Default):

      - python S3_Pipeline_dual.py

2. Customize the number of days back:

      - python S3_Pipeline_dual.py --mode dynamic --days-back 4

3. Run in Historical Mode, Specify the start and end dates:

      - python S3_Pipeline_dual.py --mode historical --start-date 2024-01-01 --end-date 2024-01-03

      This will process data from January 1, 2024, to January 3, 2024.

test local machine 
python /Users/weraphongsuaruang/Python/S3_OLCI/03_pipeline/S3_Pipeline_dual.py --mode historical --start-date 2024-01-01 --end-date 2024-01-03


