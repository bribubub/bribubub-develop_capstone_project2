import os
import boto3
from botocore.client import Config
from botocore.exceptions import NoCredentialsError, ClientError

S3_ENDPOINT = "https://lqyjvssuwnidxebfrrlc.supabase.co/storage/v1/s3"
ACCESS_KEY = "0de326ba854b5a5cbafbc7140bc0627c"
SECRET_KEY = "410da4c221a179f7aab9d5c83d076891c7703d1b009dbc350c2e154013754510"
BUCKET_NAME = "dataset"

# Region name in Supabase S3 is often "us-east-1" regardless of actual region, or omitted
s3_client = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name='ap-southeast-1',
    config=Config(s3={'addressing_style': 'path'}, signature_version='s3v4')
)

def upload_folder_to_s3(local_dir, prefix):
    print(f"🚀 Memulai proses upload dari folder: {local_dir}")
    if not os.path.exists(local_dir):
        print(f"❌ Error: Folder {local_dir} tidak ditemukan!")
        return
    files_to_upload = [f for f in os.listdir(local_dir) if os.path.isfile(os.path.join(local_dir, f))]
    if not files_to_upload:
        return
    for file_name in files_to_upload:
        local_path = os.path.join(local_dir, file_name)
        s3_path = f"{prefix}/{file_name}"
        content_type = "image/png" if file_name.lower().endswith(".png") else "image/jpeg"
        print(f"☁️  Mengupload {file_name} ke s3://{BUCKET_NAME}/{s3_path} ...")
        try:
            s3_client.upload_file(local_path, BUCKET_NAME, s3_path, ExtraArgs={'ContentType': content_type})
            print(f"   ✅ Berhasil!")
        except Exception as e:
            print(f"   ❌ Error S3: {e}")

if __name__ == "__main__":
    upload_folder_to_s3(os.path.join("dataset", "ricarda"), "ricarda")
