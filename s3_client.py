import os
import uuid
import boto3
from botocore.exceptions import ClientError

BUCKET_NAME = "haltn"
BASE_FOLDER = "KlickShare"

s3 = boto3.client(
    "s3",
    region_name=os.getenv("AWS_REGION", "ap-south-1")
)

def upload_image_to_s3(
    image_bytes: bytes,
    group_id: str,
    filename: str
) -> str:
    ext = filename.split(".")[-1].lower()
    key = f"{BASE_FOLDER}/groups/{group_id}/{uuid.uuid4().hex}.{ext}"

    try:
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=key,
            Body=image_bytes,
            ContentType=f"image/{ext}"
        )
    except ClientError as e:
        raise RuntimeError(f"S3 upload failed: {e}")

    return f"https://{BUCKET_NAME}.s3.amazonaws.com/{key}"
