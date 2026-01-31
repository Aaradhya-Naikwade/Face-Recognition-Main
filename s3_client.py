# import os
# import uuid
# import boto3
# from botocore.exceptions import ClientError

# BUCKET_NAME = "haltn"
# BASE_FOLDER = "KlickShare"

# s3 = boto3.client(
#     "s3",
#     region_name=os.getenv("AWS_REGION", "ap-south-1")
# )

# def upload_image_to_s3(
#     image_bytes: bytes,
#     group_id: str,
#     filename: str
# ) -> str:
#     ext = filename.split(".")[-1].lower()
#     key = f"{BASE_FOLDER}/groups/{group_id}/{uuid.uuid4().hex}.{ext}"

#     try:
#         s3.put_object(
#             Bucket=BUCKET_NAME,
#             Key=key,
#             Body=image_bytes,
#             ContentType=f"image/{ext}"
#         )
#     except ClientError as e:
#         raise RuntimeError(f"S3 upload failed: {e}")

#     return f"https://{BUCKET_NAME}.s3.amazonaws.com/{key}"






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

# --------------------------------------------------
# UPLOAD IMAGE (EVENT → GROUP → IMAGE)
# --------------------------------------------------
def upload_image_to_s3(
    image_bytes: bytes,
    event_id: str,
    group_id: str,
    filename: str
) -> str:
    ext = filename.split(".")[-1].lower()

    key = (
        f"{BASE_FOLDER}/events/{event_id}/groups/"
        f"{group_id}/{uuid.uuid4().hex}.{ext}"
    )

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


# --------------------------------------------------
# DELETE ALL IMAGES OF A GROUP
# --------------------------------------------------
def delete_group_from_s3(event_id: str, group_id: str):
    prefix = f"{BASE_FOLDER}/events/{event_id}/groups/{group_id}/"

    try:
        response = s3.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=prefix
        )

        if "Contents" not in response:
            return

        objects = [{"Key": obj["Key"]} for obj in response["Contents"]]

        s3.delete_objects(
            Bucket=BUCKET_NAME,
            Delete={"Objects": objects}
        )

    except ClientError as e:
        raise RuntimeError(f"S3 group delete failed: {e}")


# --------------------------------------------------
# DELETE ENTIRE EVENT (ALL GROUPS + IMAGES)
# --------------------------------------------------
def delete_event_from_s3(event_id: str):
    prefix = f"{BASE_FOLDER}/events/{event_id}/"

    try:
        response = s3.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=prefix
        )

        if "Contents" not in response:
            return

        objects = [{"Key": obj["Key"]} for obj in response["Contents"]]

        s3.delete_objects(
            Bucket=BUCKET_NAME,
            Delete={"Objects": objects}
        )

    except ClientError as e:
        raise RuntimeError(f"S3 event delete failed: {e}")
