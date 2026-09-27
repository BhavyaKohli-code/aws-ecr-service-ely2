"""Write a Bedrock Knowledge Base .metadata.json next to every document, from app/ely/knowledge/taxonomy.json.

    python tools/build_metadata.py              # dry run: show what each document would be tagged
    python tools/build_metadata.py --apply      # write the .metadata.json files to S3
    python tools/build_metadata.py --sync       # start ingestion for every data source of both knowledge bases

Needs AWS credentials with s3:ListBucket/GetObject/PutObject on both buckets and
bedrock:StartIngestionJob/GetIngestionJob on the knowledge bases.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "ely"))
from knowledge.taxonomy import document_attributes, excluded_data_sources  # noqa: E402

REGION = "ap-south-1"
SOURCES = [  # (bucket, prefix, knowledge base id)
    ("ely2-test-sales-data", "documents/", "HOLR19M08V"),
    ("elydocuments", "HRIS/", "1SPV0BK1YF"),
]
METADATA_SUFFIX = ".metadata.json"


def documents(s3, bucket, prefix):
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("/") and not key.endswith(METADATA_SUFFIX):
                yield key


def metadata_body(attrs: dict) -> str:
    return json.dumps({"metadataAttributes": {k: v for k, v in attrs.items()}}, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write the metadata files to S3")
    parser.add_argument("--sync", action="store_true", help="ingest every data source (after writing, if --apply)")
    parser.add_argument("--sync-only", action="store_true", help="only run the ingestion, write nothing")
    args = parser.parse_args()
    if args.sync_only:
        return sync(boto3.client("bedrock-agent", region_name=REGION))
    s3 = boto3.client("s3", region_name=REGION)

    summary, untagged = Counter(), []
    for bucket, prefix, _ in SOURCES:
        for key in documents(s3, bucket, prefix):
            attrs = document_attributes(bucket, key)
            if not attrs["category"]:
                untagged.append(f"{bucket}/{key}")
            summary[(bucket, attrs["category"], attrs["duplicate"])] += 1
            if args.apply:
                s3.put_object(Bucket=bucket, Key=key + METADATA_SUFFIX, Body=metadata_body(attrs).encode("utf-8"),
                              ContentType="application/json")
            else:
                print(f"{bucket}/{key}\n    -> {attrs}")

    print("\nSummary (bucket, category, duplicate): count")
    for k, n in sorted(summary.items()):
        print(f"  {k}: {n}")
    if untagged:
        print(f"\nUNTAGGED ({len(untagged)}):", *untagged[:20], sep="\n  ")
    if not args.apply:
        print("\nDry run — nothing written. Re-run with --apply to write.")

    if args.sync:
        sync(boto3.client("bedrock-agent", region_name=REGION))


def sync(agent):
    """Ingest each data source in turn (a knowledge base runs one ingestion job at a time)."""
    for _, _, kb in SOURCES:
        for ds in agent.list_data_sources(knowledgeBaseId=kb)["dataSourceSummaries"]:
            if ds["dataSourceId"] in excluded_data_sources():
                continue
            while any(j["status"] in ("STARTING", "IN_PROGRESS") for j in agent.list_ingestion_jobs(
                    knowledgeBaseId=kb, dataSourceId=ds["dataSourceId"], maxResults=5)["ingestionJobSummaries"]) or _busy(agent, kb):
                time.sleep(20)
            job = agent.start_ingestion_job(knowledgeBaseId=kb, dataSourceId=ds["dataSourceId"])["ingestionJob"]
            print(f"ingestion started: KB {kb} / {ds['name']} -> job {job['ingestionJobId']}", flush=True)
            while True:
                job = agent.get_ingestion_job(knowledgeBaseId=kb, dataSourceId=ds["dataSourceId"],
                                              ingestionJobId=job["ingestionJobId"])["ingestionJob"]
                if job["status"] not in ("STARTING", "IN_PROGRESS"):
                    print(f"  {job['status']}: {job.get('statistics')}", flush=True)
                    break
                time.sleep(20)


def _busy(agent, kb):
    for ds in agent.list_data_sources(knowledgeBaseId=kb)["dataSourceSummaries"]:
        jobs = agent.list_ingestion_jobs(knowledgeBaseId=kb, dataSourceId=ds["dataSourceId"], maxResults=5)
        if any(j["status"] in ("STARTING", "IN_PROGRESS") for j in jobs["ingestionJobSummaries"]):
            return True
    return False


if __name__ == "__main__":
    main()
