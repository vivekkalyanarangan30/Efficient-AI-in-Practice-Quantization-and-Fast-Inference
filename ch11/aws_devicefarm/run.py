"""Drive an AWS Device Farm Test Run for the ch11-bench APK end-to-end.

Reads AWS credentials from the standard env/profile chain (AWS_PROFILE,
AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, or ~/.aws/credentials). The script
does not accept credentials as arguments or print them — keep secrets out of
shell history and process listings.

Usage:
  .venv/bin/python aws_devicefarm/run.py \\
      --project-arn arn:aws:devicefarm:us-west-2:<acct>:project:<uuid> \\
      --device "Google Pixel 9"

Pipeline:
  1) Resolve / create a single-device pool that matches --device by name.
  2) Upload app-debug.apk (ANDROID_APP) and app-debug-androidTest.apk
     (INSTRUMENTATION_TEST_PACKAGE), waiting for each to finish processing.
  3) Schedule an INSTRUMENTATION run, poll until completion.
  4) Locate the "Customer Artifacts" file under all jobs/suites/tests and
     download it under android/runs/<run-id>/customer-artifacts.zip.
  5) Print the next-step ingest command.

This script does not modify results.json. After it finishes you run
`ch11_4_android.py unpack-artifacts --zip <path>` followed by
`ch11_4_android.py ingest-apk-results --input <staged-path>`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

HERE = Path(__file__).resolve().parent.parent
APP_APK = HERE / "android" / "app" / "build" / "outputs" / "apk" / "debug" / "app-debug.apk"
TEST_APK = HERE / "android" / "app" / "build" / "outputs" / "apk" / "androidTest" / "debug" / "app-debug-androidTest.apk"
RUNS_DIR = HERE / "android" / "runs"

# Device Farm processes uploads asynchronously; poll up to this long.
UPLOAD_TIMEOUT_S = 600
UPLOAD_POLL_S = 5
RUN_TIMEOUT_S = 120 * 60  # LLM matrix + vision + sustained/power ~ 75 min on-device + host overhead.
RUN_POLL_S = 30

# Artifact type names returned by list_artifacts for the Customer Artifacts zip.
CUSTOMER_ARTIFACT_NAMES = {"Customer Artifacts", "customer-artifacts.zip"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project-arn", required=True,
                   help="arn:aws:devicefarm:us-west-2:<acct>:project:<uuid>")
    p.add_argument("--device", default="Google Pixel 9",
                   help="device model name to match (case-insensitive substring)")
    p.add_argument("--region", default="us-west-2", help="AWS region")
    p.add_argument("--run-name", default=None,
                   help="run name (default: timestamp). letters/digits/_-")
    p.add_argument("--no-poll", action="store_true",
                   help="kick off the run and exit; do not wait for completion")
    p.add_argument("--extra-data", default=None,
                   help="optional path to a ZIP uploaded as EXTERNAL_DATA "
                        "(e.g. the Llama .task model). The testspec pre_test "
                        "phase pushes its .task contents to the app's scoped "
                        "external dir on-device before the instrumentation run.")
    p.add_argument("--testspec", default=None,
                   help="optional path to a custom testspec YAML uploaded as "
                        "INSTRUMENTATION_TEST_SPEC. Defaults to "
                        "aws_devicefarm/testspec.yml when present.")
    args = p.parse_args()

    if not APP_APK.is_file():
        print(f"error: app APK missing at {APP_APK}", file=sys.stderr)
        print("Run: cd android && ./gradlew assembleDebug assembleDebugAndroidTest",
              file=sys.stderr)
        return 1
    if not TEST_APK.is_file():
        print(f"error: test APK missing at {TEST_APK}", file=sys.stderr)
        return 1

    # Project ARN is supplied by the user; sanity-check format to avoid command
    # injection / typo surprises. Pattern is documented and bounded in length.
    if not re.fullmatch(r"arn:aws:devicefarm:[a-z0-9\-]+:\d{12}:project:[a-f0-9\-]{36}",
                        args.project_arn):
        print("error: --project-arn does not look like a Device Farm project ARN",
              file=sys.stderr)
        return 1
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _\-]{0,63}", args.device):
        print("error: --device contains unexpected characters", file=sys.stderr)
        return 1

    run_name = args.run_name or f"ch11-android-{int(time.time())}"
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", run_name):
        print("error: --run-name must be A-Za-z0-9_- (max 64)", file=sys.stderr)
        return 1

    try:
        client = boto3.client("devicefarm", region_name=args.region)
    except (BotoCoreError, ClientError) as exc:
        print(f"error: failed to init Device Farm client: {exc}", file=sys.stderr)
        return 1

    print(f"=== ch11 Device Farm runner ===")
    print(f"  project_arn : {args.project_arn}")
    print(f"  region      : {args.region}")
    print(f"  device match: {args.device!r}")
    print(f"  run_name    : {run_name}")

    device_arn = _resolve_device_arn(client, args.device)
    if not device_arn:
        print(f"error: no device matched {args.device!r}", file=sys.stderr)
        return 1
    print(f"  device_arn  : {device_arn}")

    pool_arn = _create_device_pool(client, args.project_arn, run_name, device_arn)
    print(f"  pool_arn    : {pool_arn}")

    print("uploading app APK…")
    app_arn = _upload(client, args.project_arn, APP_APK, "ANDROID_APP")
    print(f"  app_arn     : {app_arn}")

    print("uploading test APK…")
    test_arn = _upload(client, args.project_arn, TEST_APK, "INSTRUMENTATION_TEST_PACKAGE")
    print(f"  test_arn    : {test_arn}")

    extra_arn: str | None = None
    if args.extra_data:
        extra_path = Path(args.extra_data).expanduser().resolve()
        if not extra_path.is_file():
            print(f"error: --extra-data file not found: {extra_path}", file=sys.stderr)
            return 1
        # Guard against accidentally uploading > 2 GiB (Device Farm cap).
        size_mb = extra_path.stat().st_size / (1024 * 1024)
        if size_mb > 2048:
            print(f"error: --extra-data is {size_mb:.0f} MiB, exceeds 2 GiB cap",
                  file=sys.stderr)
            return 1
        print(f"uploading extra-data ({size_mb:.0f} MiB)…")
        extra_arn = _upload(client, args.project_arn, extra_path, "EXTERNAL_DATA")
        print(f"  extra_arn   : {extra_arn}")

    # Default testspec lives next to this script; allow override via flag.
    spec_arn: str | None = None
    spec_path: Path | None = None
    if args.testspec:
        spec_path = Path(args.testspec).expanduser().resolve()
    else:
        default = Path(__file__).resolve().parent / "testspec.yml"
        if default.is_file():
            spec_path = default
    if spec_path is not None:
        if not spec_path.is_file():
            print(f"error: --testspec file not found: {spec_path}", file=sys.stderr)
            return 1
        print(f"uploading testspec ({spec_path.name})…")
        spec_arn = _upload(client, args.project_arn, spec_path, "INSTRUMENTATION_TEST_SPEC")
        print(f"  spec_arn    : {spec_arn}")

    print("scheduling run…")
    run_arn = _schedule_run(client, args.project_arn, pool_arn, app_arn, test_arn, run_name,
                            extra_arn=extra_arn, spec_arn=spec_arn)
    print(f"  run_arn     : {run_arn}")
    print(f"\nConsole URL : https://{args.region}.console.aws.amazon.com/devicefarm/home"
          f"#/mobile/projects/{args.project_arn.split(':project:')[-1]}/runs/"
          f"{run_arn.split('/')[-1]}")

    if args.no_poll:
        print("\nrun scheduled — exiting (use --no-poll=false to wait).")
        return 0

    final = _poll_run(client, run_arn)
    print(f"\nrun completed: result={final.get('result')} status={final.get('status')}")

    out_dir = RUNS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = _download_customer_artifacts(client, run_arn, out_dir)
    if zip_path:
        print(f"\nartifacts downloaded → {zip_path.relative_to(HERE)}")
        print(f"\nNext:")
        print(f"  .venv/bin/python ch11_4_android.py unpack-artifacts --zip {zip_path}")
        print(f"  .venv/bin/python ch11_4_android.py ingest-apk-results "
              f"--input android/runs/{run_name}/results-android.json")
        return 0
    print("warning: no Customer Artifacts zip found for this run; check the Console URL above.",
          file=sys.stderr)
    return 2


def _resolve_device_arn(client: Any, name_hint: str) -> str | None:
    paginator = client.get_paginator("list_devices")
    needle = name_hint.lower()
    best: dict | None = None
    for page in paginator.paginate(filters=[{"attribute": "PLATFORM", "operator": "EQUALS", "values": ["ANDROID"]}]):
        for d in page["devices"]:
            if needle in (d.get("name") or "").lower() and d.get("availability") == "HIGHLY_AVAILABLE":
                return d["arn"]
            if needle in (d.get("name") or "").lower() and best is None:
                best = d
    return best["arn"] if best else None


def _create_device_pool(client: Any, project_arn: str, run_name: str, device_arn: str) -> str:
    resp = client.create_device_pool(
        projectArn=project_arn,
        name=f"ch11-{run_name}-pool"[:63],
        description="Auto-created single-device pool for ch11-bench TestRun",
        rules=[{"attribute": "ARN", "operator": "IN", "value": json.dumps([device_arn])}],
        maxDevices=1,
    )
    return resp["devicePool"]["arn"]


def _upload(client: Any, project_arn: str, path: Path, upload_type: str) -> str:
    resp = client.create_upload(
        projectArn=project_arn,
        name=path.name,
        type=upload_type,
        contentType="application/octet-stream",
    )
    upload = resp["upload"]
    arn = upload["arn"]
    url = upload["url"]

    # PUT to presigned S3 URL. The URL is single-purpose and short-lived.
    data = path.read_bytes()
    req = urllib.request.Request(url, data=data, method="PUT",
                                 headers={"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=300) as r:  # noqa: S310 — presigned AWS URL
        if r.status not in (200, 201):
            raise RuntimeError(f"upload HTTP {r.status} for {path.name}")

    # Poll for SUCCEEDED.
    deadline = time.time() + UPLOAD_TIMEOUT_S
    while time.time() < deadline:
        cur = client.get_upload(arn=arn)["upload"]
        st = cur["status"]
        if st == "SUCCEEDED":
            return arn
        if st in ("FAILED", "ERROR"):
            raise RuntimeError(f"upload failed: {cur.get('message')}")
        time.sleep(UPLOAD_POLL_S)
    raise TimeoutError(f"upload {path.name} did not finish within {UPLOAD_TIMEOUT_S}s")


def _schedule_run(client: Any, project_arn: str, pool_arn: str,
                  app_arn: str, test_arn: str, run_name: str,
                  extra_arn: str | None = None,
                  spec_arn: str | None = None) -> str:
    test_block: dict[str, Any] = {"type": "INSTRUMENTATION", "testPackageArn": test_arn}
    if spec_arn:
        test_block["testSpecArn"] = spec_arn
    config: dict[str, Any] = {"billingMethod": "METERED"}
    if extra_arn:
        config["extraDataPackageArn"] = extra_arn
    resp = client.schedule_run(
        projectArn=project_arn,
        appArn=app_arn,
        devicePoolArn=pool_arn,
        name=run_name,
        test=test_block,
        configuration=config,
    )
    return resp["run"]["arn"]


def _poll_run(client: Any, run_arn: str) -> dict:
    deadline = time.time() + RUN_TIMEOUT_S
    last_status = None
    while time.time() < deadline:
        run = client.get_run(arn=run_arn)["run"]
        st = run["status"]
        if st != last_status:
            print(f"  run status: {st}")
            last_status = st
        if st in ("COMPLETED", "ERRORED", "STOPPED"):
            return run
        time.sleep(RUN_POLL_S)
    raise TimeoutError(f"run {run_arn} did not complete within {RUN_TIMEOUT_S}s")


def _download_customer_artifacts(client: Any, run_arn: str, out_dir: Path) -> Path | None:
    # Walk: run -> jobs -> suites -> tests; collect FILE artifacts at each level.
    jobs = client.list_jobs(arn=run_arn)["jobs"]
    found: Path | None = None
    for job in jobs:
        suites = client.list_suites(arn=job["arn"])["suites"]
        for suite in suites:
            for art in client.list_artifacts(arn=suite["arn"], type="FILE")["artifacts"]:
                if (art.get("name") or "") in CUSTOMER_ARTIFACT_NAMES or \
                   (art.get("name") or "").lower().startswith("customer artifacts"):
                    found = _download_url(art["url"], out_dir / "customer-artifacts.zip")
            tests = client.list_tests(arn=suite["arn"])["tests"]
            for test in tests:
                for art in client.list_artifacts(arn=test["arn"], type="FILE")["artifacts"]:
                    if (art.get("name") or "") in CUSTOMER_ARTIFACT_NAMES or \
                       (art.get("name") or "").lower().startswith("customer artifacts"):
                        found = _download_url(art["url"], out_dir / "customer-artifacts.zip")
        for art in client.list_artifacts(arn=job["arn"], type="FILE")["artifacts"]:
            if (art.get("name") or "") in CUSTOMER_ARTIFACT_NAMES or \
               (art.get("name") or "").lower().startswith("customer artifacts"):
                found = _download_url(art["url"], out_dir / "customer-artifacts.zip")
    return found


def _download_url(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as r:  # noqa: S310 — Device Farm-issued URL
        if r.status != 200:
            raise RuntimeError(f"download HTTP {r.status} for {dest.name}")
        dest.write_bytes(r.read())
    return dest


if __name__ == "__main__":
    sys.exit(main())
