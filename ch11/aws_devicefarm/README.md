# AWS Device Farm Test Run — ch11-bench

The Kotlin APK is built and ready. Two paths to actually execute the run on a real Pixel 10 Pro — pick whichever is easier.

Three files to upload this time (the testspec is the new addition — the previous run failed to pull `results-android.json` because the default testspec only collects `$DEVICEFARM_LOG_DIR` on the host, not files from the app sandbox):

| Role | Path | Size |
|------|------|------|
| App APK | `android/app/build/outputs/apk/debug/app-debug.apk` | ~70 MB |
| Test APK | `android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk` | ~1 MB |
| Test spec | `aws_devicefarm/testspec.yml` | small |

The instrumentation test `BenchmarkInstrumentationTest.runFullBenchmark` launches `MainActivity`, which writes results to `getExternalFilesDir(null)/results-android.json`. The custom `testspec.yml` runs `adb shell run-as com.ch11.bench cat files/results-android.json` after the test and dumps it into `$DEVICEFARM_LOG_DIR`, where AWS collects it as Customer Artifacts.

---

## Path A — AWS Console (no setup needed, recommended for one-time)

1. Open `https://us-west-2.console.aws.amazon.com/devicefarm/home#/mobile` (same console you used earlier).
2. Click your `ch11-android` project.
3. **Create a new run** → choose **Native application (Android)**.
4. **Step 1 (Application):** Upload `android/app/build/outputs/apk/debug/app-debug.apk` (the freshly rebuilt one, ~70 MB).
5. **Step 2 (Configure):** Test type **Instrumentation**. Upload `android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk`. **Then expand "Choose your test spec"** and **upload `aws_devicefarm/testspec.yml`**. This is the new step — without it AWS won't pull our JSON.
6. **Step 3 (Select devices):** Pick **Google Pixel 10 Pro** (HIGHLY_AVAILABLE — newer Tensor G5 / EdgeTPU, Android 16, more current NNAPI/LiteRT story than Pixel 9). Single device, no pool.
7. **Step 4 (Device state):** Keep defaults. No special files needed.
8. **Step 5 (Review):** Confirm and start. Wait 10-15 min for it to finish. Cost: ~$2-3.
9. When status is **Completed**, drill into the run → the job → the suite → the test → **Files** tab → download **Customer Artifacts**. You get a `.zip` that should now contain `results-android.json`.
10. Save the ZIP somewhere local (e.g., `~/Downloads/customer-artifacts.zip`).
11. Back in the terminal:
    ```bash
    cd /Users/vivekkalyanarangan/Downloads/ch11
    .venv/bin/python ch11_4_android.py unpack-artifacts --zip ~/Downloads/customer-artifacts.zip
    .venv/bin/python ch11_4_android.py ingest-apk-results \
        --input android/runs/customer-artifacts/results-android.json
    .venv/bin/python ch11_1_aggregate.py all
    ```

---

## Path B — Scripted via boto3 (faster on reruns; needs AWS CLI creds)

Requires programmatic AWS credentials with `devicefarm:*` permission. Update `~/.aws/credentials` (the current values appear stale — `aws devicefarm list-projects` returns "security token invalid") or export:

```bash
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-west-2
```

Then:

```bash
# 1) List your projects to find the ARN
.venv/bin/python -c "
import boto3
for p in boto3.client('devicefarm', region_name='us-west-2').list_projects()['projects']:
    print(f\"{p['name']:32s} {p['arn']}\")
"

# 2) Schedule the run
.venv/bin/python aws_devicefarm/run.py \
    --project-arn arn:aws:devicefarm:us-west-2:<acct>:project:<uuid> \
    --device "Google Pixel 9"

# 3) The script polls until done, downloads the Customer Artifacts ZIP, and
#    prints the exact ingest command.
```

The runner uploads both APKs, creates a single-device pool for the chosen device, schedules an `INSTRUMENTATION` run, polls until the run completes, then downloads the Customer Artifacts ZIP under `android/runs/<run-name>/customer-artifacts.zip`.

---

## What to expect in the result

After successful ingest, `results.json` gains ~18 new records:

- 4 variants × 4 backends (XNNPACK_1T, XNNPACK_4T, GPU, NNAPI) = 16 latency+accuracy records, `compute_units` ∈ {`xnnpack_1t`, `xnnpack_4t`, `gpu`, `nnapi`}
- 1 sustained record, `compute_units = "nnapi_sustained_300s"`
- 1 power record, `compute_units = "nnapi_power_30s"`

Aggregator markers: `device.class = "phone"` → squares on the design-space plot, alongside the iPhone Core ML squares.

Combinations that fail on the device (e.g., GPU delegate not supporting int8) are skipped silently — `Logcat: Ch11Bench` shows `SKIPPED` for those combos. This is expected and not a failure.
