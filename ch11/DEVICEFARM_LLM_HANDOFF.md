# Device Farm LLM Run — Handoff Checklist

Pixel 10 Pro / Tensor G5 / Android 16 — full vision + Llama-3.2-1B matrix.

## Why the last run came back empty

`aws_test_run/5_llm/bf3c70b5-3df2-491f-812c-aad22149e792.zip` contained
only two empty directories — no `instrument.log`, no
`results-android.json`. Diagnosis: the old `testspec_llm.yml` ended the
test phase with an unconditional `exit 1` whenever the on-device test
didn't pass cleanly. On some AWS Device Farm runner versions that aborts
the host job before `post_test` runs, so no artifact pull ever happens.

The new `testspec_llm.yml` (just rewritten) fixes this in three ways:

1. **Never `exit 1` in the test phase.** The instrument result is logged
   but not used as a phase-exit code. `post_test` always runs.
2. **Every artifact pull tolerates failure** (`|| true`). If the app
   crashed before producing `results-android.json`, we still get the
   error marker and the logcat back.
3. **Broader EXTERNAL_DATA path probing.** Includes
   `$DEVICEFARM_EXTRA_DATA_PATH`, `$DEVICEFARM_AUX_APP_PATH`, and several
   observed locations so the `.task` file gets found regardless of the
   runner version.

The new spec also pulls a 20 MB logcat tail into the artifact zip — so
even if the next run produces zero LLM records, we'll have a post-mortem.

## Prerequisites

### 1. The `.task` file (conversion via Colab)

`models/mediapipe/llama_3_2_1b_instruct_int8.task` must exist and be
about 1 GB. To produce it: follow
`convert_llama_task/COLAB_INSTRUCTIONS.md`. The Docker-on-Mac path is
blocked by Google's arm64-wheel gap and is documented in that file's
header.

Don't skip the size check — a ~10 MB or smaller file usually means the
bundler ran on a corrupted/incomplete TFLite and the run will skip LLM
on-device.

### 2. The four upload files

```
android/app/build/outputs/apk/debug/app-debug.apk                 ~78 MB   (already built, 2026-05-12)
android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk   ~0.8 MB
aws_devicefarm/testspec_llm.yml                                   ~4 KB
aws_devicefarm/extradata.zip                                      ~1 GB    (produced below)
```

Sanity-check the APKs haven't been clobbered:

```bash
ls -la android/app/build/outputs/apk/debug/app-debug.apk \
       android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
```

Build them fresh if either is missing or older than the Kotlin sources:

```bash
cd android && ./gradlew assembleDebug assembleDebugAndroidTest && cd ..
```

### 3. Build `extradata.zip` from the `.task`

```bash
cd ~/Downloads/ch11
.venv/bin/python aws_devicefarm/make_extradata.py \
    --task models/mediapipe/llama_3_2_1b_instruct_int8.task \
    --out aws_devicefarm/extradata.zip
```

Expect output `wrote .../extradata.zip (~1000 MiB total)`. The script
refuses to produce a zip over 2 GiB (Device Farm's EXTERNAL_DATA cap).

## AWS Console upload — step-by-step

1. https://us-west-2.console.aws.amazon.com/devicefarm/home#/mobile →
   pick the `ch11-android` project.
2. **Create a new run** → **Native application (Android)**.
3. **Step 1 (Application):** upload
   `android/app/build/outputs/apk/debug/app-debug.apk`.
4. **Step 2 (Configure):**
   - Test type: **Instrumentation**.
   - Upload `android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk`.
   - Expand **"Choose your test spec"** and upload
     `aws_devicefarm/testspec_llm.yml`. Skipping this step is what
     caused the original "no results-android.json in artifacts" symptom
     on run 3 — without a custom spec, AWS won't run our post_test
     artifact pull.
4a. **Additional data:** still inside Step 2, expand **"Add extra
    data"** and upload `aws_devicefarm/extradata.zip`. This is what
    populates the `.task` file on-device. If you skip this, the run
    will complete but produce zero LLM records (vision-only matrix).
5. **Step 3 (Select devices):** **Google Pixel 10 Pro** (HIGHLY_AVAILABLE,
   Tensor G5). Single device, no pool. Same SKU as the working vision
   run on 2026-05-12.
6. **Step 4 (Device state):** keep defaults.
7. **Step 5 (Review):** check the upload list shows all four files —
   app APK, test APK, test spec, extra data — then start. Run cost
   ~$5–8 (the LLM matrix runs ~50–70 min wall-clock at $0.17/min).
8. **Wait** for status **Completed**.

## Pulling results back

```bash
# Get the Customer Artifacts ZIP from the AWS Console: drill into the run
# → the job → the suite → the test → Files tab → "Customer Artifacts".
# Save it locally, e.g. ~/Downloads/customer-artifacts.zip.

cd ~/Downloads/ch11
.venv/bin/python ch11_4_android.py unpack-artifacts \
    --zip ~/Downloads/customer-artifacts.zip \
    --out android/runs/pixel10pro_llm
.venv/bin/python ch11_4_android.py ingest-apk-results \
    --input android/runs/pixel10pro_llm/results-android.json
.venv/bin/python ch11_1_aggregate.py all
```

## How to know the run actually worked

After `unpack-artifacts`, the `android/runs/pixel10pro_llm/` directory
should contain:

```
results-android.json        # >100 KB, includes both vision and LLM records
results-android-error.txt   # empty (good) or app crash trace (bad)
instrument.log              # full instrumentation output
logcat.txt                  # 20 MB logcat tail; useful only for post-mortem
```

The JSON should have model entries for both `efficientnet_lite0` AND
`llama_3_2_1b_instruct`. Quick check:

```bash
grep '"model":' android/runs/pixel10pro_llm/results-android.json | sort -u
```

Expected output:

```
      "model": "efficientnet_lite0",
      "model": "llama_3_2_1b_instruct",
```

If only `efficientnet_lite0` appears: the `.task` wasn't found at
runtime. Open `instrument.log` and search for `EXTRA_DATA` and
`Probing` — the testspec's pre_test phase prints which candidate path
it picked, or "NO .task FILE FOUND IN ANY CANDIDATE PATH" if all
probes failed. That tells you whether the `extradata.zip` upload
landed where the runner expected it.