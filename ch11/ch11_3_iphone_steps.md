# Generating iPhone Performance Reports for ch11.3

The Apple-half body of section 11.3 needs `.mlperfreport` files dropped into
`reports/`. Generate them with the following six steps; `ch11_3_apple.py
ingest-iphone-report` then parses each one and writes phone-class records.

1. Launch Xcode 16+ on the M3. Open or create a throwaway iOS workspace; ensure
   the iPhone is selected as the run destination (USB or wireless pairing).
2. In the workspace, open the `.mlpackage` you want to profile from
   `models/coreml/` (drag-and-drop into the Project Navigator works).
3. Select the model file in the navigator. The Core ML model viewer opens.
   Switch to the **Performance** tab.
4. Click **+** to add a new performance test. Choose your iPhone as the
   destination and pick a target compute unit (start with **All** to mirror
   on-device defaults).
5. Click **Run**. Xcode runs an internal benchmark loop on-device and produces
   a Performance Report. Wait for the run to finish.
6. **Right-click the report** → **Show in Finder** → copy the resulting
   `.mlperfreport` file into `<repo>/reports/`. Filename pattern recommended:
   `<variant>_<computeUnits>_<device>.mlperfreport`.

Repeat steps 4–6 for each variant × compute-unit combination you want covered.
Five variants × four compute units = 20 reports for the full 11.3.2 figure;
three variants is the minimum acceptance criterion.

For the Llama-3.2-1B prefill-only Core ML packages (after `convert-coreml-llm`):
target the iPhone with computeUnit **All** and the highest-priority test
loop. The packages are large (~2.36 GB FP16, ~590 MB palettize-4bit). Mac
inspection via `coremltools.models.compute_plan.MLComputePlan` shows
**0% ANE, 100% GPU on FP16; 0% ANE, 80% GPU + 20% CPU on palettize-4bit**.
The ANE doesn't support the gather/gather_nd/concat-heavy transformer ops
at this size — the Performance Report will confirm whether the iPhone A18
ANE shows the same routing. Run `effnet_lite0_int8_linear` first to warm
up Xcode, then the Llama variants.

Once files are in `reports/`, run:

    python ch11_3_apple.py ingest-iphone-report --report reports/<file>.mlperfreport

(or one invocation per file).

The aggregator (`python ch11_1_aggregate.py figures`) re-renders 11.1.1 with
the phone-class records included.
