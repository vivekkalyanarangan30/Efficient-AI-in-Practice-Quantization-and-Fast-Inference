package com.ch11.bench

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/**
 * Translates [BenchRecord] objects into the chapter's results.json schema v11.0.
 *
 * The shape of every record we emit must round-trip through the Python
 * `ResultRecord` dataclass in `ch11_1_aggregate.py` so the aggregator and
 * existing figures don't choke. See `ch11_4_android.py ingest-apk-results`
 * for the matching parser.
 */
class ResultsWriter(private val outFile: File) {

    fun write(records: List<BenchRecord>) {
        val arr = JSONArray()
        for (r in records) arr.put(toJson(r))
        val root = JSONObject().apply {
            put("schema_version", "11.0")
            put("generated_at", iso8601Now())
            put("records", arr)
        }
        outFile.parentFile?.mkdirs()
        outFile.writeText(root.toString(2))
    }

    private fun toJson(r: BenchRecord): JSONObject {
        val device = JSONObject().apply {
            put("name", r.device.deviceName)
            put("soc", r.device.soc)
            put("os", r.device.os)
            put("class", r.device.klass)
        }
        val isLlm = r.modality == "text"
        val latency = JSONObject().apply {
            put("p50", r.latencyP50Ms)
            put("p95", r.latencyP95Ms)
            put("mean", r.latencyMeanMs)
            put("n_iters", r.nIters)
            put("warmup_iters", r.warmupIters)
            put("input_shape", JSONArray(r.inputShape))
            if (r.ttftMs != null && r.ttftMs.isFinite()) put("ttft_ms", r.ttftMs)
            if (r.tpotMs != null && r.tpotMs.isFinite()) put("tpot_ms", r.tpotMs)
        }
        val throughput = JSONObject().apply {
            if (isLlm) {
                put("samples_per_sec", JSONObject.NULL)
                put("tokens_per_sec", r.throughputTokensPerSec ?: JSONObject.NULL)
                put("prompt_length", r.promptLength ?: JSONObject.NULL)
                put("generation_length", r.generationLength ?: JSONObject.NULL)
            } else {
                put("samples_per_sec", r.throughputSamplesPerSec)
                put("tokens_per_sec", JSONObject.NULL)
                put("prompt_length", JSONObject.NULL)
                put("generation_length", JSONObject.NULL)
            }
        }
        val accuracy: Any = if (isLlm) {
            // LLM accuracy is optional — only emit when measured. Metric/value come
            // from the LLMBenchmark (HellaSwag single-token heuristic, like Mac).
            if (r.accuracyMetric != null && r.accuracyValue != null && r.accuracyValue.isFinite()) {
                JSONObject().apply {
                    put("metric", r.accuracyMetric)
                    put("value", r.accuracyValue)
                    put("secondary", JSONObject.NULL)
                    put("dataset", r.accuracyDataset)
                    put("n_samples", r.accuracyN)
                }
            } else JSONObject.NULL
        } else if (r.accuracyTop1.isFinite()) {
            JSONObject().apply {
                put("metric", "top1")
                put("value", r.accuracyTop1)
                put("secondary", JSONObject().apply { put("top5", r.accuracyTop5) })
                put("dataset", r.accuracyDataset)
                put("n_samples", r.accuracyN)
            }
        } else JSONObject.NULL

        // Prepost JSON object — populated only when the record explicitly
        // measured a pre/post-processing stage. Keys mirror the Mac side's
        // schema (ch11_5_prepost.py) so the chapter aggregator sees the
        // same shape from both platforms.
        val isPrepost = r.backend == "prepost"
        val prepostObj: Any = if (
            r.prepostDecodeMs != null || r.prepostResizeMs != null ||
            r.prepostNormalizeMs != null || r.prepostLogmelMs != null ||
            r.prepostTokenizeMs != null || r.prepostDetokenizeMs != null
        ) {
            JSONObject().apply {
                put("decode_ms", r.prepostDecodeMs ?: JSONObject.NULL)
                put("resize_ms", r.prepostResizeMs ?: JSONObject.NULL)
                put("normalize_ms", r.prepostNormalizeMs ?: JSONObject.NULL)
                put("tokenize_ms", r.prepostTokenizeMs ?: JSONObject.NULL)
                put("detokenize_ms", r.prepostDetokenizeMs ?: JSONObject.NULL)
                put("logmel_ms", r.prepostLogmelMs ?: JSONObject.NULL)
                put("nms_ms", JSONObject.NULL)
            }
        } else JSONObject.NULL

        val out = JSONObject().apply {
            put("model", r.model)
            put("modality", r.modality)
            put("variant", r.variant)
            put("backend", r.backend)
            put("device", device)
            put("size_bytes", r.sizeBytes ?: JSONObject.NULL)
            put("compute_units", r.computeUnits)
            put("latency_ms", latency)
            put("throughput", throughput)
            put("accuracy", accuracy)
            put("power_mw", JSONObject.NULL)
            put("sustained", JSONObject.NULL)
            put("ane_op_coverage", JSONObject.NULL)
            put("prepost", prepostObj)
            put("timestamp", iso8601Now())
            // Prepost rows attribute to ch11_5_prepost.py (mirror the Mac
            // §11.5 path); inference rows stay attributed to ch11_4_android.py.
            put("script", if (isPrepost) "ch11_5_prepost.py" else "ch11_4_android.py")
            put("notes", "AWS Device Farm Test Run; APK ch11-bench v1.0")
            if (r.workingMemoryMb != null && r.workingMemoryMb.isFinite()) {
                put("memory_mb", JSONObject().apply {
                    put("working_set", r.workingMemoryMb)
                })
            }
        }

        if (r.kind == RecordKind.SUSTAINED || r.kind == RecordKind.LLM_SUSTAINED) {
            out.put("sustained", JSONObject().apply {
                put("window_s", r.sustainedWindowS)
                put("throughput_first_30s", r.throughputFirst30s)
                put("throughput_last_30s", r.throughputLast30s)
                put("thermal_pressure_observed", r.thermalPressureObserved)
            })
        }
        if ((r.kind == RecordKind.POWER || r.kind == RecordKind.LLM_POWER) &&
            r.powerMeanMw != null && r.powerMeanMw.isFinite()) {
            out.put("power_mw", JSONObject().apply {
                put("mean", r.powerMeanMw)
                put("peak", r.powerPeakMw ?: JSONObject.NULL)
                put("source", r.powerSource ?: "BatteryManager")
                put("window_s", r.powerWindowS ?: 30)
            })
            if (r.energyPerInferenceMj != null && r.energyPerInferenceMj.isFinite()) {
                out.put("energy_per_inference_mj", r.energyPerInferenceMj)
            }
        }
        return out
    }

    private fun iso8601Now(): String {
        val sdf = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US)
        sdf.timeZone = TimeZone.getTimeZone("UTC")
        return sdf.format(Date())
    }
}
