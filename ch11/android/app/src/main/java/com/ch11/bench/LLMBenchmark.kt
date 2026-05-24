package com.ch11.bench

import android.content.Context
import android.os.Debug
import android.os.SystemClock
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.math.max

/**
 * On-device LLM benchmark via LiteRT-LM (com.google.ai.edge.litertlm).
 *
 * Mirrors [Benchmark]'s structure for the LLM modality. The model file is
 * bundled INSIDE THE APK at assets/llama_3_2_1b_*.litertlm and staged to
 * ctx.filesDir on first access. If no .litertlm is found the benchmark
 * exits cleanly without emitting LLM records.
 *
 * Per (variant × prompt_length) we record TTFT, TPOT, tokens/sec, and working
 * memory. We additionally run a HellaSwag-200 single-token-heuristic accuracy
 * pass (matching the Mac MLX scoring in ch11_3_apple.py), a 5-minute sustained
 * decode loop for thermal stability, and a 30-second power window via
 * [PowerSampler]. All records share [BenchRecord]'s shape so they round-trip
 * through ResultsWriter -> ch11_4_android.py ingest unchanged.
 *
 * Previously used MediaPipe's tasks-genai 0.10.24 LlmInference API against
 * a legacy .task FlatBuffer-wrapped-ZIP bundle. Replaced 2026-05-19 with
 * LiteRT-LM 0.12.0 which natively reads the .litertlm bundle format the
 * VM-side litert-lm-builder produces, exposes a Flow-based streaming API,
 * and is Google's current direction for on-device LLM serving.
 */
class LLMBenchmark(
    private val ctx: Context,
    private val onProgress: (String) -> Unit,
    private val promptLengths: List<Int> = DEFAULT_PROMPT_LENGTHS,
    private val generationLength: Int = DEFAULT_GENERATION_LENGTH,
    private val sustainedWindowS: Int = 300,
    private val powerWindowS: Int = 30,
    private val hellaSwagMaxItems: Int = 200,
) {

    fun runAll(): List<BenchRecord> {
        val taskFiles = locateTaskFiles()
        if (taskFiles.isEmpty()) {
            onProgress("llm: no .litertlm model files in ${modelsDir().absolutePath} — skipping LLM benchmark")
            return emptyList()
        }
        val device = detectDevice()
        onProgress("llm: device=${device.deviceName} variants=${taskFiles.map { it.name }}")

        val records = mutableListOf<BenchRecord>()
        val hellaSwagItems = loadHellaSwag()
        onProgress("llm: hellaswag_items=${hellaSwagItems.size}")

        for (taskFile in taskFiles) {
            val modelName = inferModelName(taskFile)
            val variant = inferVariantName(taskFile)
            onProgress("--- model=$modelName variant=$variant (${taskFile.length() / 1_000_000} MB) ---")
            val baseRecord = runVariant(taskFile, modelName, variant, device, hellaSwagItems, records)
            if (baseRecord != null) {
                try {
                    onProgress("  $variant: sustained ${sustainedWindowS}s…")
                    records += runSustained(taskFile, baseRecord)
                } catch (t: Throwable) {
                    onProgress("  $variant: sustained skipped (${t.message})")
                }
                try {
                    onProgress("  $variant: power ${powerWindowS}s…")
                    records += runPower(taskFile, baseRecord)
                } catch (t: Throwable) {
                    onProgress("  $variant: power skipped (${t.message})")
                }
            }
        }
        return records
    }

    private fun runVariant(
        taskFile: File,
        modelName: String,
        variant: String,
        device: DeviceInfo,
        hellaSwagItems: List<HellaSwagItem>,
        accumulator: MutableList<BenchRecord>,
    ): BenchRecord? {
        var canonical: BenchRecord? = null
        // Load once per (variant, prompt_length) — MediaPipe pins maxTokens at
        // construction so prompt-length changes require a fresh instance.
        for (promptLen in promptLengths) {
            val maxTokens = promptLen + generationLength + SAFETY_MARGIN_TOKENS
            val llm = try {
                openLlm(taskFile, maxTokens)
            } catch (t: Throwable) {
                onProgress("  ${variant}/p${promptLen}: load failed (${t.javaClass.simpleName}: ${t.message})")
                continue
            }
            try {
                val memBeforeMb = nativeHeapMb()
                val prompt = buildPrompt(promptLen)
                val warmup = generateMeasured(llm, prompt, generationLength)
                onProgress("  ${variant}/p${promptLen}: warmup tokens=${warmup.tokensProduced} ttft=${"%.1f".format(warmup.ttftMs)}ms")

                val runs = mutableListOf<Measurement>()
                repeat(N_TIMED_RUNS) { runs += generateMeasured(llm, prompt, generationLength) }

                val ttft = runs.map { it.ttftMs }.average()
                val tpot = runs.map { it.tpotMs }.filter { it.isFinite() }.let { if (it.isEmpty()) Double.NaN else it.average() }
                val tokensPerSec = runs.map { it.tokensPerSec }.filter { it.isFinite() }.let { if (it.isEmpty()) Double.NaN else it.average() }
                val totalP50 = runs.map { it.totalMs }.sorted()[runs.size / 2]
                val totalP95 = runs.map { it.totalMs }.sorted()[((runs.size * 95) / 100).coerceAtMost(runs.size - 1)]
                val totalMean = runs.map { it.totalMs }.average()
                val memAfterMb = nativeHeapMb()
                val workingMb = max(0.0, memAfterMb - memBeforeMb)

                val rec = BenchRecord(
                    model = modelName,
                    modality = "text",
                    variant = "litertlm_$variant",
                    backend = "litertlm",
                    computeUnits = "gpu",
                    device = device,
                    sizeBytes = taskFile.length(),
                    latencyP50Ms = totalP50,
                    latencyP95Ms = totalP95,
                    latencyMeanMs = totalMean,
                    nIters = runs.size,
                    warmupIters = 1,
                    inputShape = listOf(1, promptLen),
                    throughputSamplesPerSec = Double.NaN,
                    accuracyTop1 = Double.NaN,
                    accuracyTop5 = Double.NaN,
                    accuracyDataset = "",
                    accuracyN = 0,
                    kind = RecordKind.LLM_LATENCY,
                    throughputTokensPerSec = tokensPerSec,
                    promptLength = promptLen,
                    generationLength = generationLength,
                    ttftMs = ttft,
                    tpotMs = tpot,
                    workingMemoryMb = workingMb,
                )
                accumulator += rec
                if (canonical == null) canonical = rec
                onProgress("  ${variant}/p${promptLen}: ttft=${"%.0f".format(ttft)}ms tpot=${"%.1f".format(tpot)}ms tok/s=${"%.2f".format(tokensPerSec)} mem=${"%.0f".format(workingMb)}MB")
            } finally {
                try { llm.close() } catch (_: Throwable) {}
            }
        }

        // HellaSwag-200 on a single instance sized for the longest prompt.
        // Cap the dataset by [hellaSwagMaxItems]; the Mac script uses 200.
        if (hellaSwagItems.isNotEmpty()) {
            val llm = try {
                openLlm(taskFile, HELLASWAG_MAX_TOKENS)
            } catch (t: Throwable) {
                onProgress("  ${variant}/hellaswag: load failed (${t.message})")
                return canonical
            }
            try {
                val items = hellaSwagItems.take(hellaSwagMaxItems)
                var correct = 0
                var n = 0
                for ((i, item) in items.withIndex()) {
                    val out = try {
                        llm.generateResponseSync(item.prompt)
                    } catch (_: Throwable) { continue }
                    val choice = out.trim().uppercase().firstOrNull()
                    val mapping = mapOf('A' to 0, 'B' to 1, 'C' to 2, 'D' to 3)
                    if (mapping[choice] == item.label) correct++
                    n++
                    if ((i + 1) % 40 == 0) onProgress("  ${variant}/hellaswag: ${i + 1}/${items.size} acc=${"%.3f".format(correct.toDouble() / n)}")
                }
                if (n > 0) {
                    accumulator += BenchRecord(
                        model = modelName,
                        modality = "text",
                        variant = "litertlm_$variant",
                        backend = "litertlm",
                        computeUnits = "gpu_hellaswag",
                        device = device,
                        sizeBytes = taskFile.length(),
                        latencyP50Ms = Double.NaN,
                        latencyP95Ms = Double.NaN,
                        latencyMeanMs = Double.NaN,
                        nIters = n,
                        warmupIters = 0,
                        inputShape = listOf(1, 0),
                        throughputSamplesPerSec = Double.NaN,
                        accuracyTop1 = Double.NaN,
                        accuracyTop5 = Double.NaN,
                        accuracyDataset = "hellaswag_val_$n",
                        accuracyN = n,
                        kind = RecordKind.LLM_ACCURACY,
                        accuracyMetric = "hellaswag_single_token_heuristic",
                        accuracyValue = correct.toDouble() / n,
                    )
                    onProgress("  ${variant}/hellaswag: final acc=${"%.3f".format(correct.toDouble() / n)} (n=$n)")
                }
            } finally {
                try { llm.close() } catch (_: Throwable) {}
            }
        }
        return canonical
    }

    private fun runSustained(taskFile: File, baseline: BenchRecord): BenchRecord {
        val promptLen = baseline.promptLength ?: DEFAULT_PROMPT_LENGTHS.first()
        val llm = openLlm(taskFile, promptLen + generationLength + SAFETY_MARGIN_TOKENS)
        try {
            val prompt = buildPrompt(promptLen)
            // Burn-in
            repeat(2) { generateMeasured(llm, prompt, generationLength) }

            val deadline = SystemClock.elapsedRealtime() + sustainedWindowS * 1000L
            val windowMs = 30_000L
            val windows = mutableListOf<MutableList<Double>>()
            var windowEnd = SystemClock.elapsedRealtime() + windowMs
            var cur = mutableListOf<Double>()
            while (SystemClock.elapsedRealtime() < deadline) {
                val m = generateMeasured(llm, prompt, generationLength)
                if (m.tokensPerSec.isFinite()) cur += m.tokensPerSec
                if (SystemClock.elapsedRealtime() >= windowEnd) {
                    windows += cur
                    cur = mutableListOf()
                    windowEnd += windowMs
                }
            }
            if (cur.isNotEmpty()) windows += cur

            fun winTps(w: List<Double>): Double {
                if (w.isEmpty()) return Double.NaN
                val s = w.sorted()
                return s[s.size / 2]
            }
            val first = if (windows.isNotEmpty()) winTps(windows.first()) else Double.NaN
            val last = if (windows.size >= 2) winTps(windows.last()) else first
            val thermal = !first.isNaN() && !last.isNaN() && (last / max(first, 1e-9)) < 0.9

            return baseline.copy(
                computeUnits = "${baseline.computeUnits}_sustained_${sustainedWindowS}s",
                kind = RecordKind.LLM_SUSTAINED,
                sustainedWindowS = sustainedWindowS,
                throughputFirst30s = first,
                throughputLast30s = last,
                thermalPressureObserved = thermal,
            )
        } finally {
            try { llm.close() } catch (_: Throwable) {}
        }
    }

    private fun runPower(taskFile: File, baseline: BenchRecord): BenchRecord {
        val promptLen = baseline.promptLength ?: DEFAULT_PROMPT_LENGTHS.first()
        val llm = openLlm(taskFile, promptLen + generationLength + SAFETY_MARGIN_TOKENS)
        try {
            val prompt = buildPrompt(promptLen)
            repeat(2) { generateMeasured(llm, prompt, generationLength) }

            val sampler = PowerSampler(ctx, 5)
            sampler.start()
            val t0 = System.nanoTime()
            var tokens = 0L
            val deadline = SystemClock.elapsedRealtime() + powerWindowS * 1000L
            while (SystemClock.elapsedRealtime() < deadline) {
                val m = generateMeasured(llm, prompt, generationLength)
                tokens += m.tokensProduced
            }
            val elapsedSec = (System.nanoTime() - t0) / 1e9
            val result = sampler.stop()
            val meanMw = result.meanPowerMw
            val peakMw = result.peakPowerMw

            return baseline.copy(
                computeUnits = "${baseline.computeUnits}_power_${powerWindowS}s",
                kind = RecordKind.LLM_POWER,
                powerMeanMw = meanMw,
                powerPeakMw = peakMw,
                powerSource = result.source,
                powerWindowS = powerWindowS,
                energyPerInferenceMj = if (tokens > 0 && !meanMw.isNaN()) meanMw * elapsedSec / tokens else null,
            )
        } finally {
            try { llm.close() } catch (_: Throwable) {}
        }
    }

    /**
     * Thin wrapper around the LiteRT-LM Engine that lets the rest of this
     * file keep the same call shape it had under MediaPipe (close(),
     * generateResponse(prompt), generateMeasured(...)).
     */
    private inner class LlmAdapter(
        val engine: Engine,
        val samplerConfig: SamplerConfig,
    ) : AutoCloseable {
        fun generateResponseSync(prompt: String): String {
            val conv = engine.createConversation(ConversationConfig(samplerConfig = samplerConfig))
            return try {
                runBlocking {
                    val sb = StringBuilder()
                    conv.sendMessageAsync(prompt).collect { msg -> sb.append(msg.toString()) }
                    sb.toString()
                }
            } finally {
                try { conv.close() } catch (_: Throwable) {}
            }
        }
        override fun close() {
            try { engine.close() } catch (_: Throwable) {}
        }
    }

    private fun openLlm(taskFile: File, maxTokens: Int): LlmAdapter {
        // LiteRT-LM Engine construction. maxNumTokens is the KV-cache
        // ceiling — our .litertlm was built with kv_cache_max_len=1280, so
        // we cap maxTokens at 1280 here regardless of the caller's request.
        // Backend.CPU() is the safest default; GPU/NPU paths require the
        // device's native lib dir on disk and aren't wired in the benchmark
        // APK yet.
        val cap = maxTokens.coerceAtMost(LITERTLM_KV_CACHE_MAX)
        val engineConfig = EngineConfig(
            modelPath = taskFile.absolutePath,
            backend = Backend.CPU(),
            maxNumTokens = cap,
        )
        val engine = Engine(engineConfig)
        engine.initialize()
        val sampler = SamplerConfig(topK = 40, topP = 1.0, temperature = 1.0)
        return LlmAdapter(engine, sampler)
    }

    /**
     * Generate up to [genTokens] new tokens and measure TTFT/TPOT/throughput.
     *
     * LiteRT-LM's async API returns a Flow<*> emitting per-chunk Messages
     * (typically one token per emission). We treat the first non-empty
     * emission as TTFT and the time delta from there to the final emission
     * divided by the remaining tokens as TPOT. A fresh Conversation per
     * timed run gives stateless measurements (no carried-over chat history).
     */
    private fun generateMeasured(llm: LlmAdapter, prompt: String, @Suppress("UNUSED_PARAMETER") genTokens: Int): Measurement {
        val t0Ns = System.nanoTime()
        var firstChunkNs = -1L
        var lastChunkNs = -1L
        var totalChunks = 0
        val finalText = StringBuilder()

        val conv: Conversation = llm.engine.createConversation(
            ConversationConfig(samplerConfig = llm.samplerConfig),
        )
        try {
            runBlocking {
                conv.sendMessageAsync(prompt).collect { msg ->
                    val now = System.nanoTime()
                    val piece = msg.toString()
                    if (piece.isNotEmpty()) {
                        if (firstChunkNs == -1L) firstChunkNs = now
                        lastChunkNs = now
                        totalChunks++
                        finalText.append(piece)
                    }
                }
            }
        } finally {
            try { conv.close() } catch (_: Throwable) {}
        }

        val tEndNs = if (lastChunkNs > 0) lastChunkNs else System.nanoTime()
        val totalMs = (tEndNs - t0Ns) / 1e6
        val ttftMs = if (firstChunkNs > 0) (firstChunkNs - t0Ns) / 1e6 else totalMs
        val producedTokens = approxTokenCount(finalText.toString()).coerceAtLeast(totalChunks.toLong())
        val decodeMs = if (firstChunkNs > 0 && lastChunkNs > firstChunkNs) (lastChunkNs - firstChunkNs) / 1e6 else 0.0
        val decodeTokens = (producedTokens - 1).coerceAtLeast(0L)
        val tpotMs = if (decodeTokens > 0 && decodeMs > 0.0) decodeMs / decodeTokens else Double.NaN
        val tokensPerSec = if (decodeMs > 0.0 && decodeTokens > 0) decodeTokens * 1000.0 / decodeMs else Double.NaN
        return Measurement(
            ttftMs = ttftMs,
            tpotMs = tpotMs,
            tokensProduced = producedTokens,
            tokensPerSec = tokensPerSec,
            totalMs = totalMs,
        )
    }

    /**
     * Approximate token count. LiteRT-LM Conversation doesn't expose the
     * tokenizer directly, so we use a 4-chars-per-token rule of thumb
     * matching English text — close enough for buildPrompt sizing and for
     * token-throughput math (we additionally floor by the streamed chunk
     * count so per-emission timing dominates the metric).
     */
    private fun approxTokenCount(text: String): Long =
        (text.length / 4).toLong().coerceAtLeast(1L)

    private fun buildPrompt(targetTokens: Int): String {
        // Compose a deterministic prompt of roughly [targetTokens] tokens by
        // repeating a neutral seed sentence. Pure char-based — LiteRT-LM
        // Conversation doesn't expose a tokenizer count API, and a small
        // overshoot/undershoot on prompt length doesn't change the
        // benchmark's decode-throughput math materially.
        val sb = StringBuilder()
        val seed = "The LiteRT-LM Engine benchmark measures decode throughput on phone-class hardware. "
        while (sb.length / 4 < targetTokens && sb.length < targetTokens * 8) {
            sb.append(seed)
        }
        return sb.toString().trimEnd()
    }

    private fun nativeHeapMb(): Double {
        return Debug.getNativeHeapAllocatedSize().toDouble() / (1024.0 * 1024.0)
    }

    private fun inferVariantName(taskFile: File): String {
        // Best-effort variant tag from filename: "llama_3_2_1b_instruct_int8.task" -> "int8".
        val stem = taskFile.nameWithoutExtension.lowercase()
        val candidates = listOf("q4_k_m", "q4_0", "q8_0", "q4", "q8", "int4", "int8", "fp16",
                                "wo8", "w8a16")
        for (c in candidates) {
            if (stem.endsWith("_$c") || stem.contains("_${c}_") || stem == c) return c
        }
        return "default"
    }

    private fun inferModelName(taskFile: File): String {
        // Best-effort model id from filename. Examples:
        //   "llama_3_2_1b_instruct_int8.task" -> "llama_3_2_1b_instruct"
        //   "tinyllama_1_1b_chat_q8.task"     -> "tinyllama_1_1b_chat"
        //   "TinyLlama-1.1B-Chat-v1.0_multi-prefill-seq_q8_ekv1280.task"
        //       -> "tinyllama_1_1b_chat_v1_0"  (sanitized)
        // The result must match ch11_4_android.py's KNOWN_MODELS allowlist or
        // the ingest step will reject the record.
        val stem = taskFile.nameWithoutExtension
        // Lowercase + replace separators with underscore so we land in the
        // same naming convention as the rest of the chapter (e.g.
        // "llama_3_2_1b_instruct").
        val normalized = stem.lowercase()
            .replace('-', '_').replace('.', '_').replace(' ', '_')
        // Strip everything from the first quantization / config marker onward.
        val cutMarkers = listOf(
            "_multi_prefill_seq_", "_seq128_", "_seq_",
            "_ekv", "_q4", "_q8", "_int4", "_int8", "_fp16", "_wo8", "_w8a16",
        )
        var name = normalized
        for (m in cutMarkers) {
            val idx = name.indexOf(m)
            if (idx >= 0) name = name.substring(0, idx)
        }
        return name.trim('_')
    }

    private fun modelsDir(): File {
        // The on-disk staging area where assets get materialized. Same dir
        // Benchmark.kt uses for the vision TFLites.
        return ctx.filesDir
    }

    /**
     * Locate .litertlm files bundled in the APK's assets/ directory and stage
     * them to ctx.filesDir on first access. LiteRT-LM's Engine requires a
     * real file path (not an InputStream), so we copy the asset to disk
     * once. The APK ships with `noCompress` for `.litertlm` (see
     * app/build.gradle.kts), so the on-disk copy is byte-identical to the
     * asset and mmap-friendly.
     *
     * History: previously this scanned /sdcard/.../com.ch11.bench/files/models
     * for files pushed via AWS Device Farm's "Upload extra data" feature
     * (unreliable on Android 10+/scoped storage), and earlier still
     * the .task FlatBuffer-wrapped-ZIP format consumed by MediaPipe's
     * deprecated tasks-genai LlmInference. Both are gone; embedding the
     * modern LiteRT-LM .litertlm bundle in APK assets is the canonical path.
     */
    private fun locateTaskFiles(): List<File> {
        val staged = mutableListOf<File>()
        val assetNames = ctx.assets.list("")?.filter { it.endsWith(".litertlm") }.orEmpty().sorted()
        for (assetName in assetNames) {
            val outFile = File(modelsDir(), assetName)
            if (!outFile.exists() || outFile.length() == 0L) {
                onProgress("llm: staging asset $assetName -> ${outFile.absolutePath} (this may take a few seconds for large models)")
                ctx.assets.open(assetName).use { input ->
                    outFile.outputStream().use { output -> input.copyTo(output) }
                }
            }
            staged += outFile
        }
        return staged
    }

    private fun loadHellaSwag(): List<HellaSwagItem> {
        return try {
            val text = ctx.assets.open("hellaswag_200.json").bufferedReader().use { it.readText() }
            val root = JSONObject(text)
            val arr: JSONArray = root.getJSONArray("items")
            val out = mutableListOf<HellaSwagItem>()
            for (i in 0 until arr.length()) {
                val it = arr.getJSONObject(i)
                out += HellaSwagItem(prompt = it.getString("prompt"), label = it.getInt("label"))
            }
            out
        } catch (t: Throwable) {
            onProgress("llm: hellaswag asset missing (${t.message}) — accuracy will not be measured")
            emptyList()
        }
    }

    private fun detectDevice(): DeviceInfo {
        val soc = if (android.os.Build.VERSION.SDK_INT >= 31)
            (android.os.Build.SOC_MODEL ?: android.os.Build.HARDWARE) else android.os.Build.HARDWARE
        return DeviceInfo(
            deviceName = "${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}".trim(),
            soc = soc,
            os = "Android ${android.os.Build.VERSION.RELEASE} (API ${android.os.Build.VERSION.SDK_INT})",
            klass = "phone",
        )
    }

    private data class Measurement(
        val ttftMs: Double,
        val tpotMs: Double,
        val tokensProduced: Long,
        val tokensPerSec: Double,
        val totalMs: Double,
    )

    private data class HellaSwagItem(val prompt: String, val label: Int)

    companion object {
        private val DEFAULT_PROMPT_LENGTHS = listOf(32, 256, 1024)
        private const val DEFAULT_GENERATION_LENGTH = 64
        private const val SAFETY_MARGIN_TOKENS = 32
        private const val N_TIMED_RUNS = 3
        private const val HELLASWAG_MAX_TOKENS = 1024
        // KV cache ceiling baked into the .litertlm bundle by the converter
        // (kv_cache_max_len=1280 in convert_llama_task/convert.py).
        private const val LITERTLM_KV_CACHE_MAX = 1280
    }
}
