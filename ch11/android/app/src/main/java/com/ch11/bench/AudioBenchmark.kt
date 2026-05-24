package com.ch11.bench

import android.content.Context
import android.os.Build
import android.os.SystemClock
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.random.Random

/**
 * On-device Whisper-tiny encoder benchmark.
 *
 * Mirrors [Benchmark]'s shape for the audio modality. The encoder TFLite
 * file is NOT bundled in the APK (too large to keep the APK lean and the
 * conversion pipeline lives off-device anyway). It is expected at:
 *
 *   /sdcard/Android/data/com.ch11.bench/files/models/whisper_*.tflite
 *
 * Pushed there by the AWS Device Farm test spec's pre_test phase, which
 * unpacks the EXTERNAL_DATA upload onto the device. If no whisper_*.tflite
 * is found, this benchmark exits cleanly without emitting audio records.
 *
 * What this benchmark measures matches the Apple side's Whisper records:
 * encoder-only forward pass on a fixed input shape (typically [1, 80, 3000]
 * — 80 mel bins, 3000 frames = 30 s at 100 Hz). The Apple records have
 * `accuracy=null` for Whisper because the encoder is a featurizer, not a
 * full ASR pipeline; we follow that convention here.
 *
 * Per variant file we sweep three TFLite backends (XNNPACK 4-thread, NNAPI,
 * GPU delegate) and emit one latency record per (variant, backend). The
 * fastest combo is then used for one 300 s sustained record and one 30 s
 * power record, the same shape [Benchmark] uses for the vision matrix.
 */
class AudioBenchmark(
    private val ctx: Context,
    private val onProgress: (String) -> Unit,
    private val warmupIters: Int = 20,
    private val timedIters: Int = 200,
    private val sustainedWindowS: Int = 300,
    private val powerWindowS: Int = 30,
) {

    fun runAll(): List<BenchRecord> {
        val tflites = locateWhisperTflites()
        if (tflites.isEmpty()) {
            onProgress("audio: no whisper_*.tflite in ${modelsDir().absolutePath} — skipping audio benchmark")
            return emptyList()
        }

        val device = detectDevice()
        onProgress("audio device: ${device.deviceName} (${device.soc}, ${device.os})")
        onProgress("audio: found ${tflites.size} whisper variant(s): ${tflites.joinToString { it.name }}")

        val records = mutableListOf<BenchRecord>()
        val backends = listOf(AudioBackend.XNNPACK_4T, AudioBackend.NNAPI, AudioBackend.GPU)

        for (modelFile in tflites) {
            val variant = variantTagFromFilename(modelFile.name)
            onProgress("--- whisper_tiny ${variant} (${modelFile.length()} bytes) ---")
            val modelBuf = mapFile(modelFile)

            for (backend in backends) {
                val combo = "$variant/${backend.tag}"
                try {
                    val r = benchmarkCombo(
                        modelFile = modelFile,
                        modelBuf = modelBuf,
                        variant = variant,
                        backend = backend,
                        device = device,
                    )
                    records += r
                    onProgress("  $combo: p50=${"%.2f".format(r.latencyP50Ms)}ms")
                } catch (t: Throwable) {
                    onProgress("  $combo: SKIPPED (${t.javaClass.simpleName}: ${t.message})")
                }
            }
        }

        // Sustained + power on the fastest backend that actually succeeded.
        // Preference order mirrors the vision baseline: nnapi → xnnpack_4t → gpu.
        val baseline = listOf("nnapi", "xnnpack_4t", "gpu")
            .firstNotNullOfOrNull { cu -> records.firstOrNull { it.computeUnits == cu } }
        if (baseline != null) {
            val baselineFile = tflites.first { variantTagFromFilename(it.name) == baseline.variant }
            val baselineBackend = AudioBackend.values().first { it.tag == baseline.computeUnits }
            onProgress("audio sustained/power baseline: ${baseline.variant}/${baseline.computeUnits} (p50=${"%.2f".format(baseline.latencyP50Ms)}ms)")
            try {
                records += runSustained(baseline, baselineFile, baselineBackend)
            } catch (t: Throwable) {
                onProgress("audio sustained skipped: ${t.message}")
            }
            try {
                records += runPower(baseline, baselineFile, baselineBackend)
            } catch (t: Throwable) {
                onProgress("audio power skipped: ${t.message}")
            }
        } else {
            onProgress("audio sustained/power skipped: no working backend")
        }

        return records
    }

    private fun benchmarkCombo(
        modelFile: File,
        modelBuf: MappedByteBuffer,
        variant: String,
        backend: AudioBackend,
        device: DeviceInfo,
    ): BenchRecord {
        val (interp, delegateHandle) = newInterpreter(modelBuf, backend)
        try {
            val inputTensor = interp.getInputTensor(0)
            val inputShape = inputTensor.shape()
            val inputType = inputTensor.dataType().name
            val outputTensor = interp.getOutputTensor(0)
            val outputShape = outputTensor.shape()
            val outputType = outputTensor.dataType().name

            // For latency we don't need real audio. Fill with deterministic
            // pseudo-random floats centered around 0 — same statistical
            // profile as a typical mel-spectrogram normalized for the
            // encoder, so the model exercises its full graph.
            val inBuf = allocFilledInput(inputShape, inputType)
            val outBuf = allocOutput(outputShape, outputType)

            repeat(warmupIters) {
                inBuf.rewind(); outBuf.rewind()
                interp.run(inBuf, outBuf)
            }
            val nsPerIter = LongArray(timedIters)
            for (i in 0 until timedIters) {
                inBuf.rewind(); outBuf.rewind()
                val t0 = System.nanoTime()
                interp.run(inBuf, outBuf)
                nsPerIter[i] = System.nanoTime() - t0
            }
            nsPerIter.sort()
            val p50 = nsPerIter[(timedIters * 50 / 100).coerceAtMost(timedIters - 1)] / 1e6
            val p95 = nsPerIter[(timedIters * 95 / 100).coerceAtMost(timedIters - 1)] / 1e6
            val mean = nsPerIter.average() / 1e6

            return BenchRecord(
                model = "whisper_tiny",
                modality = "audio",
                variant = variant,
                backend = "tflite",
                computeUnits = backend.tag,
                device = device,
                sizeBytes = modelFile.length(),
                latencyP50Ms = p50,
                latencyP95Ms = p95,
                latencyMeanMs = mean,
                nIters = timedIters,
                warmupIters = warmupIters,
                inputShape = inputShape.toList(),
                throughputSamplesPerSec = 1000.0 / mean,
                // Whisper-encoder latency-only — no on-device WER; matches Apple side.
                accuracyTop1 = Double.NaN,
                accuracyTop5 = Double.NaN,
                accuracyDataset = "n/a",
                accuracyN = 0,
                kind = RecordKind.LATENCY,
            )
        } finally {
            interp.close()
            (delegateHandle as? AutoCloseable)?.close()
        }
    }

    private fun runSustained(
        baseline: BenchRecord,
        modelFile: File,
        backend: AudioBackend,
    ): BenchRecord {
        val modelBuf = mapFile(modelFile)
        val (interp, delegateHandle) = newInterpreter(modelBuf, backend)
        try {
            val inputShape = interp.getInputTensor(0).shape()
            val inputType = interp.getInputTensor(0).dataType().name
            val outputShape = interp.getOutputTensor(0).shape()
            val outputType = interp.getOutputTensor(0).dataType().name
            val inBuf = allocFilledInput(inputShape, inputType)
            val outBuf = allocOutput(outputShape, outputType)

            val deadline = SystemClock.elapsedRealtime() + sustainedWindowS * 1000L
            val windowMs = 30_000L
            val windows = mutableListOf<MutableList<Long>>()
            var windowEnd = SystemClock.elapsedRealtime() + windowMs
            var cur = mutableListOf<Long>()
            repeat(5) { inBuf.rewind(); outBuf.rewind(); interp.run(inBuf, outBuf) }

            while (SystemClock.elapsedRealtime() < deadline) {
                inBuf.rewind(); outBuf.rewind()
                val t0 = System.nanoTime()
                interp.run(inBuf, outBuf)
                cur += System.nanoTime() - t0
                if (SystemClock.elapsedRealtime() >= windowEnd) {
                    windows += cur
                    cur = mutableListOf()
                    windowEnd += windowMs
                }
            }
            if (cur.isNotEmpty()) windows += cur

            fun windowThroughput(w: List<Long>): Double {
                if (w.isEmpty()) return Double.NaN
                val sorted = w.sorted()
                return 1e9 / sorted[sorted.size / 2]
            }
            val first = if (windows.isNotEmpty()) windowThroughput(windows.first()) else Double.NaN
            val last = if (windows.size >= 2) windowThroughput(windows.last()) else first
            val thermal = !first.isNaN() && !last.isNaN() && (last / max(first, 1e-9)) < 0.9

            return baseline.copy(
                computeUnits = "${baseline.computeUnits}_sustained_${sustainedWindowS}s",
                kind = RecordKind.SUSTAINED,
                sustainedWindowS = sustainedWindowS,
                throughputFirst30s = first,
                throughputLast30s = last,
                thermalPressureObserved = thermal,
            )
        } finally {
            interp.close()
            (delegateHandle as? AutoCloseable)?.close()
        }
    }

    private fun runPower(
        baseline: BenchRecord,
        modelFile: File,
        backend: AudioBackend,
    ): BenchRecord {
        val modelBuf = mapFile(modelFile)
        val (interp, delegateHandle) = newInterpreter(modelBuf, backend)
        try {
            val inputShape = interp.getInputTensor(0).shape()
            val inputType = interp.getInputTensor(0).dataType().name
            val outputShape = interp.getOutputTensor(0).shape()
            val outputType = interp.getOutputTensor(0).dataType().name
            val inBuf = allocFilledInput(inputShape, inputType)
            val outBuf = allocOutput(outputShape, outputType)
            repeat(5) { inBuf.rewind(); outBuf.rewind(); interp.run(inBuf, outBuf) }

            val sampler = PowerSampler(ctx, 5)
            sampler.start()
            val t0 = System.nanoTime()
            var iters = 0L
            val deadline = SystemClock.elapsedRealtime() + powerWindowS * 1000L
            while (SystemClock.elapsedRealtime() < deadline) {
                inBuf.rewind(); outBuf.rewind()
                interp.run(inBuf, outBuf)
                iters++
            }
            val elapsedSec = (System.nanoTime() - t0) / 1e9
            val result = sampler.stop()

            return baseline.copy(
                computeUnits = "${baseline.computeUnits}_power_${powerWindowS}s",
                kind = RecordKind.POWER,
                powerMeanMw = result.meanPowerMw,
                powerPeakMw = result.peakPowerMw,
                powerSource = result.source,
                powerWindowS = powerWindowS,
                energyPerInferenceMj = if (iters > 0 && result.meanPowerMw.isFinite())
                    result.meanPowerMw * elapsedSec / iters else null,
            )
        } finally {
            interp.close()
            (delegateHandle as? AutoCloseable)?.close()
        }
    }

    private fun newInterpreter(modelBuf: MappedByteBuffer, backend: AudioBackend): Pair<Interpreter, Any?> {
        val opts = Interpreter.Options()
        var handle: Any? = null
        when (backend) {
            AudioBackend.XNNPACK_4T -> { opts.setNumThreads(4).setUseXNNPACK(true) }
            AudioBackend.GPU -> {
                val gpu = GpuDelegate(); handle = gpu; opts.addDelegate(gpu)
            }
            AudioBackend.NNAPI -> {
                val nnapi = NnApiDelegate(); handle = nnapi; opts.addDelegate(nnapi)
            }
        }
        return Interpreter(modelBuf, opts) to handle
    }

    private fun allocFilledInput(shape: IntArray, dtype: String): ByteBuffer {
        val elements = shape.fold(1) { a, b -> a * b }
        val bytesPerElem = when (dtype) { "FLOAT32" -> 4; "INT16" -> 2; "UINT8", "INT8" -> 1; else -> 4 }
        val buf = ByteBuffer.allocateDirect(elements * bytesPerElem).order(ByteOrder.nativeOrder())
        val rng = Random(0xC11AB1L)
        when (dtype) {
            "FLOAT32" -> repeat(elements) { buf.putFloat((rng.nextFloat() * 2f - 1f)) }
            "UINT8" -> repeat(elements) { buf.put((rng.nextInt(256)).toByte()) }
            "INT8" -> repeat(elements) { buf.put((rng.nextInt(256) - 128).toByte()) }
            "INT16" -> repeat(elements) {
                buf.putShort(((rng.nextInt(65536)) - 32768).toShort())
            }
            else -> repeat(elements) { buf.putFloat(rng.nextFloat() * 2f - 1f) }
        }
        buf.rewind()
        return buf
    }

    private fun allocOutput(shape: IntArray, dtype: String): ByteBuffer {
        val elements = shape.fold(1) { a, b -> a * b }
        val bytesPerElem = when (dtype) { "FLOAT32" -> 4; "INT16" -> 2; "UINT8", "INT8" -> 1; else -> 4 }
        return ByteBuffer.allocateDirect(elements * bytesPerElem).order(ByteOrder.nativeOrder())
    }

    private fun mapFile(f: File): MappedByteBuffer =
        FileInputStream(f).channel.map(FileChannel.MapMode.READ_ONLY, 0, f.length())

    private fun modelsDir(): File = ctx.filesDir

    /**
     * Locate `whisper_*.tflite` files bundled in the APK's assets/ directory
     * and stage them to ctx.filesDir on first access. The TFLite Interpreter
     * mmaps the on-disk file, so we materialize the asset before mapping.
     * The APK ships with `noCompress` for `.tflite` (see build.gradle.kts),
     * so the on-disk copy is byte-identical to the asset.
     *
     * Previously this looked at /sdcard/Android/data/com.ch11.bench/files/models
     * for files pushed via AWS Device Farm's "Upload extra data" feature, but
     * that feature was unreliable on Android 10+ / scoped storage. Embedding
     * the model files directly in the APK assets bypasses delivery quirks.
     */
    private fun locateWhisperTflites(): List<File> {
        val staged = mutableListOf<File>()
        val assetNames = ctx.assets.list("")
            ?.filter { it.lowercase().startsWith("whisper") && it.endsWith(".tflite") }
            .orEmpty().sorted()
        for (assetName in assetNames) {
            val outFile = File(modelsDir(), assetName)
            if (!outFile.exists() || outFile.length() == 0L) {
                onProgress("audio: staging asset $assetName -> ${outFile.absolutePath}")
                ctx.assets.open(assetName).use { input ->
                    outFile.outputStream().use { output -> input.copyTo(output) }
                }
            }
            staged += outFile
        }
        return staged
    }

    /**
     * Map a Whisper TFLite filename to a chapter-canonical variant tag.
     * Examples:
     *   whisper_tiny_encoder_fp32.tflite     -> "tflite_fp32"
     *   whisper_tiny_encoder_int8.tflite     -> "tflite_int8"
     *   whisper_tiny_encoder_dynrange.tflite -> "tflite_dynrange"
     */
    private fun variantTagFromFilename(name: String): String {
        val stem = name.removeSuffix(".tflite").lowercase()
        return when {
            stem.endsWith("_int8") -> "tflite_int8"
            stem.endsWith("_dynrange") -> "tflite_dynrange"
            stem.endsWith("_fp32") || stem.endsWith("_float32") -> "tflite_fp32"
            stem.endsWith("_fp16") || stem.endsWith("_float16") -> "tflite_fp16"
            stem.endsWith("_int16x8") -> "tflite_int16x8"
            else -> "tflite_${stem.substringAfterLast('_')}"
        }
    }

    private fun detectDevice(): DeviceInfo {
        val soc = if (Build.VERSION.SDK_INT >= 31) (Build.SOC_MODEL ?: Build.HARDWARE) else Build.HARDWARE
        return DeviceInfo(
            deviceName = "${Build.MANUFACTURER} ${Build.MODEL}".trim(),
            soc = soc,
            os = "Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})",
            klass = "phone",
        )
    }
}

private enum class AudioBackend(val tag: String) {
    XNNPACK_4T("xnnpack_4t"),
    NNAPI("nnapi"),
    GPU("gpu"),
}
