package com.ch11.bench

import android.content.Context
import android.os.Build
import android.os.SystemClock
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max

/**
 * Self-contained TFLite benchmark over the chapter's 4 model variants and
 * 4 backends (XNNPACK 1/4 thread, GPU delegate, NNAPI).
 *
 * For each (variant, backend) combination we emit a [BenchRecord] with
 * latency p50/p95/mean from 50 warmup + 200 timed iterations. We also run
 * an on-device top-1/top-5 accuracy pass over 100 ImageNet samples bundled
 * in `assets/imagenet_val_100.bin` and a 300-second sustained loop for
 * the best NNAPI variant.
 *
 * Records are returned as plain dataclasses; the JSON shape is produced
 * by [ResultsWriter] to match chapter schema v11.0 exactly.
 */
class Benchmark(
    private val ctx: Context,
    private val onProgress: (String) -> Unit,
    private val warmupIters: Int = WARMUP_ITERS,
    private val timedIters: Int = TIMED_ITERS,
    private val sustainedWindowS: Int = 300,
    private val powerWindowS: Int = 30,
) {

    fun runAll(): List<BenchRecord> {
        val device = detectDevice()
        onProgress("device: ${device.deviceName} (${device.soc}, ${device.os})")

        val samples = SampleBundle.load(ctx)
        onProgress("samples: ${samples.numSamples} (shape ${samples.shape.contentToString()})")

        val records = mutableListOf<BenchRecord>()

        for (variant in VARIANTS) {
            val modelBuf = loadModelAsset(variant.assetName)
            onProgress("--- ${variant.variantName} (${modelBuf.capacity()} bytes) ---")

            for (backend in BACKENDS) {
                val combo = "${variant.variantName}/${backend.backendName}"
                try {
                    val r = benchmarkCombo(
                        variant = variant,
                        backend = backend,
                        modelBuf = modelBuf,
                        samples = samples,
                        device = device,
                    )
                    records += r
                    onProgress("  $combo: p50=${"%.2f".format(r.latencyP50Ms)}ms  acc=${"%.3f".format(r.accuracyTop1)}")
                } catch (t: Throwable) {
                    onProgress("  $combo: SKIPPED (${t.javaClass.simpleName}: ${t.message})")
                }
            }
        }

        // Sustained + power run on the fastest int8 combination that actually
        // succeeded on this device. Preference order: nnapi → xnnpack_4t →
        // xnnpack_1t → gpu. On Pixel 10 Pro (Tensor G5) NNAPI rejects this
        // model's int8 schema, so we fall through to XNNPACK rather than skipping.
        val baseline = listOf("nnapi", "xnnpack_4t", "xnnpack_1t", "gpu")
            .firstNotNullOfOrNull { cu ->
                records.firstOrNull { it.variant == "tflite_int8" && it.computeUnits == cu }
            }
        if (baseline != null) {
            onProgress("sustained/power baseline: ${baseline.variant}/${baseline.computeUnits} (p50=${"%.2f".format(baseline.latencyP50Ms)}ms)")
            try {
                onProgress("sustained ${sustainedWindowS}s on ${baseline.variant}/${baseline.computeUnits}…")
                records += runSustained(baseline, samples)
            } catch (t: Throwable) {
                onProgress("sustained skipped: ${t.message}")
            }
            try {
                onProgress("power ${powerWindowS}s on ${baseline.variant}/${baseline.computeUnits}…")
                records += runPower(baseline, samples)
            } catch (t: Throwable) {
                onProgress("power skipped: ${t.message}")
            }
        } else {
            onProgress("sustained/power skipped: no working int8 combo")
        }

        return records
    }

    private fun benchmarkCombo(
        variant: Variant,
        backend: Backend,
        modelBuf: MappedByteBuffer,
        samples: SampleBundle,
        device: DeviceInfo,
    ): BenchRecord {
        val (interp, delegateHandle) = newInterpreter(modelBuf, backend)
        try {
            val inputDetails = interp.getInputTensor(0)
            val inputShape = inputDetails.shape()
            val inputType = inputDetails.dataType().name
            val inQuant = inputDetails.quantizationParams()
            val outputDetails = interp.getOutputTensor(0)
            val outputShape = outputDetails.shape()
            val outputType = outputDetails.dataType().name
            val outQuant = outputDetails.quantizationParams()

            val inBuf = allocInput(inputShape, inputType, inQuant.scale, inQuant.zeroPoint, samples)
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

            val (top1, top5) = evaluateAccuracy(
                interp, samples,
                inputShape, inputType, inQuant.scale, inQuant.zeroPoint,
                outputShape, outputType, outQuant.scale, outQuant.zeroPoint,
            )

            return BenchRecord(
                model = "efficientnet_lite0",
                modality = "vision",
                variant = variant.variantName,
                backend = "tflite",
                computeUnits = backend.backendName,
                device = device,
                sizeBytes = variant.sizeBytes(ctx),
                latencyP50Ms = p50,
                latencyP95Ms = p95,
                latencyMeanMs = mean,
                nIters = timedIters,
                warmupIters = warmupIters,
                inputShape = inputShape.toList(),
                throughputSamplesPerSec = 1000.0 / mean,
                accuracyTop1 = top1,
                accuracyTop5 = top5,
                accuracyDataset = "imagenet-1k-val(${samples.numSamples})",
                accuracyN = samples.numSamples,
                kind = RecordKind.LATENCY,
            )
        } finally {
            interp.close()
            (delegateHandle as? AutoCloseable)?.close()
        }
    }

    private fun runSustained(baseline: BenchRecord, samples: SampleBundle): BenchRecord {
        val variant = VARIANTS.first { it.variantName == baseline.variant }
        val backend = BACKENDS.first { it.backendName == baseline.computeUnits }
        val modelBuf = loadModelAsset(variant.assetName)
        val (interp, delegateHandle) = newInterpreter(modelBuf, backend)
        try {
            val inputShape = interp.getInputTensor(0).shape()
            val inputType = interp.getInputTensor(0).dataType().name
            val inQuant = interp.getInputTensor(0).quantizationParams()
            val outputShape = interp.getOutputTensor(0).shape()
            val outputType = interp.getOutputTensor(0).dataType().name
            val inBuf = allocInput(inputShape, inputType, inQuant.scale, inQuant.zeroPoint, samples)
            val outBuf = allocOutput(outputShape, outputType)

            val deadline = SystemClock.elapsedRealtime() + sustainedWindowS * 1000L
            val windowMs = 30_000L
            val windows = mutableListOf<MutableList<Long>>()
            var windowEnd = SystemClock.elapsedRealtime() + windowMs
            var cur = mutableListOf<Long>()

            repeat(5) {
                inBuf.rewind(); outBuf.rewind(); interp.run(inBuf, outBuf)
            }

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
                val p50Ns = sorted[sorted.size / 2]
                return 1e9 / p50Ns
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

    private fun runPower(baseline: BenchRecord, samples: SampleBundle): BenchRecord {
        val variant = VARIANTS.first { it.variantName == baseline.variant }
        val backend = BACKENDS.first { it.backendName == baseline.computeUnits }
        val modelBuf = loadModelAsset(variant.assetName)
        val (interp, delegateHandle) = newInterpreter(modelBuf, backend)
        try {
            val inputShape = interp.getInputTensor(0).shape()
            val inputType = interp.getInputTensor(0).dataType().name
            val inQuant = interp.getInputTensor(0).quantizationParams()
            val outputShape = interp.getOutputTensor(0).shape()
            val outputType = interp.getOutputTensor(0).dataType().name
            val inBuf = allocInput(inputShape, inputType, inQuant.scale, inQuant.zeroPoint, samples)
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
            val meanMw = result.meanPowerMw
            val peakMw = result.peakPowerMw

            return baseline.copy(
                computeUnits = "${baseline.computeUnits}_power_${powerWindowS}s",
                kind = RecordKind.POWER,
                powerMeanMw = meanMw,
                powerPeakMw = peakMw,
                powerSource = result.source,
                powerWindowS = powerWindowS,
                energyPerInferenceMj = if (iters > 0 && !meanMw.isNaN()) meanMw * elapsedSec / iters else null,
            )
        } finally {
            interp.close()
            (delegateHandle as? AutoCloseable)?.close()
        }
    }

    private fun newInterpreter(modelBuf: MappedByteBuffer, backend: Backend): Pair<Interpreter, Any?> {
        val opts = Interpreter.Options()
        var handle: Any? = null
        when (backend) {
            Backend.XNNPACK_1T -> { opts.setNumThreads(1).setUseXNNPACK(true) }
            Backend.XNNPACK_4T -> { opts.setNumThreads(4).setUseXNNPACK(true) }
            Backend.GPU -> {
                val gpu = GpuDelegate()
                handle = gpu
                opts.addDelegate(gpu)
            }
            Backend.NNAPI -> {
                val nnapi = NnApiDelegate()
                handle = nnapi
                opts.addDelegate(nnapi)
            }
        }
        return Interpreter(modelBuf, opts) to handle
    }

    private fun allocInput(
        shape: IntArray,
        dtype: String,
        scale: Float,
        zeroPoint: Int,
        samples: SampleBundle,
    ): ByteBuffer {
        val elements = shape.fold(1) { a, b -> a * b }
        val bytesPerElem = when (dtype) { "FLOAT32" -> 4; "INT16" -> 2; "UINT8", "INT8" -> 1; else -> 4 }
        val buf = ByteBuffer.allocateDirect(elements * bytesPerElem).order(ByteOrder.nativeOrder())
        // Sample 0 is canonical for warmup + latency loops. evaluateAccuracy fills per-sample separately.
        fillInputFromSample(buf, samples.rawSample(0), dtype, scale, zeroPoint)
        buf.rewind()
        return buf
    }

    private fun allocOutput(shape: IntArray, dtype: String): ByteBuffer {
        val elements = shape.fold(1) { a, b -> a * b }
        val bytesPerElem = when (dtype) { "FLOAT32" -> 4; "INT16" -> 2; "UINT8", "INT8" -> 1; else -> 4 }
        return ByteBuffer.allocateDirect(elements * bytesPerElem).order(ByteOrder.nativeOrder())
    }

    /**
     * Convert one uint8 image into the model's input dtype, applying
     * quantization params for int8/int16. Matches the Python reference in
     * `ch11_2_tflite.py verify-accuracy` (line ~422-427): float in [0,1]
     * then `clip(round(f/scale + zp))`.
     */
    private fun fillInputFromSample(
        dst: ByteBuffer,
        sampleBytes: ByteArray,
        dtype: String,
        scale: Float,
        zeroPoint: Int,
    ) {
        dst.rewind()
        when (dtype) {
            "FLOAT32" -> {
                for (b in sampleBytes) {
                    val f = (b.toInt() and 0xff) / 255.0f
                    dst.putFloat(f)
                }
            }
            "UINT8" -> {
                // Most uint8-input models treat input as raw pixels.
                dst.put(sampleBytes)
            }
            "INT8" -> {
                // EfficientNet-Lite0 int8: input scale ≈ 1/128, zp = -128 in Python reference.
                val s = if (scale == 0f) (1.0f / 128.0f) else scale
                val zp = if (scale == 0f) -128 else zeroPoint
                for (b in sampleBytes) {
                    val f = (b.toInt() and 0xff) / 255.0f
                    val q = Math.round(f / s + zp).coerceIn(-128, 127)
                    dst.put(q.toByte())
                }
            }
            "INT16" -> {
                val s = if (scale == 0f) (1.0f / 32768.0f) else scale
                val zp = if (scale == 0f) 0 else zeroPoint
                for (b in sampleBytes) {
                    val f = (b.toInt() and 0xff) / 255.0f
                    val q = Math.round(f / s + zp).coerceIn(-32768, 32767)
                    dst.putShort(q.toShort())
                }
            }
            else -> {
                // Unknown dtype: fall back to float32-style write; runtime will error if shape/dtype mismatch.
                for (b in sampleBytes) dst.putFloat((b.toInt() and 0xff) / 255.0f)
            }
        }
    }

    private fun evaluateAccuracy(
        interp: Interpreter,
        samples: SampleBundle,
        inputShape: IntArray,
        inputType: String,
        inScale: Float,
        inZeroPoint: Int,
        outputShape: IntArray,
        outputType: String,
        outScale: Float,
        outZeroPoint: Int,
    ): Pair<Double, Double> {
        val inBuf = allocOutput(inputShape, inputType)
        val outBuf = allocOutput(outputShape, outputType)
        var top1 = 0
        var top5 = 0
        for (i in 0 until samples.numSamples) {
            fillInputFromSample(inBuf, samples.rawSample(i), inputType, inScale, inZeroPoint)
            inBuf.rewind(); outBuf.rewind()
            interp.run(inBuf, outBuf)
            val logits = readLogits(outBuf, outputShape, outputType, outScale, outZeroPoint)
            val label = samples.labelAt(i)
            val topK = logits.indices.sortedByDescending { logits[it] }.take(5)
            if (topK.firstOrNull() == label) top1++
            if (label in topK) top5++
        }
        return Pair(top1.toDouble() / samples.numSamples, top5.toDouble() / samples.numSamples)
    }

    private fun readLogits(
        buf: ByteBuffer,
        shape: IntArray,
        dtype: String,
        scale: Float,
        zeroPoint: Int,
    ): FloatArray {
        val n = shape.fold(1) { a, b -> a * b }
        buf.rewind()
        return when (dtype) {
            "FLOAT32" -> FloatArray(n).also { for (i in 0 until n) it[i] = buf.float }
            "UINT8" -> FloatArray(n).also {
                for (i in 0 until n) {
                    val q = (buf.get().toInt() and 0xff)
                    it[i] = (q - zeroPoint) * scale
                }
            }
            "INT8" -> FloatArray(n).also {
                for (i in 0 until n) {
                    val q = buf.get().toInt()
                    it[i] = (q - zeroPoint) * scale
                }
            }
            "INT16" -> FloatArray(n).also {
                for (i in 0 until n) {
                    val q = buf.short.toInt()
                    it[i] = (q - zeroPoint) * scale
                }
            }
            else -> FloatArray(n).also { for (i in 0 until n) it[i] = buf.float }
        }
    }

    private fun loadModelAsset(name: String): MappedByteBuffer {
        val outFile = File(ctx.filesDir, name)
        if (!outFile.exists() || outFile.length() == 0L) {
            ctx.assets.open(name).use { input ->
                outFile.outputStream().use { output -> input.copyTo(output) }
            }
        }
        return FileInputStream(outFile).channel.map(FileChannel.MapMode.READ_ONLY, 0, outFile.length())
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

    enum class Backend(val backendName: String) {
        XNNPACK_1T("xnnpack_1t"),
        XNNPACK_4T("xnnpack_4t"),
        GPU("gpu"),
        NNAPI("nnapi"),
    }

    data class Variant(val variantName: String, val assetName: String) {
        fun sizeBytes(ctx: Context): Long? {
            return try {
                ctx.assets.openFd(assetName).use { it.length }
            } catch (e: Exception) {
                try {
                    val tmp = ByteArrayOutputStream()
                    ctx.assets.open(assetName).use { it.copyTo(tmp) }
                    tmp.size().toLong()
                } catch (e2: Exception) {
                    null
                }
            }
        }
    }

    companion object {
        private const val WARMUP_ITERS = 50
        private const val TIMED_ITERS = 200

        val VARIANTS = listOf(
            Variant("tflite_fp32",     "effnet_lite0_fp32.tflite"),
            Variant("tflite_dynrange", "effnet_lite0_dynrange.tflite"),
            Variant("tflite_int8",     "effnet_lite0_int8.tflite"),
            Variant("tflite_int16x8",  "effnet_lite0_int16x8.tflite"),
        )
        val BACKENDS = Backend.values().toList()
    }
}

enum class RecordKind { LATENCY, SUSTAINED, POWER, LLM_LATENCY, LLM_SUSTAINED, LLM_POWER, LLM_ACCURACY }

data class DeviceInfo(
    val deviceName: String,
    val soc: String,
    val os: String,
    val klass: String,
)

data class BenchRecord(
    val model: String,
    val modality: String,
    val variant: String,
    val backend: String,
    val computeUnits: String,
    val device: DeviceInfo,
    val sizeBytes: Long?,
    val latencyP50Ms: Double,
    val latencyP95Ms: Double,
    val latencyMeanMs: Double,
    val nIters: Int,
    val warmupIters: Int,
    val inputShape: List<Int>,
    val throughputSamplesPerSec: Double,
    val accuracyTop1: Double,
    val accuracyTop5: Double,
    val accuracyDataset: String,
    val accuracyN: Int,
    val kind: RecordKind,
    val sustainedWindowS: Int? = null,
    val throughputFirst30s: Double? = null,
    val throughputLast30s: Double? = null,
    val thermalPressureObserved: Boolean? = null,
    val powerMeanMw: Double? = null,
    val powerPeakMw: Double? = null,
    val powerSource: String? = null,
    val powerWindowS: Int? = null,
    val energyPerInferenceMj: Double? = null,
    // LLM-only fields. Null for vision/audio rows so the JSON omits the keys.
    val throughputTokensPerSec: Double? = null,
    val promptLength: Int? = null,
    val generationLength: Int? = null,
    val ttftMs: Double? = null,
    val tpotMs: Double? = null,
    val workingMemoryMb: Double? = null,
    val accuracyMetric: String? = null,
    val accuracyValue: Double? = null,
    // Prepost-only fields (PrepostBenchmark). Null for inference rows so the
    // JSON's prepost object stays null unless the record specifically measured
    // a pre/post-processing stage.
    val prepostDecodeMs: Double? = null,
    val prepostResizeMs: Double? = null,
    val prepostNormalizeMs: Double? = null,
    val prepostLogmelMs: Double? = null,
    val prepostTokenizeMs: Double? = null,
    val prepostDetokenizeMs: Double? = null,
)
