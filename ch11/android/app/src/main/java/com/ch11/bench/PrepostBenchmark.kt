package com.ch11.bench

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.SystemClock
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * Pre- and post-processing stage timing on the Pixel device — populates the
 * `prepost.*` fields of records that otherwise leave them null. The Mac
 * side already produces equivalent prepost records via ch11_5_prepost.py;
 * this module is the §11.5 Android counterpart so the chapter can compare
 * decode / resize / normalize / log-mel costs across platforms.
 *
 * Synthesis strategy: we don't bundle a representative JPEG / WAV corpus
 * (it would balloon the APK). Instead, inputs are deterministically
 * synthesized once at start (seed-driven), then the prepost stages run
 * against in-memory buffers — exactly the surface real apps exercise.
 *
 *   Vision (EfficientNet-Lite0): synthesize a 320×320 RGB pixel array
 *     (deterministic noisy gradient), encode once to JPEG at quality=85,
 *     then per-iteration time:
 *       decode_ms     : BitmapFactory.decodeByteArray on the JPEG bytes
 *       resize_ms     : Bitmap.createScaledBitmap to 224×224, FILTER_BILINEAR
 *       normalize_ms  : fill an fp32 [1,224,224,3] direct ByteBuffer with
 *                       per-channel zero-centered float values
 *
 *   Audio  (Whisper-tiny):     synthesize 30 s @ 16 kHz of PCM samples
 *     (deterministic sin-sweep + noise), then per-iteration time:
 *       logmel_ms     : Hann-windowed STFT (n_fft=400, hop=160) +
 *                       80-band log-mel projection — matches the
 *                       Whisper preprocessing the Mac side measures.
 *
 *   LLM    (Llama-3.2-1B):     LiteRT-LM 0.12.0 does not expose the
 *     tokenizer as a public API surface separate from `Engine` /
 *     `Conversation`, so tokenize/detokenize timing on Android is not
 *     measurable without scraping internals — flagged in caveats and
 *     deferred to a future LiteRT-LM API surface.
 */
class PrepostBenchmark(
    private val ctx: Context,
    private val onProgress: (String) -> Unit,
    private val iters: Int = 50,
    private val warmupIters: Int = 10,
) {

    fun runAll(device: DeviceInfo): List<BenchRecord> {
        val records = mutableListOf<BenchRecord>()

        onProgress("--- prepost: vision (EfficientNet-Lite0) ---")
        try {
            records += benchVisionPrepost(device)
        } catch (t: Throwable) {
            onProgress("  vision prepost: SKIPPED (${t.javaClass.simpleName}: ${t.message})")
        }

        onProgress("--- prepost: audio (Whisper-tiny) ---")
        try {
            records += benchAudioPrepost(device)
        } catch (t: Throwable) {
            onProgress("  audio prepost: SKIPPED (${t.javaClass.simpleName}: ${t.message})")
        }

        return records
    }

    // ----------------------------------------------------------------------- //
    // Vision prepost                                                          //
    // ----------------------------------------------------------------------- //
    private fun benchVisionPrepost(device: DeviceInfo): BenchRecord {
        val srcW = 320
        val srcH = 320
        val jpegBytes = synthesizeJpeg(srcW, srcH, seed = SEED_VISION)
        onProgress("  vision: jpeg synth = ${jpegBytes.size} bytes (q=85)")

        val decodeMs = DoubleArray(iters)
        val resizeMs = DoubleArray(iters)
        val normalizeMs = DoubleArray(iters)

        // Pre-allocate the normalize destination once — the per-iteration cost
        // we want is *fill*, not allocate.
        val normBuf = ByteBuffer.allocateDirect(1 * 224 * 224 * 3 * 4)
            .order(ByteOrder.nativeOrder())

        repeat(warmupIters) {
            val bm = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
                ?: error("warmup decode returned null")
            val sm = Bitmap.createScaledBitmap(bm, 224, 224, /* filter = */ true)
            fillNormalize(sm, normBuf)
            bm.recycle()
            if (sm !== bm) sm.recycle()
        }

        for (i in 0 until iters) {
            val t0 = SystemClock.elapsedRealtimeNanos()
            val bm = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
                ?: error("decode returned null at iter $i")
            val t1 = SystemClock.elapsedRealtimeNanos()
            val sm = Bitmap.createScaledBitmap(bm, 224, 224, true)
            val t2 = SystemClock.elapsedRealtimeNanos()
            fillNormalize(sm, normBuf)
            val t3 = SystemClock.elapsedRealtimeNanos()

            decodeMs[i] = (t1 - t0) / 1_000_000.0
            resizeMs[i] = (t2 - t1) / 1_000_000.0
            normalizeMs[i] = (t3 - t2) / 1_000_000.0

            bm.recycle()
            if (sm !== bm) sm.recycle()
        }

        val decodeMean = decodeMs.average()
        val resizeMean = resizeMs.average()
        val normalizeMean = normalizeMs.average()
        val totalP50 = percentile(
            DoubleArray(iters) { decodeMs[it] + resizeMs[it] + normalizeMs[it] },
            50.0,
        )
        onProgress(
            "  vision prepost: decode=${"%.2f".format(decodeMean)}ms" +
                "  resize=${"%.2f".format(resizeMean)}ms" +
                "  normalize=${"%.2f".format(normalizeMean)}ms",
        )

        return BenchRecord(
            model = "efficientnet_lite0",
            modality = "vision",
            variant = "android_decode_bitmapfactory_resize_canvas_normalize_loop",
            backend = "prepost",
            computeUnits = "cpu",
            device = device,
            sizeBytes = null,
            latencyP50Ms = totalP50,
            latencyP95Ms = percentile(
                DoubleArray(iters) { decodeMs[it] + resizeMs[it] + normalizeMs[it] }, 95.0,
            ),
            latencyMeanMs = decodeMean + resizeMean + normalizeMean,
            nIters = iters,
            warmupIters = warmupIters,
            inputShape = listOf(1, 224, 224, 3),
            throughputSamplesPerSec = 1000.0 / (decodeMean + resizeMean + normalizeMean),
            accuracyTop1 = Double.NaN,
            accuracyTop5 = Double.NaN,
            accuracyDataset = "synthetic_jpeg",
            accuracyN = 0,
            kind = RecordKind.LATENCY,
            prepostDecodeMs = decodeMean,
            prepostResizeMs = resizeMean,
            prepostNormalizeMs = normalizeMean,
        )
    }

    private fun synthesizeJpeg(w: Int, h: Int, seed: Long): ByteArray {
        // Deterministic noisy gradient — produces a JPEG of realistic compressibility
        // (mid-entropy). Hand-rolled so we don't depend on assets on disk.
        val pixels = IntArray(w * h)
        var s = seed
        for (y in 0 until h) {
            for (x in 0 until w) {
                s = s * 6364136223846793005L + 1442695040888963407L
                val noise = ((s ushr 33).toInt() and 0x1F)
                val r = ((x * 255) / w + noise).coerceIn(0, 255)
                val g = ((y * 255) / h + noise).coerceIn(0, 255)
                val b = (((x + y) * 255) / (w + h) + noise).coerceIn(0, 255)
                pixels[y * w + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        val bm = Bitmap.createBitmap(pixels, w, h, Bitmap.Config.ARGB_8888)
        val baos = ByteArrayOutputStream()
        bm.compress(Bitmap.CompressFormat.JPEG, 85, baos)
        bm.recycle()
        return baos.toByteArray()
    }

    private fun fillNormalize(src: Bitmap, dst: ByteBuffer) {
        // Standard ImageNet-ish normalization: x/255 - mean / std, channel-major
        // NHWC fp32. Per-pixel; mirrors the Mac script's `normalize_vec` path.
        dst.rewind()
        val w = src.width
        val h = src.height
        val pixels = IntArray(w * h)
        src.getPixels(pixels, 0, w, 0, 0, w, h)
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val stdInv = floatArrayOf(1f / 0.229f, 1f / 0.224f, 1f / 0.225f)
        for (i in 0 until w * h) {
            val p = pixels[i]
            val r = ((p shr 16) and 0xFF) / 255f
            val g = ((p shr 8) and 0xFF) / 255f
            val b = (p and 0xFF) / 255f
            dst.putFloat((r - mean[0]) * stdInv[0])
            dst.putFloat((g - mean[1]) * stdInv[1])
            dst.putFloat((b - mean[2]) * stdInv[2])
        }
    }

    // ----------------------------------------------------------------------- //
    // Audio prepost                                                           //
    // ----------------------------------------------------------------------- //
    private fun benchAudioPrepost(device: DeviceInfo): BenchRecord {
        val sampleRate = 16000
        val durationS = 30
        val nSamples = sampleRate * durationS  // 480_000
        val pcm = synthesizePcm(nSamples, seed = SEED_AUDIO)

        // Whisper's reference STFT uses n_fft=400 (not a power of 2). Real
        // production implementations zero-pad each frame to the next power of
        // two (512) and use radix-2 FFT — a 400-point Bluestein would be
        // ~3-4× slower for marginal spectral difference at the mel-band
        // resolution we care about. We follow the production convention:
        // pad-to-512 + radix-2 FFT, then project to 80 mel bands.
        val frameLen = 400
        val nFft = 512  // next pow-2 ≥ frameLen
        val hop = 160
        val nMels = 80
        val window = hannWindow(frameLen)
        val melFb = buildMelFilterbank(nMels = nMels, nFft = nFft, sampleRate = sampleRate)

        val logmelMs = DoubleArray(iters)
        repeat(warmupIters) {
            computeLogMel(pcm, window, frameLen, nFft, hop, melFb)
        }
        for (i in 0 until iters) {
            val t0 = SystemClock.elapsedRealtimeNanos()
            val mel = computeLogMel(pcm, window, frameLen, nFft, hop, melFb)
            val t1 = SystemClock.elapsedRealtimeNanos()
            // Touch the output so JIT/AOT can't dead-code-eliminate the call.
            if (mel.isEmpty() || mel[0].isNaN()) error("mel returned empty/NaN")
            logmelMs[i] = (t1 - t0) / 1_000_000.0
        }
        val mean = logmelMs.average()
        onProgress("  audio prepost: logmel=${"%.2f".format(mean)}ms (frame=400 zero-pad=512 hop=160 mels=80)")

        return BenchRecord(
            model = "whisper_tiny",
            modality = "audio",
            variant = "android_logmel_kotlin_stft",
            backend = "prepost",
            computeUnits = "cpu",
            device = device,
            sizeBytes = null,
            latencyP50Ms = percentile(logmelMs, 50.0),
            latencyP95Ms = percentile(logmelMs, 95.0),
            latencyMeanMs = mean,
            nIters = iters,
            warmupIters = warmupIters,
            inputShape = listOf(1, 80, 3000),
            throughputSamplesPerSec = 1000.0 / mean,
            accuracyTop1 = Double.NaN,
            accuracyTop5 = Double.NaN,
            accuracyDataset = "synthetic_pcm_30s_16khz",
            accuracyN = 0,
            kind = RecordKind.LATENCY,
            prepostLogmelMs = mean,
        )
    }

    private fun synthesizePcm(n: Int, seed: Long): FloatArray {
        // Deterministic sin-sweep + small noise; replaces a real WAV decode
        // step (cheap and consistent across runs).
        val out = FloatArray(n)
        var s = seed
        var phase = 0.0
        var f = 100.0
        for (i in 0 until n) {
            phase += 2 * Math.PI * f / 16000.0
            f += 0.0005
            if (f > 4000.0) f = 100.0
            s = s * 6364136223846793005L + 1442695040888963407L
            val noise = ((s ushr 40).toInt() and 0xFFFF) / 65535f - 0.5f
            out[i] = (0.6f * sin(phase).toFloat()) + 0.05f * noise
        }
        return out
    }

    private fun hannWindow(n: Int): FloatArray {
        val w = FloatArray(n)
        for (i in 0 until n) {
            w[i] = 0.5f - 0.5f * cos(2.0 * Math.PI * i / (n - 1)).toFloat()
        }
        return w
    }

    private fun buildMelFilterbank(nMels: Int, nFft: Int, sampleRate: Int): Array<FloatArray> {
        // Standard log-mel filterbank (Slaney convention). Hand-rolled —
        // avoids pulling a TF-Lite STFT subgraph just for a one-shot
        // preprocessing timer.
        val fMin = 0.0
        val fMax = sampleRate / 2.0
        fun hzToMel(hz: Double) = 2595.0 * Math.log10(1 + hz / 700.0)
        fun melToHz(m: Double) = 700.0 * (Math.pow(10.0, m / 2595.0) - 1)
        val melMin = hzToMel(fMin)
        val melMax = hzToMel(fMax)
        val melPoints = DoubleArray(nMels + 2) { melMin + (melMax - melMin) * it / (nMels + 1) }
        val hzPoints = DoubleArray(nMels + 2) { melToHz(melPoints[it]) }
        val nBins = nFft / 2 + 1
        val binFreqs = DoubleArray(nBins) { it * sampleRate.toDouble() / nFft }

        val fb = Array(nMels) { FloatArray(nBins) }
        for (m in 0 until nMels) {
            val lo = hzPoints[m]
            val mid = hzPoints[m + 1]
            val hi = hzPoints[m + 2]
            for (k in 0 until nBins) {
                val f = binFreqs[k]
                val up = (f - lo) / (mid - lo)
                val down = (hi - f) / (hi - mid)
                val v = max(0.0, min(up, down))
                fb[m][k] = v.toFloat()
            }
        }
        return fb
    }

    private fun computeLogMel(
        pcm: FloatArray,
        window: FloatArray,
        frameLen: Int,
        nFft: Int,
        hop: Int,
        melFb: Array<FloatArray>,
    ): FloatArray {
        val nFrames = max(0, (pcm.size - frameLen) / hop + 1)
        val nBins = nFft / 2 + 1
        val nMels = melFb.size
        // Per-frame buffers, sized for the zero-padded FFT length.
        val real = FloatArray(nFft)
        val imag = FloatArray(nFft)
        val mel = FloatArray(nFrames * nMels)
        for (frame in 0 until nFrames) {
            val off = frame * hop
            // Load windowed frame into real, clear imag. Zero-pad the tail
            // (frameLen..nFft-1) implicitly by overwriting then zeroing.
            for (n in 0 until frameLen) real[n] = pcm[off + n] * window[n]
            for (n in frameLen until nFft) real[n] = 0f
            for (n in 0 until nFft) imag[n] = 0f
            fftRadix2InPlace(real, imag, nFft)
            for (m in 0 until nMels) {
                var acc = 0.0f
                val row = melFb[m]
                for (k in 0 until nBins) {
                    val mag = real[k] * real[k] + imag[k] * imag[k]
                    acc += row[k] * mag
                }
                mel[frame * nMels + m] = ln(max(1e-10f, acc))
            }
        }
        return mel
    }

    /** In-place radix-2 Cooley-Tukey FFT. `n` must be a power of two. */
    private fun fftRadix2InPlace(real: FloatArray, imag: FloatArray, n: Int) {
        // Bit-reverse permutation.
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j xor bit
            if (i < j) {
                val tr = real[i]; real[i] = real[j]; real[j] = tr
                val ti = imag[i]; imag[i] = imag[j]; imag[j] = ti
            }
        }
        // Butterflies, length 2 → 4 → … → n.
        var len = 2
        while (len <= n) {
            val angle = -2.0 * Math.PI / len
            val wlenR = cos(angle).toFloat()
            val wlenI = sin(angle).toFloat()
            var i = 0
            while (i < n) {
                var wr = 1.0f
                var wi = 0.0f
                val half = len / 2
                for (k in 0 until half) {
                    val ur = real[i + k]
                    val ui = imag[i + k]
                    val vr = real[i + k + half] * wr - imag[i + k + half] * wi
                    val vi = real[i + k + half] * wi + imag[i + k + half] * wr
                    real[i + k] = ur + vr
                    imag[i + k] = ui + vi
                    real[i + k + half] = ur - vr
                    imag[i + k + half] = ui - vi
                    val nwr = wr * wlenR - wi * wlenI
                    wi = wr * wlenI + wi * wlenR
                    wr = nwr
                }
                i += len
            }
            len = len shl 1
        }
    }

    private fun percentile(xs: DoubleArray, p: Double): Double {
        val sorted = xs.copyOf().also { it.sort() }
        val idx = ((p / 100.0) * (sorted.size - 1)).toInt().coerceIn(0, sorted.size - 1)
        return sorted[idx]
    }

    companion object {
        private const val SEED_VISION = 0xC11AB1L
        private const val SEED_AUDIO = 0xC11AB2L
    }
}
