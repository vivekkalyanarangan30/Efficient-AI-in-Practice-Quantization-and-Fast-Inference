package com.ch11.bench

import android.content.Context
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.DataInputStream

/**
 * Loads the 100-image ImageNet validation bundle baked into the APK.
 *
 * On-disk format (`assets/imagenet_val_100.bin`), all integers little-endian:
 *   - magic (4 bytes ASCII): "CH11"
 *   - num_samples (int32 LE)
 *   - height (int32 LE)
 *   - width (int32 LE)
 *   - channels (int32 LE)
 *   - pixel data: N * H * W * C uint8 bytes (NHWC, EfficientNet input ordering)
 *
 * Labels are in `assets/labels.json`: {"labels": [int, ...]} with one
 * ground-truth class id per sample (ImageNet-1k class ids 0..999).
 *
 * The bundle stores uint8 [0,255] pixels; per-iteration we convert into
 * the model's expected dtype (fp32 normalized to [0,1], or int8/int16 via
 * model quantization params). The conversion lives in [Benchmark] which
 * has the [org.tensorflow.lite.Tensor.QuantizationParams] in scope.
 */
class SampleBundle(
    val numSamples: Int,
    val shape: IntArray, // [H, W, C]
    private val pixels: ByteArray,
    private val labels: IntArray,
) {

    val pixelsPerSample: Int get() = shape[0] * shape[1] * shape[2]

    fun labelAt(i: Int): Int = labels[i]

    /** Raw uint8 pixels [0..255] for sample [index], length = H*W*C. */
    fun rawSample(index: Int): ByteArray {
        val per = pixelsPerSample
        val out = ByteArray(per)
        System.arraycopy(pixels, index * per, out, 0, per)
        return out
    }

    companion object {

        /** Little-endian int reader (DataInputStream is big-endian). */
        private fun DataInputStream.readIntLE(): Int = Integer.reverseBytes(this.readInt())

        fun load(ctx: Context): SampleBundle {
            val raw = ctx.assets.open("imagenet_val_100.bin").use { input ->
                val all = ByteArrayOutputStream()
                input.copyTo(all)
                all.toByteArray()
            }
            require(raw.size >= 20) { "imagenet_val_100.bin too small: ${raw.size} bytes" }
            DataInputStream(raw.inputStream()).use { d ->
                val magic = ByteArray(4).also { d.readFully(it) }
                require(magic.toString(Charsets.US_ASCII) == "CH11") {
                    "imagenet_val_100.bin magic mismatch: got ${magic.toString(Charsets.US_ASCII)}"
                }
                val n = d.readIntLE()
                val h = d.readIntLE()
                val w = d.readIntLE()
                val c = d.readIntLE()
                val expected = n * h * w * c
                val pixels = ByteArray(expected)
                d.readFully(pixels)

                val json = ctx.assets.open("labels.json").bufferedReader().use { it.readText() }
                val labelsArr = JSONObject(json).getJSONArray("labels")
                require(labelsArr.length() == n) {
                    "labels.json count ${labelsArr.length()} != sample count $n"
                }
                val labels = IntArray(n) { labelsArr.getInt(it) }
                return SampleBundle(n, intArrayOf(h, w, c), pixels, labels)
            }
        }
    }
}
