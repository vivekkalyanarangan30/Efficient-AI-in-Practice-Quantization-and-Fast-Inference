package com.ch11.bench

import android.os.Bundle
import android.util.Log
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import java.io.File

/**
 * Single-activity host that runs the benchmark suite on launch.
 *
 * On launch, the activity:
 *  1. Spawns the [Benchmark] orchestrator on a background dispatcher.
 *  2. Streams per-combo progress lines into the UI TextView for visual debug.
 *  3. Writes the final schema-conformant JSON to [externalOutputFile()].
 *
 * The unattended Espresso test (BenchmarkInstrumentationTest) drives the same
 * path and polls for the output file as the completion signal.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var statusView: TextView
    private var benchmarkJob: Job? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        statusView = TextView(this).apply {
            textSize = 12f
            setPadding(24, 24, 24, 24)
            text = "Initializing…"
        }
        val scroll = ScrollView(this).apply { addView(statusView) }
        setContentView(scroll)

        benchmarkJob = CoroutineScope(Dispatchers.Default).launch {
            try {
                runBenchmark()
            } catch (t: Throwable) {
                Log.e(TAG, "benchmark failed", t)
                updateStatus("FAILED: ${t.javaClass.simpleName}: ${t.message}")
                writeErrorMarker(t)
            }
        }
    }

    private suspend fun runBenchmark() {
        val outFile = externalOutputFile()
        // Remove stale output so completion detection is unambiguous.
        if (outFile.exists()) outFile.delete()

        updateStatus("Running benchmark — output → ${outFile.absolutePath}")
        Log.i(TAG, "output path: ${outFile.absolutePath}")

        val bench = Benchmark(
            ctx = applicationContext,
            onProgress = { line ->
                Log.i(TAG, line)
                updateStatus(line)
            },
        )
        val records = mutableListOf<BenchRecord>()
        records += bench.runAll()

        // Audio (Whisper-tiny encoder) runs after vision. Resilient to a
        // missing whisper_*.tflite file (returns an empty list and continues),
        // so the vision-only path still produces a valid results-android.json.
        try {
            val audio = AudioBenchmark(
                ctx = applicationContext,
                onProgress = { line ->
                    Log.i(TAG, line)
                    updateStatus(line)
                },
            )
            records += audio.runAll()
        } catch (t: Throwable) {
            Log.e(TAG, "audio benchmark failed", t)
            updateStatus("Audio benchmark failed: ${t.javaClass.simpleName}: ${t.message}")
        }

        // LLM benchmark runs after audio. It's resilient to a missing .task
        // model file (returns an empty list and continues), so the vision-only
        // path still produces a valid results-android.json.
        try {
            val llm = LLMBenchmark(
                ctx = applicationContext,
                onProgress = { line ->
                    Log.i(TAG, line)
                    updateStatus(line)
                },
            )
            records += llm.runAll()
        } catch (t: Throwable) {
            Log.e(TAG, "llm benchmark failed", t)
            updateStatus("LLM benchmark failed: ${t.javaClass.simpleName}: ${t.message}")
        }

        // Prepost benchmark runs last. It synthesizes its own inputs (JPEG +
        // PCM) so no model assets are required. Inherits the device fingerprint
        // from the first inference record; if no inference records exist
        // (assets missing), prepost is skipped — the device-info path is the
        // only thing we'd lack, and we don't want to duplicate detectDevice().
        if (records.isNotEmpty()) {
            try {
                val deviceForPrepost = records.first().device
                val prepost = PrepostBenchmark(
                    ctx = applicationContext,
                    onProgress = { line ->
                        Log.i(TAG, line)
                        updateStatus(line)
                    },
                )
                records += prepost.runAll(deviceForPrepost)
            } catch (t: Throwable) {
                Log.e(TAG, "prepost benchmark failed", t)
                updateStatus("Prepost benchmark failed: ${t.javaClass.simpleName}: ${t.message}")
            }
        } else {
            updateStatus("Prepost benchmark skipped: no inference records to inherit device info from")
        }

        val writer = ResultsWriter(outFile)
        writer.write(records)
        updateStatus("DONE — ${records.size} records written to ${outFile.absolutePath}")
        Log.i(TAG, "DONE: wrote ${records.size} records")
    }

    private fun externalOutputFile(): File {
        // Write to *internal* storage (filesDir) rather than external (getExternalFilesDir),
        // so `adb shell run-as com.ch11.bench cat files/results-android.json` in the AWS
        // Device Farm post_test phase can read it. `run-as` resolves `files/` to the
        // internal sandbox at /data/data/<pkg>/files/, not the external scoped dir.
        return File(filesDir, "results-android.json")
    }

    private fun writeErrorMarker(t: Throwable) {
        File(filesDir, "results-android-error.txt").writeText(
            "${t.javaClass.name}: ${t.message}\n${t.stackTraceToString()}\n",
        )
    }

    private fun updateStatus(line: String) {
        runOnUiThread {
            statusView.append("\n$line")
        }
    }

    companion object {
        const val TAG = "Ch11Bench"
    }
}
