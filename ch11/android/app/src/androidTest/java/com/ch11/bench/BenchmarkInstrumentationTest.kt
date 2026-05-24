package com.ch11.bench

import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Espresso instrumentation test that launches [MainActivity] and polls the
 * external files dir for the results JSON. AWS Device Farm Test Run uses
 * this as the entry point so the full benchmark runs unattended.
 *
 * The test only validates that the benchmark completes within a hard budget;
 * the actual data correctness is enforced by the host-side schema validator
 * after the artifact is downloaded.
 */
@RunWith(AndroidJUnit4::class)
class BenchmarkInstrumentationTest {

    @Test
    fun runFullBenchmark() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        // Internal storage so AWS Device Farm post_test phase (run-as) can pull it.
        val outFile = File(ctx.filesDir, "results-android.json")
        val errorFile = File(ctx.filesDir, "results-android-error.txt")
        if (outFile.exists()) outFile.delete()
        if (errorFile.exists()) errorFile.delete()

        // Launch the activity; it kicks off the benchmark coroutine on Dispatchers.Default.
        ActivityScenario.launch(MainActivity::class.java).use {
            val deadline = System.currentTimeMillis() + BENCHMARK_DEADLINE_MS
            while (System.currentTimeMillis() < deadline) {
                if (errorFile.exists()) {
                    val err = errorFile.readText()
                    throw AssertionError("Benchmark reported failure:\n$err")
                }
                if (outFile.exists() && outFile.length() > 0) {
                    break
                }
                Thread.sleep(POLL_INTERVAL_MS)
            }
            assertTrue(
                "Benchmark did not produce $outFile within ${BENCHMARK_DEADLINE_MS / 1000}s",
                outFile.exists() && outFile.length() > 0,
            )
        }
    }

    companion object {
        // Full matrix:
        //   Vision: 4 variants × 4 backends × (200 timed + 50 warmup + 100 acc)
        //           + 300s sustained + 30s power (≈ 22 min on Pixel 10 Pro).
        //   LLM:    1+ .task variants × 3 prompt lengths × generate(64) × 3 runs
        //           + 200-item HellaSwag pass (≈ 2 sec/item)
        //           + 300s sustained + 30s power (≈ 30–40 min).
        // 75 minutes leaves margin for cold-start and host overhead.
        private const val BENCHMARK_DEADLINE_MS = 75 * 60 * 1000L
        private const val POLL_INTERVAL_MS = 5_000L
    }
}
