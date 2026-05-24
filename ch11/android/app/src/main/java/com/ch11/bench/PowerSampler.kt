package com.ch11.bench

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.SystemClock
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.max

/**
 * Crude on-device power sampler driven by [BatteryManager].
 *
 * Preferred path: BATTERY_PROPERTY_ENERGY_COUNTER (long, nWh) sampled
 * at start and end → ΔE / Δt → mean power. This avoids voltage-sign and
 * vendor-disagreement on CURRENT_NOW signs.
 *
 * Fallback path: BATTERY_PROPERTY_CURRENT_NOW (µA) × voltage from the
 * ACTION_BATTERY_CHANGED sticky broadcast (mV), sampled at [hz] Hz. We take
 * |I| because OEMs disagree on sign convention for charging vs draw.
 *
 * Caveat: ±20% accuracy at best. On AWS Device Farm the phone is plugged
 * in, so the reading reflects charge current rather than discharge — we
 * surface this in the source field so the reader sees the limitation.
 */
class PowerSampler(
    private val ctx: Context,
    private val hz: Int,
) {
    private val bm: BatteryManager =
        ctx.applicationContext.getSystemService(Context.BATTERY_SERVICE) as BatteryManager

    private var scheduler: ScheduledExecutorService? = null
    private var task: ScheduledFuture<*>? = null
    private val samplesMw = mutableListOf<Double>()
    private var energyStartNwh: Long = Long.MIN_VALUE
    private var t0Ms: Long = 0
    private var energyAvailable: Boolean = false

    fun start() {
        samplesMw.clear()
        t0Ms = SystemClock.elapsedRealtime()

        // Try energy counter first.
        val e0 = try {
            bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER)
        } catch (_: Throwable) {
            Long.MIN_VALUE
        }
        energyAvailable = e0 != Long.MIN_VALUE && e0 != 0L
        energyStartNwh = e0

        // Always start fallback sampler too — it may give finer-grained samples even
        // when energy counter exists, and is a backup if the counter doesn't update.
        scheduler = Executors.newSingleThreadScheduledExecutor()
        val periodMs = (1000L / max(1, hz))
        task = scheduler?.scheduleAtFixedRate({
            try {
                val mw = sampleNowFallback()
                if (!mw.isNaN()) {
                    synchronized(samplesMw) { samplesMw += mw }
                }
            } catch (_: Throwable) {
                // Sampling fall-through.
            }
        }, 0L, periodMs, TimeUnit.MILLISECONDS)
    }

    fun stop(): Result {
        task?.cancel(false)
        scheduler?.shutdown()
        scheduler?.awaitTermination(2, TimeUnit.SECONDS)
        scheduler = null
        task = null
        val elapsedSec = (SystemClock.elapsedRealtime() - t0Ms) / 1000.0

        // Prefer energy counter (more accurate than I*V integration).
        if (energyAvailable) {
            val e1 = try {
                bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER)
            } catch (_: Throwable) { Long.MIN_VALUE }
            if (e1 != Long.MIN_VALUE && elapsedSec > 0.0) {
                // |Δnwh| / hours -> mW. 1 nWh / 1h = 1e-9 / 3600 = 2.778e-13 W -> mW: 2.778e-10
                // Simpler: ΔnWh * 3.6 / Δs gives μJ/s = μW; divide by 1000 -> mW.
                // ΔnWh -> nJ: ΔnWh * 3.6  (since 1 Wh = 3600 J, 1 nWh = 3.6 nJ)
                // Mean mW = ΔnJ / Δs / 1e6
                val deltaNj = abs(e1 - energyStartNwh).toDouble() * 3.6
                val meanMw = deltaNj / elapsedSec / 1_000_000.0
                // Peak from fallback (if any); else NaN.
                val peak = synchronized(samplesMw) { samplesMw.maxOrNull() ?: Double.NaN }
                if (meanMw > 0.0 && meanMw.isFinite()) {
                    return Result(
                        meanPowerMw = meanMw,
                        peakPowerMw = peak,
                        source = "BatteryManager.ENERGY_COUNTER (±20%)",
                    )
                }
            }
        }

        val snapshot: List<Double> = synchronized(samplesMw) { samplesMw.toList() }
        if (snapshot.isEmpty()) {
            return Result(meanPowerMw = Double.NaN, peakPowerMw = Double.NaN, source = "battery_manager_unavailable")
        }
        return Result(
            meanPowerMw = snapshot.average(),
            peakPowerMw = snapshot.maxOrNull() ?: Double.NaN,
            source = "BatteryManager I*V fallback (±20%)",
        )
    }

    private fun sampleNowFallback(): Double {
        val currentUa = try {
            bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
        } catch (_: Throwable) { Long.MIN_VALUE }
        if (currentUa == Long.MIN_VALUE) return Double.NaN

        // Voltage comes from the sticky ACTION_BATTERY_CHANGED broadcast.
        val intent: Intent? = ctx.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val voltageMv = intent?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, -1) ?: -1
        if (voltageMv <= 0) return Double.NaN

        // |I| µA × V mV → nW; /1e6 → mW.
        val absUa = if (currentUa < 0) -currentUa else currentUa
        return absUa.toDouble() * voltageMv.toDouble() / 1_000_000.0
    }

    data class Result(val meanPowerMw: Double, val peakPowerMw: Double, val source: String)
}
