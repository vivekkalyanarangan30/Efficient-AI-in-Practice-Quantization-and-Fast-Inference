plugins {
    id("com.android.application") version "8.5.2" apply false
    // Kotlin 2.2.x: required because litertlm-android 0.12.0 transitively
    // pulls in kotlin-stdlib 2.2.21 (metadata v2.2 unreadable by Kotlin 1.9).
    id("org.jetbrains.kotlin.android") version "2.2.21" apply false
}
