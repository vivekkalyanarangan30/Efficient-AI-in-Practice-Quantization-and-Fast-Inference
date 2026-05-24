plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.ch11.bench"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.ch11.bench"
        minSdk = 28
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk {
            // Pixel 9 + AWS Device Farm phones are arm64-v8a. Keep APK small.
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("debug")
        }
        debug {
            isMinifyEnabled = false
        }
    }

    androidResources {
        // .tflite files must be mmap-able at runtime; don't compress them.
        // .bin contains preprocessed validation samples — also leave uncompressed for direct read.
        // .task bundles host the MediaPipe LLM weights and are mmap'd by LlmInference.
        noCompress += listOf("tflite", "bin", "task", "litertlm")
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    packaging {
        // TFLite GPU delegate ships duplicate META-INF entries across artifacts.
        resources.excludes += listOf(
            "META-INF/AL2.0",
            "META-INF/LGPL2.1",
            "META-INF/*.kotlin_module",
        )
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")
    // Coroutines 1.10.2 supports Kotlin 2.2.x (1.7.3 doesn't).
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")

    // TFLite runtime + delegates.
    // 2.17.0 is the first artifact with FULLY_CONNECTED opcode v12 support, which
    // the dynamic-range .tflite (converted on the Mac with TF >=2.18) requires.
    // tensorflow-lite-support 0.4.4 is omitted: it pulls in tflite-api 2.13 which
    // collides with the litert-api transitive shipped by tflite 2.17. We don't
    // use any of its helpers anyway.
    implementation("org.tensorflow:tensorflow-lite:2.17.0")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.17.0")
    implementation("org.tensorflow:tensorflow-lite-gpu-api:2.17.0")

    // LiteRT-LM (Google AI Edge LLM runtime). Replaces the older
    // com.google.mediapipe:tasks-genai 0.10.24 which only reads the legacy
    // .task FlatBuffer-wrapped-ZIP format. LiteRT-LM 0.12.0 natively reads
    // the .litertlm bundle produced by litert-lm-builder on the VM
    // conversion path (schema 1.5.0). Exposes a Flow-based async streaming
    // API (`sendMessageAsync(...).collect { ... }`) that gives us clean
    // per-token timestamps for TTFT/TPOT measurement.
    implementation("com.google.ai.edge.litertlm:litertlm-android:0.12.0")

    // Instrumentation test deps — run the benchmark unattended on AWS Device Farm
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test:rules:1.6.1")
}
