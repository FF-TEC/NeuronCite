// NeuronCite -- local, privacy-preserving semantic document search engine.
// Copyright (C) 2026 NeuronCite Contributors
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

//! ONNX Runtime session configuration and creation.
//!
//! Manages the `ort::session::Session` lifecycle, including execution provider
//! selection (CUDA, DirectML, CoreML, ROCm, or CPU), graph optimization level, thread pool
//! configuration, and memory arena settings. A single session instance is
//! shared across all inference calls for a given model.
//!
//! GPU execution provider selection follows a cascading fallback strategy
//! that adapts to the host platform:
//!
//! | Priority | Provider  | Platforms        | Hardware                              |
//! |----------|-----------|------------------|---------------------------------------|
//! | 1        | CUDA      | Windows, Linux   | NVIDIA GPUs with cuDNN 9.x            |
//! | 2        | DirectML  | Windows          | Any GPU with DirectX 12 (NVIDIA/AMD/Intel) |
//! | 3        | CoreML    | macOS            | Apple Neural Engine + GPU (M1-M4)     |
//! | 4        | ROCm      | Linux x86_64     | AMD GPUs with ROCm driver             |
//! | 5        | CPU       | all              | any x86_64 or arm64 CPU               |
//!
//! The `load-dynamic` feature on the `ort` crate satisfies the compile-time
//! cfg gates for all execution providers (CUDA, DirectML, CoreML, ROCm), so a single
//! binary per platform works on different hardware configurations without
//! compile-time feature changes.
//!
//! ## ORT library loading without environment variables
//!
//! `ensure_ort_library` uses `ort::init_from(path)` instead of
//! `std::env::set_var("ORT_DYLIB_PATH", ...)`. `ort::init_from` calls
//! `load_dylib_from_path` which invokes `LoadLibraryW` (Windows) or
//! `dlopen` (Linux/macOS) directly, bypassing the process environment
//! entirely. This eliminates the data race that would occur if concurrent
//! threads read `ORT_DYLIB_PATH` via `std::env::var` while initialization
//! was writing it.
//!
//! The `ORT_LIB_INIT` Mutex serializes concurrent initialization attempts.
//! CUDA runtime PATH configuration (`ensure_cuda_runtime_on_path`) still
//! modifies the PATH environment variable but happens before any async
//! threads are spawned and is always called from the initialization path.

use std::path::PathBuf;
use std::sync::OnceLock;

use neuroncite_core::InferenceCapabilities;
use ort::session::Session;

use crate::error::EmbedError;

/// Serializes concurrent ORT shared library initialization attempts. Uses a
/// Mutex so that failed download/init attempts can be retried by the user
/// (e.g., after a transient network failure when downloading onnxruntime from
/// GitHub, or after freeing disk space). A Mutex allows leaving the state as
/// None on failure, while OnceLock would permanently cache the first result --
/// including errors -- for the entire process lifetime with no recovery path.
static ORT_LIB_INIT: std::sync::Mutex<Option<Result<(), String>>> = std::sync::Mutex::new(None);

/// Caches the result of the one-time CUDA runtime availability check.
/// Set to `true` if both the GPU variant of ORT is installed AND cuDNN 9.x
/// is found on the system. On macOS, this is always `false` because CUDA
/// is not available on Apple hardware. Once determined, this value does not
/// change for the lifetime of the process. OnceLock is correct here because
/// CUDA hardware availability does not change during the process lifetime.
static CUDA_RUNTIME_READY: OnceLock<bool> = OnceLock::new();

/// Ensures the ONNX Runtime shared library is available for the `ort` crate's
/// `load-dynamic` feature.
///
/// Platform-specific behavior:
/// - **Windows**: Downloads `onnxruntime.dll` if needed, sets `ORT_DYLIB_PATH`,
///   and configures the CUDA runtime PATH for cuDNN discovery.
/// - **macOS**: Downloads `libonnxruntime.dylib` if needed, sets `ORT_DYLIB_PATH`.
///   CUDA is not available on macOS; GPU acceleration uses CoreML instead.
/// - **Linux**: Downloads `libonnxruntime.so` if needed, sets `ORT_DYLIB_PATH`,
///   and configures `LD_LIBRARY_PATH` for cuDNN discovery. If `ORT_DYLIB_PATH`
///   is already set (e.g., in Docker), the auto-download is skipped.
///
/// This function is idempotent: successful initialization is cached permanently.
/// Failed attempts are NOT cached, allowing the user to retry after fixing the
/// issue (e.g., restoring network connectivity, freeing disk space).
fn ensure_ort_library() -> Result<(), EmbedError> {
    let mut guard = ORT_LIB_INIT
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    // Previous successful initialization is cached permanently.
    if let Some(Ok(())) = guard.as_ref() {
        return Ok(());
    }

    let result = ort_library_init_inner();

    match &result {
        Ok(()) => {
            *guard = Some(Ok(()));
        }
        Err(_) => {
            // Do NOT cache failures. Leave the guard as None so the user can
            // retry initialization after fixing the issue (e.g., restoring
            // network connectivity, freeing disk space, granting file permissions).
        }
    }

    result
}

/// Performs the platform-specific ORT library download, environment variable
/// configuration, and CUDA runtime detection. Called by `ensure_ort_library`
/// on the first successful attempt or on each retry after a failure.
fn ort_library_init_inner() -> Result<(), EmbedError> {
    #[cfg(target_os = "windows")]
    {
        let dll_path =
            crate::cache::ensure_ort_runtime().map_err(|e| EmbedError::Download(e.to_string()))?;

        // Set up CUDA runtime PATH BEFORE loading the ORT shared library.
        // The GPU variant's onnxruntime.dll dynamically loads
        // onnxruntime_providers_cuda.dll via onnxruntime_providers_shared.dll,
        // which depends on cuDNN 9.x (cudnn64_9.dll) and CUDA runtime libraries
        // (cublas64_12.dll, etc.). These DLLs must be discoverable via PATH
        // before LoadLibraryW is called on onnxruntime.dll, otherwise the
        // transitive dependency resolution during DLL loading fails.
        if crate::cache::is_gpu_ort_runtime() {
            let cudnn_available = crate::cache::ensure_cuda_runtime_on_path();
            CUDA_RUNTIME_READY.get_or_init(|| cudnn_available);
        } else {
            CUDA_RUNTIME_READY.get_or_init(|| false);
        }

        // Load the ORT shared library directly via LoadLibraryW (Windows) without
        // touching the process environment. ort::init_from passes the path straight
        // to the ORT C API's load-dynamic mechanism, avoiding any set_var data race
        // with concurrent env::var readers in other threads.
        //
        // commit() stores the environment configuration globally. The actual ORT
        // Environment is created lazily when the first session is built.
        ort::init_from(&dll_path)
            .map_err(|e| EmbedError::ModelLoad(format!("ORT library init failed: {e}")))?
            .commit();
        tracing::info!(
            path = %dll_path.display(),
            "ORT shared library loaded for load-dynamic (Windows)"
        );

        Ok(())
    }

    #[cfg(target_os = "macos")]
    {
        let dylib_path =
            crate::cache::ensure_ort_runtime().map_err(|e| EmbedError::Download(e.to_string()))?;

        // Load the ORT shared library directly via dlopen (macOS) without touching
        // the process environment. ort::init_from passes the path straight to the
        // ORT C API's load-dynamic mechanism, eliminating any set_var data race.
        ort::init_from(&dylib_path)
            .map_err(|e| EmbedError::ModelLoad(format!("ORT library init failed: {e}")))?
            .commit();
        tracing::info!(
            path = %dylib_path.display(),
            "ORT shared library loaded for load-dynamic (macOS)"
        );

        // macOS does not support CUDA; GPU acceleration uses CoreML through
        // the Apple Neural Engine and integrated GPU.
        CUDA_RUNTIME_READY.get_or_init(|| false);

        Ok(())
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        // On Linux, auto-download ORT from GitHub if not already present.
        // If ORT_DYLIB_PATH is pre-set (Docker, manual install), the
        // ensure_ort_runtime function respects the existing path.
        let so_path =
            crate::cache::ensure_ort_runtime().map_err(|e| EmbedError::Download(e.to_string()))?;

        // Set up CUDA runtime LD_LIBRARY_PATH BEFORE loading the ORT shared
        // library. The GPU variant's libonnxruntime.so dynamically links to
        // libonnxruntime_providers_cuda.so, which depends on cuDNN 9.x
        // (libcudnn.so.9) and CUDA runtime libraries. These must be
        // discoverable via LD_LIBRARY_PATH before dlopen is called on the
        // main ORT library, otherwise the dependent .so files cannot be
        // resolved by the dynamic linker.
        if crate::cache::is_gpu_ort_runtime() {
            let cudnn_available = crate::cache::ensure_cuda_runtime_on_path();
            CUDA_RUNTIME_READY.get_or_init(|| cudnn_available);
        } else {
            CUDA_RUNTIME_READY.get_or_init(|| false);
        }

        // Load the ORT shared library directly via dlopen (Linux) without touching
        // the process environment. ort::init_from passes the path straight to the
        // ORT C API's load-dynamic mechanism, eliminating any set_var data race.
        ort::init_from(&so_path)
            .map_err(|e| EmbedError::ModelLoad(format!("ORT library init failed: {e}")))?
            .commit();
        tracing::info!(
            path = %so_path.display(),
            "ORT shared library loaded for load-dynamic (Linux)"
        );

        Ok(())
    }
}

/// Returns whether the loaded ONNX Runtime shared library includes GPU
/// (CUDA) execution provider support.
///
/// - **Windows**: Checks for `onnxruntime_providers_cuda.dll` in the ORT cache.
/// - **Linux**: Checks for `libonnxruntime_providers_cuda.so` in the ORT cache.
/// - **macOS**: Always returns `false` (no CUDA on macOS; CoreML is used instead).
fn is_gpu_runtime_loaded() -> bool {
    #[cfg(not(target_os = "macos"))]
    {
        crate::cache::is_gpu_ort_runtime()
    }
    #[cfg(target_os = "macos")]
    {
        // macOS does not support NVIDIA GPUs. GPU acceleration is provided
        // by CoreML through the Apple Neural Engine and integrated GPU.
        false
    }
}

/// Returns whether the CUDA runtime dependencies (cuDNN 9.x, cuBLAS 12.x)
/// are available on the system. This is determined during `ensure_ort_library`
/// initialization and cached for the process lifetime.
fn is_cuda_runtime_ready() -> bool {
    *CUDA_RUNTIME_READY.get().unwrap_or(&false)
}

/// Configuration for creating an ONNX Runtime inference session.
/// Specifies the model file path and whether to attempt GPU acceleration.
#[derive(Debug, Clone)]
pub struct OrtSessionConfig {
    /// Filesystem path to the `.onnx` model file.
    pub model_path: PathBuf,

    /// Whether to attempt hardware-accelerated execution provider registration.
    /// When true, the session creation attempts platform-specific GPU providers
    /// (CUDA on Windows/Linux, CoreML on macOS, ROCm on Linux) before falling
    /// back to the CPU provider. When false, the CPU provider is used directly.
    pub use_gpu: bool,
}

/// Determines the number of intra-op threads for ORT's CPU thread pool,
/// adapted to the target execution provider.
///
/// When a hardware-accelerated EP manages its own dispatch (CoreML → ANE/GPU,
/// CUDA → GPU kernels), ORT's CPU thread pool only handles the few operators
/// the accelerator cannot run. Allocating many CPU threads in that scenario
/// wastes resources and increases context-switching overhead without improving
/// accelerator throughput.
///
/// | EP        | Max threads | Rationale                                        |
/// |-----------|-------------|--------------------------------------------------|
/// | CoreML    | 2           | ANE/GPU dispatch is internal; CPU handles fallback |
/// | CUDA/ROCm | 8           | GPU kernels dominate; CPU does data prep          |
/// | DirectML  | 8           | Similar to CUDA on Windows GPUs                  |
/// | CPU       | 16          | Maximize parallelism for GEMM operations          |
fn intra_op_threads(target_ep: &str) -> usize {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    let max = match target_ep {
        // CoreML dispatches most operators to ANE/GPU internally. The ORT
        // CPU thread pool only handles fallback operators (shape ops, Gather,
        // Unsqueeze) that CoreML cannot run. 4 threads give these CPU-side
        // operators enough parallelism without contending with the ANE/GPU
        // dispatch threads. Previously set to 2, which starved CPU-fallback
        // operators when the CoreML graph partition was incomplete.
        "CoreML" => 4,
        "CUDA" | "ROCm" | "DirectML" => 8,
        _ => 16,
    };

    cores.min(max).max(1)
}

/// Selects the graph optimization level appropriate for the target execution
/// provider.
///
/// - **CoreML**: Level1 (constant folding and dead-code elimination only).
///   Level2/Level3 create fused operators (`FusedAttention`,
///   `EmbedLayerNormalization`, `FusedGELU`) that CoreML's MLProgram format
///   does not support, causing them to fall back to the CPU execution provider.
///   This graph partitioning overhead plus the CPU<->CoreML data transfers
///   makes fused graphs *slower* than pure-CPU inference on Apple Silicon.
///   Level1 keeps BERT operators in their unfused form (separate MatMul,
///   Softmax, LayerNormalization, Gelu nodes), which CoreML supports natively
///   on the Apple Neural Engine and integrated GPU.
///
/// - **All other EPs** (CUDA, DirectML, ROCm, CPU): Level3 (full operator
///   fusion). These providers natively support fused operators and benefit
///   from the 15-40% throughput gain that fusion provides.
fn optimization_level(target_ep: &str) -> ort::session::builder::GraphOptimizationLevel {
    match target_ep {
        "CoreML" => ort::session::builder::GraphOptimizationLevel::Level1,
        _ => ort::session::builder::GraphOptimizationLevel::Level3,
    }
}

/// Applies graph optimization and threading configuration to a session builder.
///
/// The `target_ep` parameter selects both the optimization level (see
/// [`optimization_level`]) and the intra-op thread count (see
/// [`intra_op_threads`]) appropriate for the execution provider being
/// attempted.
fn configure_builder(
    builder: ort::session::builder::SessionBuilder,
    target_ep: &str,
) -> Result<ort::session::builder::SessionBuilder, EmbedError> {
    let threads = intra_op_threads(target_ep);
    let opt_level = optimization_level(target_ep);
    let opt_level_num: u8 = match opt_level {
        ort::session::builder::GraphOptimizationLevel::Disable => 0,
        ort::session::builder::GraphOptimizationLevel::Level1 => 1,
        ort::session::builder::GraphOptimizationLevel::Level2 => 2,
        ort::session::builder::GraphOptimizationLevel::Level3
        | ort::session::builder::GraphOptimizationLevel::All => 3,
    };
    tracing::debug!(
        optimization_level = opt_level_num,
        intra_threads = threads,
        target_ep,
        "configuring ORT session builder"
    );

    builder
        .with_optimization_level(opt_level)
        .map_err(|e| EmbedError::ModelLoad(format!("failed to set optimization level: {e}")))?
        .with_intra_threads(threads)
        .map_err(|e| EmbedError::ModelLoad(format!("failed to set intra-op threads: {e}")))
}

/// Attempts to create an ONNX session using a specific execution provider.
/// Creates a session builder, applies Level3 graph optimization and threading
/// configuration, registers the execution provider, and loads the model from
/// disk.
///
/// Returns `Ok(Some(session))` if all steps succeed, `Ok(None)` if the EP
/// registration or model loading fails (non-fatal, allows fallback to the
/// next EP in the cascade). Returns `Err` only for unrecoverable errors
/// during session builder creation or graph optimization configuration.
fn try_execution_provider(
    config: &OrtSessionConfig,
    ep_name: &str,
    ep: ort::execution_providers::ExecutionProviderDispatch,
) -> Result<Option<Session>, EmbedError> {
    tracing::info!(
        provider = ep_name,
        "attempting {ep_name} execution provider"
    );

    let builder = Session::builder()
        .map_err(|e| EmbedError::ModelLoad(format!("failed to create session builder: {e}")))?;
    let builder = configure_builder(builder, ep_name)?;

    let mut builder = match builder.with_execution_providers([ep]) {
        Ok(b) => b,
        Err(e) => {
            tracing::info!(provider = ep_name, "{ep_name} EP registration failed: {e}");
            return Ok(None);
        }
    };

    match builder.commit_from_file(&config.model_path) {
        Ok(session) => {
            tracing::info!(
                model = %config.model_path.display(),
                provider = ep_name,
                "ONNX session created with {ep_name} execution provider"
            );
            Ok(Some(session))
        }
        Err(e) => {
            tracing::warn!(
                provider = ep_name,
                "model load with {ep_name} EP failed: {e}"
            );
            Ok(None)
        }
    }
}

// ---------------------------------------------------------------------------
// System memory detection
// ---------------------------------------------------------------------------

/// Detects the total physical system memory in bytes using platform-specific
/// commands. Falls back to a conservative 8 GB estimate on detection failure.
///
/// - **macOS**: `sysctl -n hw.memsize` (same pattern as `detect_apple_silicon_name` in cache.rs)
/// - **Linux**: Parses `MemTotal` from `/proc/meminfo`
/// - **Windows**: `wmic computersystem get TotalPhysicalMemory`
fn detect_system_memory() -> u64 {
    const FALLBACK: u64 = 8 * 1024 * 1024 * 1024;

    let result = detect_system_memory_inner();
    match result {
        Some(bytes) if bytes > 0 => {
            tracing::info!(
                memory_gb = bytes / (1024 * 1024 * 1024),
                memory_bytes = bytes,
                "detected system memory"
            );
            bytes
        }
        _ => {
            tracing::warn!(
                fallback_gb = FALLBACK / (1024 * 1024 * 1024),
                "system memory detection failed, using conservative fallback"
            );
            FALLBACK
        }
    }
}

/// Platform-specific memory detection implementation. Returns `None` on failure.
fn detect_system_memory_inner() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        stdout.trim().parse::<u64>().ok()
    }

    #[cfg(target_os = "linux")]
    {
        let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in contents.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                // Format: "MemTotal:       16384000 kB"
                let kb_str = rest.trim().strip_suffix("kB")?.trim();
                let kb: u64 = kb_str.parse().ok()?;
                return Some(kb * 1024);
            }
        }
        None
    }

    #[cfg(target_os = "windows")]
    {
        let output = std::process::Command::new("wmic")
            .args(["computersystem", "get", "TotalPhysicalMemory"])
            .output()
            .ok()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        // Output has a header line and a data line with the byte count.
        for line in stdout.lines().skip(1) {
            if let Ok(bytes) = line.trim().parse::<u64>() {
                return Some(bytes);
            }
        }
        None
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        None
    }
}

/// Returns whether the current platform uses a unified memory architecture
/// where CPU and GPU share the same physical DRAM. Currently true only on
/// Apple Silicon (aarch64 macOS).
fn is_unified_memory() -> bool {
    cfg!(target_os = "macos") && cfg!(target_arch = "aarch64")
}

/// Constructs an [`InferenceCapabilities`] for the given execution provider.
fn build_inference_capabilities(active_ep: &str) -> InferenceCapabilities {
    InferenceCapabilities {
        active_ep: active_ep.to_string(),
        system_memory_bytes: detect_system_memory(),
        unified_memory: is_unified_memory(),
    }
}

// ---------------------------------------------------------------------------
// Session creation
// ---------------------------------------------------------------------------

/// Creates an ONNX Runtime session from the given configuration.
///
/// The session is configured with Level3 graph optimization (Attention Fusion,
/// GELU Fusion, Constant Folding) and an intra-op thread pool sized to the
/// system's available parallelism.
///
/// When `config.use_gpu` is true, the function attempts platform-specific GPU
/// execution providers in cascading order:
///
/// 1. **CUDA** (Windows/Linux) -- requires GPU ORT variant + cuDNN 9.x
/// 2. **DirectML** (Windows) -- any GPU with DirectX 12 (NVIDIA, AMD, Intel)
/// 3. **CoreML** (macOS) -- uses Apple Neural Engine + GPU on M1-M4 chips
/// 4. **ROCm** (Linux) -- requires AMD GPU + ROCm driver + ROCm ORT variant
/// 5. **CPU** -- always available as the final fallback on all platforms
///
/// Each GPU provider attempt is self-contained: if EP registration or model
/// loading fails, the function proceeds to the next provider in the cascade.
///
/// # Arguments
///
/// * `config` - Session configuration specifying the model path and GPU preference.
///
/// # Errors
///
/// Returns `EmbedError::ModelLoad` if the ONNX model file cannot be loaded
/// with any available execution provider.
pub fn create_session(
    config: &OrtSessionConfig,
) -> Result<(Session, InferenceCapabilities), EmbedError> {
    // Ensure the ORT shared library is downloaded, ORT_DYLIB_PATH is set,
    // and platform-specific runtime dependencies are configured. This is
    // required before any ort crate API calls.
    ensure_ort_library()?;

    if config.use_gpu {
        // -----------------------------------------------------------------
        // CUDA (NVIDIA) -- Windows and Linux only.
        // macOS does not support NVIDIA GPUs (Apple hardware has no CUDA
        // driver). Three preconditions must be met for the CUDA EP:
        // 1. GPU variant of ORT is loaded (onnxruntime_providers_cuda present)
        // 2. cuDNN 9.x is installed and on PATH
        // 3. CUDA EP registration succeeds with the loaded ORT library
        // -----------------------------------------------------------------
        #[cfg(not(target_os = "macos"))]
        {
            let gpu_runtime_present = is_gpu_runtime_loaded();
            let cuda_deps_available = is_cuda_runtime_ready();

            if gpu_runtime_present && cuda_deps_available {
                if let Some(session) = try_execution_provider(
                    config,
                    "CUDA",
                    ort::execution_providers::CUDAExecutionProvider::default().build(),
                )? {
                    return Ok((session, build_inference_capabilities("CUDA")));
                }
            } else if gpu_runtime_present && !cuda_deps_available {
                // GPU ORT variant is loaded but cuDNN is missing. The CUDA EP
                // would silently fail and fall back to CPU. Skip the attempt
                // entirely and log the specific missing dependency.
                tracing::warn!(
                    "GPU ORT runtime loaded but cuDNN 9.x not installed; \
                     CUDA EP requires cuDNN, skipping"
                );
            } else if !gpu_runtime_present {
                // CPU-only ORT is loaded (no NVIDIA GPU was detected during
                // the ORT download phase).
                tracing::info!(
                    "CPU-only ORT runtime loaded (no CUDA provider libraries); \
                     skipping CUDA execution provider"
                );
            }
        }

        // -----------------------------------------------------------------
        // DirectML (Windows only) -- any GPU with DirectX 12 support.
        // DirectML provides hardware acceleration on NVIDIA, AMD, and Intel
        // GPUs through the DirectX 12 API. It requires no external runtime
        // installation because DirectML.dll ships with Windows 10 1903+.
        // Placed after CUDA because CUDA generally provides higher throughput
        // on NVIDIA hardware, but DirectML is the fallback for AMD and Intel
        // GPUs on Windows where CUDA is unavailable.
        // -----------------------------------------------------------------
        #[cfg(target_os = "windows")]
        {
            if let Some(session) = try_execution_provider(
                config,
                "DirectML",
                ort::execution_providers::DirectMLExecutionProvider::default().build(),
            )? {
                return Ok((session, build_inference_capabilities("DirectML")));
            }
        }

        // -----------------------------------------------------------------
        // CoreML (Apple) -- macOS only.
        // Uses Apple's CoreML framework to dispatch inference to the Apple
        // Neural Engine (ANE) and integrated GPU on Apple Silicon (M-series).
        // On Intel Macs, CoreML dispatches to the CPU via Apple's Accelerate
        // framework, still providing SIMD-optimized inference.
        // CoreML is a system framework on macOS, so no additional runtime
        // dependencies are needed beyond the CoreML-enabled ORT library.
        //
        // Configuration details:
        //   - ComputeUnits::All  enables ANE + GPU + CPU; without this flag,
        //     CoreML defaults to CPU-only, leaving the Neural Engine idle.
        //   - ModelFormat::MLProgram  uses the modern ML Program format which
        //     supports a wider range of ONNX operators on ANE/GPU than the
        //     legacy NeuralNetwork format (especially Attention layers).
        //   - FastPrediction specialization  tells CoreML to optimise for
        //     repeated low-latency inference, matching batch-embedding usage.
        //   - Model cache directory  stores the compiled CoreML model on disk
        //     so the expensive ONNX-to-CoreML compilation (10-30 s for BERT,
        //     minutes for larger models) only happens once.
        // -----------------------------------------------------------------
        #[cfg(target_os = "macos")]
        {
            let coreml_cache = neuroncite_core::paths::runtime_dir().join("coreml_cache");
            if let Some(session) = try_execution_provider(
                config,
                "CoreML",
                ort::execution_providers::CoreMLExecutionProvider::default()
                    .with_compute_units(ort::execution_providers::coreml::ComputeUnits::All)
                    .with_model_format(ort::execution_providers::coreml::ModelFormat::MLProgram)
                    .with_specialization_strategy(
                        ort::execution_providers::coreml::SpecializationStrategy::FastPrediction,
                    )
                    .with_low_precision_accumulation_on_gpu(true)
                    .with_model_cache_dir(coreml_cache.to_string_lossy().to_string())
                    .build(),
            )? {
                tracing::info!(
                    model = %config.model_path.display(),
                    optimization_level = "Level1 (unfused, CoreML-compatible)",
                    intra_threads = intra_op_threads("CoreML"),
                    model_cache = %coreml_cache.display(),
                    compute_units = "All (ANE + GPU + CPU)",
                    model_format = "MLProgram",
                    "CoreML session active — first inference may be slow while \
                     CoreML compiles the model for the Neural Engine"
                );
                return Ok((session, build_inference_capabilities("CoreML")));
            }

            tracing::warn!(
                model = %config.model_path.display(),
                "CoreML execution provider registration or model load failed; \
                 falling back to CPU. Indexing will be significantly slower. \
                 Ensure macOS is up to date and the ONNX model uses operators \
                 supported by CoreML MLProgram format."
            );
        }

        // -----------------------------------------------------------------
        // ROCm (AMD) -- Linux x86_64 only.
        // Requires an AMD GPU with ROCm driver and the ROCm variant of the
        // ORT shared library. The ROCm EP is platform-gated inside the ort
        // crate to Linux x86_64; on Windows and macOS, EP registration is
        // a no-op and the session falls through to the CPU provider.
        // -----------------------------------------------------------------
        #[cfg(not(target_os = "macos"))]
        {
            if let Some(session) = try_execution_provider(
                config,
                "ROCm",
                ort::execution_providers::ROCmExecutionProvider::default().build(),
            )? {
                return Ok((session, build_inference_capabilities("ROCm")));
            }
        }
    }

    // -----------------------------------------------------------------
    // CPU -- final fallback on all platforms.
    // Either GPU was not requested, all GPU EP attempts failed, or the
    // required GPU runtime dependencies are missing.
    // -----------------------------------------------------------------
    let builder = Session::builder()
        .map_err(|e| EmbedError::ModelLoad(format!("failed to create session builder: {e}")))?;

    let mut builder = configure_builder(builder, "CPU")?;

    let session = builder.commit_from_file(&config.model_path).map_err(|e| {
        EmbedError::ModelLoad(format!(
            "failed to load ONNX model: {}: {e}",
            config.model_path.display()
        ))
    })?;

    tracing::info!(
        model = %config.model_path.display(),
        "ONNX session created with CPU execution provider"
    );
    Ok((session, build_inference_capabilities("CPU")))
}

/// Returns whether the ORT shared library has been successfully loaded into
/// the current process. This is a non-blocking read of the cached initialization
/// state. Returns `false` when ORT has not yet been initialized or when the
/// previous initialization attempt failed.
///
/// Use this function to guard informational GPU queries (e.g., health checks,
/// model catalogs) that should not trigger a download. Functions that require
/// ORT for actual inference should call `ensure_ort_library()` instead.
#[must_use]
pub fn is_ort_library_loaded() -> bool {
    let guard = ORT_LIB_INIT
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    matches!(guard.as_ref(), Some(Ok(())))
}

/// Checks whether the CUDA execution provider is fully available on the
/// current system. This verifies three conditions:
///
/// 1. The ORT shared library is loadable.
/// 2. The GPU variant of ORT is installed (contains CUDA provider DLLs).
/// 3. cuDNN 9.x and the CUDA 12.x runtime are installed and on PATH.
///
/// Returns `true` only when all conditions are met, `false` otherwise.
/// On macOS, always returns `false` (CUDA is not available on Apple hardware).
///
/// This function does NOT attempt EP registration (which would trigger
/// misleading ort internal log messages).
#[must_use]
pub fn is_cuda_available() -> bool {
    // Ensure the ORT shared library is available before checking CUDA.
    if ensure_ort_library().is_err() {
        return false;
    }

    // Check both the GPU ORT variant AND cuDNN availability.
    if !is_gpu_runtime_loaded() {
        tracing::debug!("CPU-only ORT runtime loaded; CUDA execution provider is not available");
        return false;
    }

    if !is_cuda_runtime_ready() {
        tracing::debug!(
            "GPU ORT runtime loaded but cuDNN not found; \
             CUDA execution provider is not available"
        );
        return false;
    }

    true
}

/// Checks whether the DirectML execution provider is available on the current
/// system. Returns `true` on Windows where DirectML ships as part of the OS
/// (Windows 10 version 1903 and later), `false` on all other platforms.
///
/// DirectML provides hardware-accelerated ML inference through DirectX 12.
/// It supports any GPU with a DirectX 12 driver, including NVIDIA, AMD,
/// and Intel integrated/discrete GPUs. DirectML requires no separate
/// runtime installation and is loaded at runtime via the ORT load-dynamic
/// mechanism.
///
/// This function does NOT verify that a DirectX 12 GPU is physically present;
/// it only checks that the platform supports DirectML. EP registration at
/// session creation time handles the actual hardware check.
#[must_use]
pub fn is_directml_available() -> bool {
    #[cfg(target_os = "windows")]
    {
        ensure_ort_library().is_ok()
    }
    #[cfg(not(target_os = "windows"))]
    {
        false
    }
}

/// Checks whether the CoreML execution provider is available on the current
/// system. Returns `true` on macOS where CoreML is a built-in system framework,
/// `false` on all other platforms.
///
/// On macOS, CoreML is always available as a system framework. The actual
/// inference acceleration depends on the hardware:
/// - Apple Silicon (M1-M4): Neural Engine + GPU hardware acceleration
/// - Intel Macs: CPU-only via Apple's Accelerate framework with SIMD
///
/// Whether the loaded ORT library includes CoreML EP support is determined
/// at session creation time by the EP registration call, not by this function.
#[must_use]
pub fn is_coreml_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        // CoreML is a system framework on macOS. The only precondition is
        // that the ORT shared library is loadable.
        ensure_ort_library().is_ok()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-EMB-020: The `ORT_LIB_INIT` Mutex starts as None and does not
    /// permanently cache failures. This was changed from OnceLock to Mutex
    /// so that transient failures (network timeout, disk full) can be retried
    /// without restarting the process. The Mutex state must never contain a
    /// cached Err; failures leave it as None for retry.
    #[test]
    fn t_emb_020_ort_lib_init_mutex_does_not_cache_failures() {
        let guard = ORT_LIB_INIT.lock().unwrap_or_else(|p| p.into_inner());

        match guard.as_ref() {
            None => {
                // No initialization attempted yet in this test process.
                // This is the expected initial state for unit tests that
                // do not invoke ensure_ort_library().
            }
            Some(Ok(())) => {
                // A previous initialization (e.g., integration test) succeeded.
                // The cached success is correct and permanent.
            }
            Some(Err(_)) => {
                panic!(
                    "ORT_LIB_INIT Mutex contains a cached error. \
                     Failures must not be cached (the Mutex should be left as None \
                     on failure to allow retries)."
                );
            }
        }
    }

    /// T-EMB-021: The `CUDA_RUNTIME_READY` OnceLock is correctly typed as
    /// OnceLock<bool>. Unlike ORT_LIB_INIT, CUDA hardware availability does
    /// not change during the process lifetime, so OnceLock is the correct
    /// primitive for this value.
    #[test]
    fn t_emb_021_cuda_runtime_ready_oncelock_type() {
        // Before ensure_ort_library is called, CUDA_RUNTIME_READY is unset.
        // After ensure_ort_library, it holds a boolean. Both states are valid.
        let value: Option<&bool> = CUDA_RUNTIME_READY.get();
        match value {
            None => {
                // Not yet determined. is_cuda_runtime_ready() returns false.
                assert!(
                    !is_cuda_runtime_ready(),
                    "is_cuda_runtime_ready must return false before initialization"
                );
            }
            Some(&ready) => {
                // Already determined by a prior initialization.
                assert_eq!(
                    ready,
                    is_cuda_runtime_ready(),
                    "is_cuda_runtime_ready must match the OnceLock value"
                );
            }
        }
    }

    /// T-EMB-022: The ORT Mutex poison recovery pattern compiles and produces
    /// a usable guard. Validates the `unwrap_or_else(|p| p.into_inner())`
    /// pattern used in `ensure_ort_library` for Mutex poison recovery.
    #[test]
    fn t_emb_022_mutex_poison_recovery_pattern() {
        let test_mutex: std::sync::Mutex<Option<Result<(), String>>> =
            std::sync::Mutex::new(Some(Ok(())));

        let guard = test_mutex
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        assert!(
            guard.is_some(),
            "guard should contain the test value after non-poisoned lock"
        );
    }

    /// T-EMB-023: `intra_op_threads` returns a value within the EP-specific
    /// upper bound. CoreML <= 4, CUDA/ROCm/DirectML <= 8, CPU <= 16.
    /// All EPs must return at least 1.
    #[test]
    fn t_emb_023_intra_op_threads_range() {
        for (ep, max) in [
            ("CPU", 16),
            ("CoreML", 4),
            ("CUDA", 8),
            ("ROCm", 8),
            ("DirectML", 8),
            ("unknown", 16),
        ] {
            let threads = intra_op_threads(ep);
            assert!(
                threads >= 1,
                "intra_op_threads({ep}) must be at least 1, got {threads}"
            );
            assert!(
                threads <= max,
                "intra_op_threads({ep}) must be at most {max}, got {threads}"
            );
        }
    }
}
