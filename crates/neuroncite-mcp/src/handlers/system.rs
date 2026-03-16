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

//! Handlers for the `neuroncite_doctor` and `neuroncite_health` MCP tools.
//!
//! Provides system capability checks (GPU, CUDA, backends, OCR) and server
//! health status reporting. CUDA availability checks are gated behind the
//! backend-ort feature flag.

use std::sync::Arc;

use neuroncite_api::AppState;

/// Returns true if CUDA is available. Falls back to false when the backend-ort
/// feature is not compiled.
fn cuda_available() -> bool {
    #[cfg(feature = "backend-ort")]
    {
        neuroncite_embed::is_cuda_available()
    }
    #[cfg(not(feature = "backend-ort"))]
    {
        false
    }
}

/// Returns true if CoreML is available (macOS only). Falls back to false
/// when the backend-ort feature is not compiled or on non-macOS platforms.
fn coreml_available() -> bool {
    #[cfg(feature = "backend-ort")]
    {
        neuroncite_embed::is_coreml_available()
    }
    #[cfg(not(feature = "backend-ort"))]
    {
        false
    }
}

/// Checks system capabilities and reports available features.
///
/// Returns GPU detection results, CUDA availability, compiled backends,
/// reranker availability, and optional dependency status (Tesseract, pdfium).
pub async fn handle_doctor(
    state: &Arc<AppState>,
    _params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let backends = neuroncite_embed::list_available_backends();
    let cuda = cuda_available();

    let backend_names: Vec<String> = backends.iter().map(|b| b.name.clone()).collect();

    // Probe pdfium and Tesseract availability by checking the centralized
    // runtime cache directory and known filesystem locations. The neuroncite-mcp
    // crate does not depend on neuroncite-pdf, so these checks mirror the
    // probe logic in the GUI Doctor panel (neuroncite-gui::panels::doctor).
    let pdfium = probe_pdfium_available();
    let tesseract = probe_tesseract_available();

    let coreml = coreml_available();

    Ok(serde_json::json!({
        "cuda_available": cuda,
        "coreml_available": coreml,
        "compiled_backends": backend_names,
        "reranker_available": state.worker_handle.reranker_available(),
        "pdfium_available": pdfium,
        "tesseract_available": tesseract,
        "cache_directory": neuroncite_embed::cache_dir().display().to_string(),
    }))
}

/// Returns server health status including version and backend information.
/// The vector_dimension field is an AtomicUsize, so a relaxed load is
/// performed to read the current value as a plain usize for the JSON response.
pub async fn handle_health(
    state: &Arc<AppState>,
    _params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let backends = neuroncite_embed::list_available_backends();
    let active_backend = backends
        .first()
        .map(|b| b.name.clone())
        .unwrap_or_else(|| "none".to_string());

    // GPU availability is the union of all platform-specific providers:
    // CUDA on Windows/Linux, CoreML on macOS.
    let gpu = cuda_available() || coreml_available();

    Ok(serde_json::json!({
        "api_version": "v1",
        "version": env!("CARGO_PKG_VERSION"),
        "active_backend": active_backend,
        "gpu_available": gpu,
        "cuda_available": cuda_available(),
        "coreml_available": coreml_available(),
        "reranker_available": state.worker_handle.reranker_available(),
        "vector_dimension": state.index.vector_dimension.load(std::sync::atomic::Ordering::Relaxed),
    }))
}

// ---------------------------------------------------------------------------
// Filesystem probes for optional runtime dependencies
// ---------------------------------------------------------------------------

/// Platform-specific pdfium shared library filename.
#[cfg(target_os = "windows")]
const PDFIUM_LIB_NAME: &str = "pdfium.dll";

#[cfg(target_os = "linux")]
const PDFIUM_LIB_NAME: &str = "libpdfium.so";

#[cfg(target_os = "macos")]
const PDFIUM_LIB_NAME: &str = "libpdfium.dylib";

/// Checks whether the pdfium shared library is present in any of the search
/// locations: the executable directory, the current working directory, the
/// centralized runtime cache, or the system PATH. Mirrors the search order
/// in `pdfium_binding::bind_pdfium()` and `gui::panels::doctor::probe_pdfium()`.
fn probe_pdfium_available() -> bool {
    // Check the directory containing the running executable.
    if let Ok(exe_path) = std::env::current_exe()
        && let Some(exe_dir) = exe_path.parent()
        && exe_dir.join(PDFIUM_LIB_NAME).exists()
    {
        return true;
    }

    // Check the current working directory.
    if let Ok(cwd) = std::env::current_dir()
        && cwd.join(PDFIUM_LIB_NAME).exists()
    {
        return true;
    }

    // Check the centralized runtime cache directory.
    let cache_dir = neuroncite_core::paths::runtime_dir().join("pdfium");
    if cache_dir.join(PDFIUM_LIB_NAME).exists() {
        return true;
    }

    // Check system PATH directories.
    if let Some(paths) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&paths) {
            if dir.join(PDFIUM_LIB_NAME).exists() {
                return true;
            }
        }
    }

    false
}

/// Checks whether the Tesseract OCR binary is available on the system PATH,
/// in the centralized runtime cache directory, or in known installation
/// locations. Mirrors the search order in `neuroncite_pdf::deps::cached_tesseract_path()`
/// and `gui::panels::doctor::probe_tesseract()`.
fn probe_tesseract_available() -> bool {
    // Check system PATH by attempting to spawn `tesseract --version`.
    if std::process::Command::new("tesseract")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map(|mut child| {
            let _ = child.wait();
            true
        })
        .unwrap_or(false)
    {
        return true;
    }

    // Check the centralized runtime cache directory and platform-specific
    // installation locations. These checks must match the locations searched
    // by `neuroncite_pdf::deps::cached_tesseract_path()` to prevent false
    // negatives in the doctor report when Tesseract is installed but not on PATH.
    //
    // The cache_dir variable is only used on Windows and Linux. On macOS,
    // Tesseract is installed via Homebrew into fixed system paths, so the
    // cache directory is not checked.
    #[cfg(target_os = "windows")]
    {
        let cache_dir = neuroncite_core::paths::runtime_dir().join("tesseract");
        // Direct executable in cache directory.
        if cache_dir.join("tesseract.exe").exists() {
            return true;
        }
        // The UB-Mannheim NSIS installer may create a Tesseract-OCR subdirectory
        // inside the /D= target path instead of placing files directly.
        if cache_dir
            .join("Tesseract-OCR")
            .join("tesseract.exe")
            .exists()
        {
            return true;
        }
        // Default Program Files installation directory (system-wide install).
        if std::path::Path::new(r"C:\Program Files\Tesseract-OCR\tesseract.exe").exists() {
            return true;
        }
    }

    #[cfg(target_os = "linux")]
    {
        let cache_dir = neuroncite_core::paths::runtime_dir().join("tesseract");
        if cache_dir.join("tesseract.AppImage").exists() {
            return true;
        }
    }

    #[cfg(target_os = "macos")]
    {
        // Homebrew paths for macOS (Intel and Apple Silicon).
        for path in &["/usr/local/bin/tesseract", "/opt/homebrew/bin/tesseract"] {
            if std::path::Path::new(path).exists() {
                return true;
            }
        }
    }

    false
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// T-MCP-035: `probe_pdfium_available` returns a boolean without panicking.
    /// On a machine where pdfium has been downloaded to the runtime cache, this
    /// returns true; otherwise false. Either outcome is valid.
    #[test]
    fn t_mcp_035_probe_pdfium_returns_bool() {
        let result = probe_pdfium_available();
        // Verify the result is consistent with the filesystem state of the
        // runtime cache directory.
        let cache_dir = neuroncite_core::paths::runtime_dir().join("pdfium");
        let lib_exists = cache_dir.join(PDFIUM_LIB_NAME).exists();
        if lib_exists {
            assert!(
                result,
                "pdfium library file exists in cache but probe returned false"
            );
        }
        // When lib_exists is false, the probe could still return true if the
        // library is found in the exe directory, CWD, or PATH, so we do not
        // assert !result in that case.
    }

    /// T-MCP-036: `probe_tesseract_available` returns a boolean without
    /// panicking. Verifies that the function completes and the return value
    /// is consistent with the runtime cache state (including all search
    /// locations that `cached_tesseract_path()` checks in neuroncite-pdf).
    #[test]
    fn t_mcp_036_probe_tesseract_returns_bool() {
        let result = probe_tesseract_available();
        // Check all runtime cache locations for the platform-specific binary,
        // matching the search order in probe_tesseract_available().
        let cache_dir = neuroncite_core::paths::runtime_dir().join("tesseract");
        #[cfg(target_os = "windows")]
        let cached = cache_dir.join("tesseract.exe").exists()
            || cache_dir
                .join("Tesseract-OCR")
                .join("tesseract.exe")
                .exists()
            || std::path::Path::new(r"C:\Program Files\Tesseract-OCR\tesseract.exe").exists();
        #[cfg(target_os = "linux")]
        let cached = cache_dir.join("tesseract.AppImage").exists();
        #[cfg(target_os = "macos")]
        let cached = std::path::Path::new("/usr/local/bin/tesseract").exists()
            || std::path::Path::new("/opt/homebrew/bin/tesseract").exists()
            || cache_dir.join("tesseract").exists();

        if cached {
            // If the binary exists in any searched location, the probe should agree.
            assert!(
                result,
                "tesseract binary exists in cache but probe returned false"
            );
        }
    }

    /// T-MCP-037: The PDFIUM_LIB_NAME constant in system.rs matches the
    /// platform-specific shared library naming convention.
    #[test]
    fn t_mcp_037_pdfium_lib_name_matches_platform() {
        if cfg!(target_os = "windows") {
            assert_eq!(PDFIUM_LIB_NAME, "pdfium.dll");
        } else if cfg!(target_os = "linux") {
            assert_eq!(PDFIUM_LIB_NAME, "libpdfium.so");
        } else if cfg!(target_os = "macos") {
            assert_eq!(PDFIUM_LIB_NAME, "libpdfium.dylib");
        }
    }

    /// T-MCP-038: Both probe functions return consistent results across
    /// repeated calls (no flaky behavior from process spawning or file
    /// system races).
    #[test]
    fn t_mcp_038_probes_are_deterministic() {
        let pdfium_1 = probe_pdfium_available();
        let pdfium_2 = probe_pdfium_available();
        assert_eq!(
            pdfium_1, pdfium_2,
            "probe_pdfium_available should return the same value on consecutive calls"
        );

        let tesseract_1 = probe_tesseract_available();
        let tesseract_2 = probe_tesseract_available();
        assert_eq!(
            tesseract_1, tesseract_2,
            "probe_tesseract_available should return the same value on consecutive calls"
        );
    }

    /// T-MCP-039: probe_tesseract_available and cached_tesseract_path (from
    /// neuroncite-pdf) must agree on Tesseract availability. This regression test
    /// catches the BUG-1 scenario where the MCP doctor probe checked fewer
    /// filesystem locations than the actual extraction pipeline, causing a false
    /// negative "tesseract_available: false" even when Tesseract is installed.
    ///
    /// The test verifies that when Tesseract is found by `cached_tesseract_path()`
    /// in deps.rs, `probe_tesseract_available()` in system.rs also returns true.
    /// This ensures the doctor report and extraction pipeline are consistent.
    #[test]
    fn t_mcp_039_probe_consistent_with_cached_tesseract_path() {
        let cached = neuroncite_pdf::deps::cached_tesseract_path();
        let probed = probe_tesseract_available();

        if let Some(path) = cached {
            assert!(
                probed,
                "BUG-1 regression: cached_tesseract_path() found Tesseract at {:?}, \
                 but probe_tesseract_available() returned false. The doctor probe \
                 is missing search locations that the extraction pipeline checks.",
                path
            );
        }
        // When cached is None, probed may still be true (found on PATH only),
        // so we do not assert the inverse.
    }

    /// T-MCP-093b: On Windows, probe_tesseract_available must check the
    /// Tesseract-OCR subdirectory inside the runtime cache. This is the
    /// specific location where the UB-Mannheim NSIS installer places files
    /// when invoked with /D=<cache_dir>.
    #[cfg(target_os = "windows")]
    #[test]
    fn t_mcp_039b_probe_checks_tesseract_ocr_subdirectory() {
        let cache_dir = neuroncite_core::paths::runtime_dir().join("tesseract");
        let subdir_exe = cache_dir.join("Tesseract-OCR").join("tesseract.exe");

        // If the Tesseract-OCR subdirectory exists with tesseract.exe,
        // the probe must detect it. This is the BUG-1 regression scenario.
        if subdir_exe.is_file() {
            assert!(
                probe_tesseract_available(),
                "BUG-1 regression: tesseract.exe exists at {} but probe returned false",
                subdir_exe.display()
            );
        }
    }

    /// T-MCP-094c: On Windows, probe_tesseract_available must check the
    /// default Program Files installation directory.
    #[cfg(target_os = "windows")]
    #[test]
    fn t_mcp_039c_probe_checks_program_files() {
        let program_files = std::path::Path::new(r"C:\Program Files\Tesseract-OCR\tesseract.exe");

        if program_files.is_file() {
            assert!(
                probe_tesseract_available(),
                "BUG-1 regression: tesseract.exe exists at {} but probe returned false",
                program_files.display()
            );
        }
    }
}
