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

//! Dependency probe handlers for the web frontend.
//!
//! Checks whether external dependencies (pdfium, tesseract, ONNX Runtime,
//! Ollama, poppler) are available on the system. Each dependency is probed by
//! checking for the presence of its shared library, executable in the system
//! PATH, or service reachability via HTTP.
//!
//! The install handler delegates to the download functions in
//! `neuroncite_pdf::deps` and `neuroncite_embed::cache` which download portable
//! binary distributions from GitHub and cache them in the
//! `<Documents>/NeuronCite/runtime/` directory.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use crate::WebState;

/// Single dependency probe result sent to the frontend.
#[derive(Serialize)]
pub struct DependencyProbe {
    /// Dependency name (e.g., "pdfium", "ONNX Runtime (GPU)").
    pub name: String,
    /// Short identifier accepted by the POST /api/v1/web/doctor/install
    /// endpoint (e.g., "pdfium", "tesseract", "onnxruntime"). Empty string
    /// for dependencies that cannot be auto-installed (Ollama, poppler).
    /// The frontend uses this value directly in install requests instead of
    /// deriving it from the display name via string matching.
    pub install_id: String,
    /// One-line description of what this dependency does, shown beneath the
    /// name in both the WelcomeDialog and DoctorPanel.
    pub purpose: String,
    /// Whether the dependency is available on this system.
    pub available: bool,
    /// Whether the dependency can be installed automatically via the web UI
    /// (Install button shown).
    pub installable: bool,
    /// Hint text shown when the dependency is not available. Describes the
    /// dependency's purpose and (for non-installable deps) manual install steps.
    pub hint: String,
    /// URL for manual installation instructions. Shown as a clickable link when
    /// the dependency is neither available nor auto-installable. Empty string
    /// when no external link is relevant.
    pub link: String,
    /// Short version string for display (e.g., "1.23.2", "v5.4.0"). Empty
    /// string when no specific version is tracked.
    pub version: String,
}

/// Runs all dependency probes and returns the results.
/// Probes check for pdfium, tesseract, ONNX Runtime, Ollama, and
/// poppler/pdftotext. The Ollama probe requires an async HTTP request;
/// all other probes are synchronous filesystem checks.
pub async fn run_probes(State(_state): State<Arc<WebState>>) -> Json<Vec<DependencyProbe>> {
    // Ollama probe: async HTTP check against localhost:11434
    let ollama_available = probe_ollama().await;

    let probes = vec![
        DependencyProbe {
            name: "pdfium".to_string(),
            install_id: "pdfium".to_string(),
            purpose: "Extracts text and structure from PDF files".to_string(),
            available: probe_pdfium(),
            installable: true,
            hint: "Required for PDF text extraction. Click Install to download automatically."
                .to_string(),
            link: "https://github.com/nicehash/pdfium-binaries/releases".to_string(),
            version: String::new(),
        },
        DependencyProbe {
            name: "tesseract".to_string(),
            install_id: "tesseract".to_string(),
            purpose: "Reads text from scanned or image-based pages (OCR)".to_string(),
            available: neuroncite_pdf::deps::cached_tesseract_path().is_some(),
            installable: true,
            hint: "Required for OCR fallback on image-heavy pages. Click Install to download."
                .to_string(),
            link: "https://github.com/tesseract-ocr/tesseract".to_string(),
            version: String::new(),
        },
        probe_onnx_runtime(),
        DependencyProbe {
            name: "Ollama".to_string(),
            install_id: String::new(),
            purpose: "Local LLM server — optional, used for citation verification agents"
                .to_string(),
            available: ollama_available,
            installable: false,
            hint: "Local LLM inference server for citation verification sub-agents. \
                   Download and install from ollama.com."
                .to_string(),
            link: "https://ollama.com/download".to_string(),
            version: String::new(),
        },
        DependencyProbe {
            name: "poppler (pdftotext)".to_string(),
            install_id: String::new(),
            purpose: "Alternative PDF extractor — optional if pdfium is installed".to_string(),
            available: probe_executable("pdftotext"),
            installable: false,
            hint: "Alternative PDF text extractor. Not required when pdfium is installed."
                .to_string(),
            link: "https://poppler.freedesktop.org/".to_string(),
            version: String::new(),
        },
    ];

    Json(probes)
}

/// Checks whether pdfium is available by looking for the shared library
/// in the neuroncite executable directory or the pdfium cache directory.
fn probe_pdfium() -> bool {
    // Check the directory next to the running executable
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        let pdfium_name = if cfg!(target_os = "windows") {
            "pdfium.dll"
        } else if cfg!(target_os = "macos") {
            "libpdfium.dylib"
        } else {
            "libpdfium.so"
        };
        if dir.join(pdfium_name).exists() {
            return true;
        }
    }

    // Check the NeuronCite cache directory where download_pdfium() stores the library
    neuroncite_pdf::deps::cached_pdfium_path().is_some()
}

/// Probes for ONNX Runtime shared library availability. Checks the
/// `<Documents>/NeuronCite/runtime/ort/` cache directory for the
/// platform-specific shared library file.
///
/// On Windows and macOS, ONNX Runtime is auto-installable via the
/// `ensure_ort_runtime()` download function. On Linux, system-wide
/// installation is expected (via Docker image or package manager).
fn probe_onnx_runtime() -> DependencyProbe {
    let ort_dir = neuroncite_core::paths::runtime_dir().join("ort");

    // All platforms with backend-ort support auto-download from GitHub.
    let (lib_name, installable) = if cfg!(target_os = "windows") {
        ("onnxruntime.dll", true)
    } else if cfg!(target_os = "macos") {
        ("libonnxruntime.dylib", true)
    } else {
        ("libonnxruntime.so", true)
    };

    let available = ort_dir.join(lib_name).exists();

    // Detect GPU variant by checking for the CUDA provider shared library.
    // Windows uses .dll, Linux uses .so extensions.
    let variant = if available {
        let has_cuda = if cfg!(target_os = "windows") {
            ort_dir.join("onnxruntime_providers_cuda.dll").exists()
        } else if cfg!(target_os = "linux") {
            ort_dir.join("libonnxruntime_providers_cuda.so").exists()
        } else {
            false
        };
        if has_cuda { " (GPU)" } else { " (CPU)" }
    } else {
        ""
    };

    DependencyProbe {
        name: format!("ONNX Runtime{variant}"),
        install_id: "onnxruntime".to_string(),
        purpose: "AI inference engine — required to run embedding and reranker models".to_string(),
        available,
        installable,
        hint: "Required inference backend for embedding models. \
               Click Install to download from Microsoft GitHub."
            .to_string(),
        link: "https://github.com/microsoft/onnxruntime/releases".to_string(),
        version: String::new(),
    }
}

/// Checks whether an Ollama instance is reachable at the default localhost
/// address. Delegates to `OllamaBackend::is_reachable()` which sends a
/// lightweight HTTP GET to /api/tags with a short timeout. Returns false
/// if the HTTP client cannot be constructed (TLS backend unavailable) or
/// if the Ollama server does not respond.
async fn probe_ollama() -> bool {
    let config = neuroncite_llm::types::LlmConfig {
        base_url: "http://localhost:11434".to_string(),
        model: String::new(),
        temperature: 0.0,
        max_tokens: 0,
        json_mode: false,
    };
    match neuroncite_llm::ollama::OllamaBackend::new(config) {
        Ok(backend) => backend.is_reachable().await,
        Err(_) => false,
    }
}

/// Checks whether an executable is available in the system PATH.
fn probe_executable(name: &str) -> bool {
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("where")
            .arg(name)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    #[cfg(not(target_os = "windows"))]
    {
        std::process::Command::new("which")
            .arg(name)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }
}

/// Request body for POST /api/v1/web/doctor/install.
#[derive(Deserialize)]
pub struct InstallRequest {
    /// Dependency name to install (e.g., "pdfium", "tesseract", "onnxruntime").
    pub dependency: String,
}

/// Downloads and installs a dependency. Delegates to the download functions
/// in `neuroncite_pdf::deps` and `neuroncite_embed::cache` which download
/// portable binary distributions from GitHub and cache them locally. The
/// download runs on a blocking thread via `tokio::task::spawn_blocking`
/// because the download functions perform synchronous HTTP requests and
/// filesystem I/O.
///
/// Supported dependencies: "pdfium", "tesseract", "onnxruntime".
/// Returns 202 Accepted with the installed dependency name on success,
/// 400 for unknown dependencies, 500 on download failure.
pub async fn install_dependency(
    State(_state): State<Arc<WebState>>,
    Json(req): Json<InstallRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    match req.dependency.as_str() {
        "pdfium" => {
            let result = tokio::task::spawn_blocking(neuroncite_pdf::deps::download_pdfium).await;
            match result {
                Ok(Ok(path)) => (
                    StatusCode::ACCEPTED,
                    Json(serde_json::json!({
                        "status": "installed",
                        "dependency": "pdfium",
                        "path": path.to_string_lossy()
                    })),
                ),
                Ok(Err(e)) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": format!("Pdfium download failed: {e}")
                    })),
                ),
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": format!("Install task panicked: {e}")
                    })),
                ),
            }
        }
        "tesseract" => {
            let result =
                tokio::task::spawn_blocking(neuroncite_pdf::deps::download_tesseract_with_tessdata)
                    .await;
            match result {
                Ok(Ok(path)) => (
                    StatusCode::ACCEPTED,
                    Json(serde_json::json!({
                        "status": "installed",
                        "dependency": "tesseract",
                        "path": path.to_string_lossy()
                    })),
                ),
                Ok(Err(e)) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": format!("Tesseract download failed: {e}")
                    })),
                ),
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": format!("Install task panicked: {e}")
                    })),
                ),
            }
        }
        "onnxruntime" => install_onnxruntime().await,
        _ => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!(
                    "Unknown dependency: '{}'. Supported: 'pdfium', 'tesseract', 'onnxruntime'.",
                    req.dependency
                )
            })),
        ),
    }
}

/// Downloads and installs the ONNX Runtime shared library. With the
/// `backend-ort` feature enabled, delegates to
/// `neuroncite_embed::ensure_ort_runtime()` which downloads the correct
/// platform variant from Microsoft's GitHub releases.
/// Auto-download is available on Windows, macOS, and Linux (x64 and ARM64).
async fn install_onnxruntime() -> (StatusCode, Json<serde_json::Value>) {
    #[cfg(feature = "backend-ort")]
    {
        let result = tokio::task::spawn_blocking(neuroncite_embed::ensure_ort_runtime).await;
        match result {
            Ok(Ok(path)) => (
                StatusCode::ACCEPTED,
                Json(serde_json::json!({
                    "status": "installed",
                    "dependency": "onnxruntime",
                    "path": path.to_string_lossy()
                })),
            ),
            Ok(Err(e)) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("ONNX Runtime download failed: {e}")
                })),
            ),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("Install task panicked: {e}")
                })),
            ),
        }
    }

    #[cfg(not(feature = "backend-ort"))]
    {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "ONNX Runtime auto-install requires the backend-ort feature."
            })),
        )
    }
}
