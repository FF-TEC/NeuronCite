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

//! Shared pdfium library binding logic used by the pdfium extraction backend
//! and the OCR module.
//!
//! The `bind_pdfium` function searches for the pdfium shared library in
//! multiple locations before falling back to auto-download. The search order:
//!
//! 1. The directory containing the current executable (`std::env::current_exe()`).
//!    This is the standard deployment location: pdfium.dll sits next to neuroncite.exe.
//!
//! 2. The current working directory. This covers development and testing scenarios
//!    where the DLL is placed in the project root.
//!
//! 3. The `<Documents>/NeuronCite/runtime/pdfium/` centralized cache directory.
//!    This is where the auto-download mechanism stores the pdfium shared library.
//!
//! 4. System library paths via `Pdfium::bind_to_system_library()`. This covers
//!    system-wide installations (e.g., pdfium in /usr/lib on Linux).
//!
//! 5. Auto-download from GitHub (bblanchon/pdfium-binaries) to the cache
//!    directory, then retry binding from there.

use pdfium_render::prelude::*;

use crate::error::PdfError;

/// Platform-specific pdfium shared library filename.
#[cfg(target_os = "windows")]
const PDFIUM_LIB_NAME: &str = "pdfium.dll";

#[cfg(target_os = "linux")]
const PDFIUM_LIB_NAME: &str = "libpdfium.so";

#[cfg(target_os = "macos")]
const PDFIUM_LIB_NAME: &str = "libpdfium.dylib";

// Emit a compile-time error on platforms where no pdfium shared library
// filename is defined. The pdfium C library is only distributed as
// pre-built binaries for Windows, Linux, and macOS.
#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
compile_error!(
    "neuroncite-pdf requires pdfium, which is only available on Windows, Linux, and macOS"
);

/// Attempts to bind to the pdfium shared library at the given directory path.
/// Returns `Ok(bindings)` on success, `Err(error_message)` on failure.
fn try_bind_at_dir(dir: &std::path::Path) -> Result<Box<dyn PdfiumLibraryBindings>, String> {
    let candidate = dir.join(PDFIUM_LIB_NAME);
    if !candidate.is_file() {
        return Err(format!(
            "{} not found at {}",
            PDFIUM_LIB_NAME,
            dir.display()
        ));
    }

    let path_str = candidate.to_string_lossy().to_string();
    tracing::debug!("attempting pdfium bind from: {path_str}");

    Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path(dir)).map_err(|e| {
        let msg = format!("pdfium bind failed at {path_str}: {e}");
        tracing::debug!("{msg}");
        msg
    })
}

/// Attempts to bind to the pdfium shared library by searching multiple
/// locations in a defined order (see module documentation). Falls back to
/// auto-downloading pdfium from GitHub if all local search paths fail.
///
/// Returns the `PdfiumLibraryBindings` on success, or a `PdfError::Pdfium`
/// describing all attempted paths and failures.
pub fn bind_pdfium() -> Result<Box<dyn PdfiumLibraryBindings>, PdfError> {
    // Candidate 1: directory containing the current executable.
    if let Ok(exe_path) = std::env::current_exe()
        && let Some(exe_dir) = exe_path.parent()
        && let Ok(bindings) = try_bind_at_dir(exe_dir)
    {
        tracing::info!("pdfium loaded from exe directory");
        return Ok(bindings);
    }

    if let Some(cache_dir) = crate::deps::cached_pdfium_path()
        && let Ok(bindings) = try_bind_at_dir(&cache_dir)
    {
        tracing::info!("pdfium loaded from cache directory");
        return Ok(bindings);
    }

    // Candidate 4: system library paths (PATH, LD_LIBRARY_PATH, etc.).
    tracing::debug!("attempting pdfium bind from system library paths");
    if let Ok(bindings) = Pdfium::bind_to_system_library() {
        tracing::info!("pdfium loaded from system library paths");
        return Ok(bindings);
    }

    // Candidate 5: auto-download pdfium to cache, then retry binding.
    tracing::info!("pdfium not found locally; triggering auto-download");
    let download_dir = crate::deps::ensure_pdfium()?;

    try_bind_at_dir(&download_dir).map_err(|bind_err| {
        PdfError::Pdfium(format!(
            "pdfium auto-download succeeded but binding failed: {bind_err}. \
             Searched: (1) exe directory, (2) CWD, (3) cache directory, \
             (4) system library paths, (5) auto-download. \
             Downloaded {PDFIUM_LIB_NAME} to {} but could not load it.",
            download_dir.display()
        ))
    })
}
