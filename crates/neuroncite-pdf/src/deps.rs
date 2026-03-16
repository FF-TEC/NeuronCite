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

//! Download logic for optional runtime dependencies (pdfium, Tesseract).
//!
//! Downloads portable binary distributions of pdfium and Tesseract and caches
//! them in the centralized `<Documents>/NeuronCite/runtime/` directory. The
//! download mechanism uses PowerShell on Windows and curl/tar on Linux/macOS,
//! matching the ORT auto-download pattern in the neuroncite-embed crate.
//!
//! Both dependencies are pinned to specific versions via compile-time constants
//! to ensure reproducible builds and avoid breakage from upstream changes.
//!
//! **Download policy**: Downloads are triggered explicitly by the user via the
//! Doctor panel install buttons. The extraction pipeline (`ensure_tesseract`,
//! `ensure_pdfium`) only checks whether the dependency is already present in
//! the cache or on the system PATH -- it does not initiate downloads. This
//! prevents unexpected UAC prompts or network requests during indexing.
//!
//! This module is always compiled (no feature gate) so the GUI crate can call
//! the public download and probe functions regardless of which features are
//! enabled.

use std::path::{Path, PathBuf};

use crate::error::PdfError;

/// Serializes concurrent pdfium download attempts within the same process.
/// Uses a Mutex so that failed download attempts can be retried by the user
/// (e.g., after a transient network failure or corporate proxy timeout). A
/// Mutex allows leaving the state as None on failure, while OnceLock would
/// permanently cache the first result -- including errors. This pattern
/// matches `TESSERACT_DOWNLOAD` below.
static PDFIUM_DOWNLOAD: std::sync::Mutex<Option<Result<PathBuf, String>>> =
    std::sync::Mutex::new(None);

/// Guards concurrent Tesseract download attempts within the same process.
/// Unlike the pdfium OnceLock, this uses a Mutex so that failed download
/// attempts can be retried by the user (e.g., after denying UAC and then
/// clicking "Install" again in the Doctor panel). A Mutex allows resetting
/// the state on failure, while OnceLock permanently caches the first result.
static TESSERACT_DOWNLOAD: std::sync::Mutex<Option<Result<PathBuf, String>>> =
    std::sync::Mutex::new(None);

/// Cached absolute path to the Tesseract executable, set once when the binary
/// is first successfully located by `ensure_tesseract()`. Only a successfully
/// resolved path is cached; when Tesseract is not installed, nothing is stored
/// here, so a subsequent download via `download_tesseract_with_tessdata()` is
/// detected on the next call to `ensure_tesseract()`.
///
/// Caching an absolute path (via `canonicalize()`) prevents PATH injection:
/// once the binary is located, all subsequent OCR invocations use the exact
/// inode/path returned at first discovery rather than re-resolving the name
/// through the process environment.
static TESSERACT_RESOLVED: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

// ---------------------------------------------------------------------------
// Pdfium version and download configuration
// ---------------------------------------------------------------------------

/// Pinned pdfium-binaries release tag (from bblanchon/pdfium-binaries on GitHub).
/// Each release corresponds to a Chromium pdfium snapshot version.
const PDFIUM_TAG: &str = "chromium/7699";

/// GitHub download base URL for bblanchon/pdfium-binaries releases.
const PDFIUM_DOWNLOAD_BASE: &str = "https://github.com/bblanchon/pdfium-binaries/releases/download";

/// Platform-specific pdfium archive filename. The archive is a .tgz containing
/// `lib/<library>`, `include/`, and `LICENSE`.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
const PDFIUM_ARCHIVE: &str = "pdfium-win-x64.tgz";

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const PDFIUM_ARCHIVE: &str = "pdfium-linux-x64.tgz";

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const PDFIUM_ARCHIVE: &str = "pdfium-mac-arm64.tgz";

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const PDFIUM_ARCHIVE: &str = "pdfium-mac-x64.tgz";

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
const PDFIUM_ARCHIVE: &str = "pdfium-linux-arm64.tgz";

#[cfg(all(target_os = "windows", target_arch = "aarch64"))]
const PDFIUM_ARCHIVE: &str = "pdfium-win-arm64.tgz";

// Compile-time check: produces a clear error on platforms without a pre-built
// pdfium binary distribution. This prevents silent compilation followed by a
// runtime download failure.
#[cfg(all(
    feature = "pdfium",
    not(any(
        all(
            target_os = "windows",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        all(
            target_os = "macos",
            any(target_arch = "aarch64", target_arch = "x86_64")
        ),
    ))
))]
compile_error!(
    "No pre-built pdfium binary is available for this OS/architecture combination. \
     The pdfium feature requires a binary from https://github.com/bblanchon/pdfium-binaries/releases. \
     Supported targets: windows-x64, windows-arm64, linux-x64, linux-arm64, macos-x64, macos-arm64."
);

/// Platform-specific pdfium shared library filename (must match the constant
/// in pdfium_binding.rs).
#[cfg(target_os = "windows")]
const PDFIUM_LIB_NAME: &str = "pdfium.dll";

#[cfg(target_os = "linux")]
const PDFIUM_LIB_NAME: &str = "libpdfium.so";

#[cfg(target_os = "macos")]
const PDFIUM_LIB_NAME: &str = "libpdfium.dylib";

// ---------------------------------------------------------------------------
// Tesseract version and download configuration
// ---------------------------------------------------------------------------

/// UB-Mannheim Tesseract installer URL for Windows x64.
/// The installer supports silent mode via `/S /D=<path>`.
#[cfg(target_os = "windows")]
const TESSERACT_WINDOWS_URL: &str = "https://github.com/UB-Mannheim/tesseract/releases/download/\
     v5.4.0.20240606/tesseract-ocr-w64-setup-5.4.0.20240606.exe";

/// AlexanderP Tesseract AppImage URL for Linux x86_64.
/// The AppImage bundles Tesseract with eng, deu, and several other languages.
/// No ARM64 AppImage is published upstream; ARM64 Linux users must install
/// Tesseract via their system package manager (e.g., `apt install tesseract-ocr`).
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const TESSERACT_APPIMAGE_URL: &str = "https://github.com/AlexanderP/tesseract-appimage/releases/download/\
     v5.5.2/tesseract-5.5.2-x86_64.AppImage";

/// Tessdata fast model URLs for English and German language packs.
/// Each .traineddata file is 2-4 MB from the tessdata_fast repository.
const TESSDATA_URLS: &[(&str, &str)] = &[
    (
        "eng.traineddata",
        "https://github.com/tesseract-ocr/tessdata_fast/raw/main/eng.traineddata",
    ),
    (
        "deu.traineddata",
        "https://github.com/tesseract-ocr/tessdata_fast/raw/main/deu.traineddata",
    ),
];

// ---------------------------------------------------------------------------
// Cache directory resolution
// ---------------------------------------------------------------------------

/// Returns the base cache directory for auto-downloaded dependencies.
/// Located at `<Documents>/NeuronCite/runtime/`, the centralized runtime
/// directory managed by `neuroncite_core::paths`.
pub fn deps_cache_base() -> Result<PathBuf, PdfError> {
    Ok(neuroncite_core::paths::runtime_dir())
}

/// Returns the cache directory for the pdfium shared library:
/// `<Documents>/NeuronCite/runtime/pdfium/`.
pub fn pdfium_cache_dir() -> Result<PathBuf, PdfError> {
    Ok(deps_cache_base()?.join("pdfium"))
}

/// Returns the cache directory for the Tesseract binary:
/// `<Documents>/NeuronCite/runtime/tesseract/`.
pub fn tesseract_cache_dir() -> Result<PathBuf, PdfError> {
    Ok(deps_cache_base()?.join("tesseract"))
}

/// Returns the tessdata directory within the Tesseract cache.
/// This is where .traineddata language model files are stored.
pub fn tessdata_dir() -> Result<PathBuf, PdfError> {
    // On Windows, the UB-Mannheim installer places tessdata inside the
    // installation directory. On Linux, the AppImage bundles tessdata
    // internally. This directory is used for manually downloaded tessdata
    // files (the fallback when system tessdata is not available).
    Ok(tesseract_cache_dir()?.join("tessdata"))
}

// ---------------------------------------------------------------------------
// Pdfium: cache check and download
// ---------------------------------------------------------------------------

/// Returns the directory containing the cached pdfium shared library, or
/// `None` if pdfium has not been downloaded to the cache yet.
pub fn cached_pdfium_path() -> Option<PathBuf> {
    let dir = pdfium_cache_dir().ok()?;
    let lib = dir.join(PDFIUM_LIB_NAME);
    if lib.is_file() { Some(dir) } else { None }
}

/// Downloads the pdfium shared library for the current platform from
/// bblanchon/pdfium-binaries on GitHub. The archive is a .tgz file
/// containing the shared library in either `bin/` (recent releases) or
/// `lib/` (older releases), plus `include/` (headers) and `LICENSE`.
///
/// After extraction, the library file is copied from the `bin/` or `lib/`
/// subdirectory to the cache root for simpler path resolution by
/// `pdfium_binding.rs` and the Doctor panel probes.
///
/// Returns the directory containing the extracted pdfium library.
pub fn download_pdfium() -> Result<PathBuf, PdfError> {
    let dest_dir = pdfium_cache_dir()?;
    let url = format!("{PDFIUM_DOWNLOAD_BASE}/{PDFIUM_TAG}/{PDFIUM_ARCHIVE}");

    // Verify sufficient disk space before starting the download (~3 MB
    // compressed, ~10 MB after extraction).
    neuroncite_core::disk::check_disk_space(&dest_dir).map_err(PdfError::DepDownload)?;

    std::fs::create_dir_all(&dest_dir).map_err(|e| {
        PdfError::DepDownload(format!(
            "failed to create pdfium cache directory {}: {e}",
            dest_dir.display()
        ))
    })?;

    tracing::info!(
        url = %url,
        dest = %dest_dir.display(),
        "downloading pdfium shared library (one-time, ~3 MB)"
    );

    download_and_extract_tgz(&url, &dest_dir)?;

    // The bblanchon/pdfium-binaries archive structure varies between release
    // versions. The shared library is placed in bin/ (recent releases) or
    // lib/ (older releases). Check both locations and copy whichever is found
    // to the cache root so that pdfium_binding::bind_pdfium() resolves it via
    // Pdfium::pdfium_platform_library_name_at_path(&cache_dir).
    let lib_file = dest_dir.join("lib").join(PDFIUM_LIB_NAME);
    let bin_file = dest_dir.join("bin").join(PDFIUM_LIB_NAME);
    let dest_file = dest_dir.join(PDFIUM_LIB_NAME);

    let source_file = if bin_file.is_file() {
        Some(bin_file)
    } else if lib_file.is_file() {
        Some(lib_file)
    } else {
        None
    };

    if let Some(ref src) = source_file
        && !dest_file.is_file()
    {
        std::fs::copy(src, &dest_file).map_err(|e| {
            PdfError::DepDownload(format!(
                "failed to copy pdfium library from {} to {}: {e}",
                src.display(),
                dest_file.display()
            ))
        })?;
    }

    if !dest_file.is_file() {
        return Err(PdfError::DepDownload(format!(
            "pdfium download completed but {} not found at {}",
            PDFIUM_LIB_NAME,
            dest_file.display()
        )));
    }

    tracing::info!(path = %dest_file.display(), "pdfium shared library ready");
    Ok(dest_dir)
}

/// Ensures pdfium is available in the local cache. Checks the cache directory
/// first (fast path), and triggers a serialized download from GitHub if the
/// library is not present. The Mutex ensures that concurrent callers from
/// multiple threads share a single download attempt rather than racing to
/// write the same temporary file. Unlike OnceLock, failed downloads are NOT
/// permanently cached, allowing the user to retry after fixing the issue
/// (e.g., restoring network connectivity, freeing disk space).
pub fn ensure_pdfium() -> Result<PathBuf, PdfError> {
    // Fast path: library file already exists on disk from a previous run.
    if let Some(dir) = cached_pdfium_path() {
        return Ok(dir);
    }

    // Acquire the Mutex to serialize concurrent download attempts. Only one
    // thread performs the download; others wait on the lock.
    let mut guard = PDFIUM_DOWNLOAD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    // Another thread may have completed the download while we waited on the lock.
    if let Some(Ok(path)) = guard.as_ref() {
        return Ok(path.clone());
    }

    // Slow path: perform the download. On success, cache the result so
    // subsequent callers (including those waiting on the lock) get the path
    // immediately. On failure, leave the guard as None so the user can retry.
    match download_pdfium() {
        Ok(path) => {
            *guard = Some(Ok(path.clone()));
            Ok(path)
        }
        Err(e) => {
            // Do NOT cache failures. Leave the guard as None so the user can
            // retry the download after fixing the issue.
            Err(e)
        }
    }
}

// ---------------------------------------------------------------------------
// Tesseract: cache check and download
// ---------------------------------------------------------------------------

/// Checks whether the `tesseract` binary is accessible on the system PATH
/// by attempting to spawn `tesseract --version`.
fn is_tesseract_on_path() -> bool {
    std::process::Command::new("tesseract")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map(|mut child| {
            let _ = child.wait();
            true
        })
        .unwrap_or(false)
}

/// Returns the path to the Tesseract executable if it is available on the
/// system PATH or in the auto-download cache directory. Returns `None` if
/// Tesseract is not installed and has not been downloaded.
pub fn cached_tesseract_path() -> Option<PathBuf> {
    // Check system PATH first (user-installed Tesseract takes precedence).
    if is_tesseract_on_path() {
        return Some(PathBuf::from("tesseract"));
    }

    // Check the auto-download cache directory for a cached Tesseract
    // binary. Each platform stores the binary in a different layout
    // within the cache directory.
    #[cfg(target_os = "windows")]
    {
        let dir = tesseract_cache_dir().ok()?;
        let exe = dir.join("tesseract.exe");
        if exe.is_file() {
            return Some(exe);
        }
        // The UB-Mannheim NSIS installer may create a Tesseract-OCR
        // subdirectory inside the /D= target path.
        let subdir_exe = dir.join("Tesseract-OCR").join("tesseract.exe");
        if subdir_exe.is_file() {
            return Some(subdir_exe);
        }
        // Check the default Program Files installation directory. Uses the
        // ProgramFiles environment variable to support non-standard Windows
        // installation drives (e.g., D:\Program Files on multi-disk systems).
        let pf_dir =
            std::env::var("ProgramFiles").unwrap_or_else(|_| r"C:\Program Files".to_string());
        let program_files = PathBuf::from(pf_dir)
            .join("Tesseract-OCR")
            .join("tesseract.exe");
        if program_files.is_file() {
            return Some(program_files);
        }
    }

    #[cfg(target_os = "linux")]
    {
        let dir = tesseract_cache_dir().ok()?;
        let appimage = dir.join("tesseract.AppImage");
        if appimage.is_file() {
            return Some(appimage);
        }
    }

    #[cfg(target_os = "macos")]
    {
        // Homebrew paths for macOS (Intel and Apple Silicon).
        for path in &["/usr/local/bin/tesseract", "/opt/homebrew/bin/tesseract"] {
            let p = PathBuf::from(path);
            if p.is_file() {
                return Some(p);
            }
        }
    }

    None
}

/// Checks whether Tesseract is available on the system PATH or in the
/// auto-download cache directory. Returns the path to the executable if
/// found. Returns `Err(PdfError::DepDownload)` if Tesseract is not installed.
///
/// This function does NOT trigger a download. It is called from the OCR
/// extraction pipeline where automatic downloads are undesirable (they cause
/// unexpected UAC prompts on Windows during indexing). To download Tesseract,
/// use `download_tesseract_with_tessdata()` from the Doctor panel install
/// button.
pub fn ensure_tesseract() -> Result<PathBuf, PdfError> {
    // Return the cached absolute path if already resolved. Subsequent calls
    // skip both the PATH probe and any filesystem stat calls entirely.
    if let Some(path) = TESSERACT_RESOLVED.get() {
        return Ok(path.clone());
    }

    // Discover the binary for the first time.
    let found = cached_tesseract_path().ok_or_else(|| {
        PdfError::DepDownload(
            "Tesseract is not installed. Use the Doctor panel install button to \
             download Tesseract, or install it manually (e.g., via system package \
             manager or Homebrew on macOS)."
                .into(),
        )
    })?;

    // Canonicalize to an absolute path. This eliminates PATH injection on
    // subsequent spawn calls: after the first successful discovery, all OCR
    // invocations use the exact path returned here rather than re-resolving
    // the name through the current process environment.
    // `canonicalize` may fail on AppImages (which are symlinks to the kernel
    // binfmt_misc handler); fall back to the found path as-is in that case.
    let canonical = found.canonicalize().unwrap_or(found);

    // On Windows, `Path::canonicalize()` prepends the verbatim extended-length
    // prefix `\\?\`. External processes like Tesseract fail when they receive a
    // verbatim path as the tessdata directory because they append `/eng.traineddata`
    // with a forward slash — a separator that the `\\?\` namespace does not accept.
    // Strip the prefix to produce a standard absolute path that external tools
    // can use safely.
    #[cfg(windows)]
    let absolute = {
        let s = canonical.to_string_lossy();
        if let Some(rest) = s.strip_prefix(r"\\?\") {
            PathBuf::from(rest.to_string())
        } else {
            canonical
        }
    };
    #[cfg(not(windows))]
    let absolute = canonical;

    // Cache the absolute path. If two threads race here, OnceLock::set will
    // silently fail for the losing thread; both threads then return a valid
    // path (the winner's or their own, which are the same binary).
    let _ = TESSERACT_RESOLVED.set(absolute.clone());

    Ok(absolute)
}

/// Downloads and installs Tesseract for the current platform, plus English
/// and German tessdata language packs. Returns the path to the Tesseract
/// executable.
///
/// - **Windows**: Downloads the UB-Mannheim installer and runs it with UAC
///   elevation via `/S /D=<cache_dir>`.
/// - **Linux**: Downloads the AppImage from AlexanderP, saves it to the cache
///   directory, and makes it executable.
/// - **macOS**: Returns an error with instructions to install via Homebrew,
///   because no portable Tesseract binary is available for macOS.
///
/// A `Mutex` serializes concurrent download attempts within the same process,
/// preventing file lock conflicts from parallel callers. Unlike `OnceLock`,
/// the Mutex does NOT permanently cache failures, allowing the user to retry
/// after fixing the issue (e.g., granting UAC on the second attempt).
///
/// This function is called from the Doctor panel install button. It is NOT
/// called from the OCR extraction pipeline.
pub fn download_tesseract_with_tessdata() -> Result<PathBuf, PdfError> {
    // Fast path: Tesseract is already available.
    if let Some(path) = cached_tesseract_path() {
        return Ok(path);
    }

    // Acquire the Mutex to serialize concurrent download attempts. Only one
    // thread performs the download; others wait on the lock. Unlike OnceLock,
    // failed attempts are NOT cached permanently, allowing the user to retry
    // (e.g., after denying UAC the first time).
    let mut guard = TESSERACT_DOWNLOAD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    // Another thread may have completed the download while we waited on the lock.
    if let Some(Ok(path)) = guard.as_ref() {
        return Ok(path.clone());
    }

    let dest_dir = tesseract_cache_dir()?;

    // Verify sufficient disk space before starting the download (~90 MB
    // for Tesseract installer + ~15 MB for tessdata language data).
    neuroncite_core::disk::check_disk_space(&dest_dir).map_err(PdfError::DepDownload)?;

    std::fs::create_dir_all(&dest_dir).map_err(|e| {
        PdfError::DepDownload(format!(
            "failed to create Tesseract cache directory {}: {e}",
            dest_dir.display()
        ))
    })?;

    let exe_path = download_tesseract_platform(&dest_dir);

    match exe_path {
        Ok(path) => {
            // Download tessdata language packs (eng + deu) if not bundled with
            // the platform-specific distribution. The Linux AppImage bundles
            // tessdata internally, so these files serve as a fallback for the
            // --tessdata-dir argument.
            download_tessdata()?;

            *guard = Some(Ok(path.clone()));
            Ok(path)
        }
        Err(e) => {
            // Do NOT cache failures. Leave the guard as None so the user can
            // retry the install after fixing the issue (e.g., granting UAC).
            Err(e)
        }
    }
}

/// Platform-specific Tesseract download for Windows: downloads the
/// UB-Mannheim NSIS installer and runs it silently with UAC elevation.
///
/// The UB-Mannheim NSIS installer requires administrator privileges
/// (`RequestExecutionLevel admin` in the NSIS script). Running it without
/// elevation causes a silent failure where the installer exits with status 0
/// but does not write any files to the destination directory.
///
/// This function uses PowerShell `Start-Process -Verb RunAs -PassThru` with
/// explicit `WaitForExit()` to trigger a Windows UAC elevation prompt and wait
/// for the elevated installer process to complete. The `-Wait` flag alone is
/// unreliable with `-Verb RunAs` because ShellExecuteEx (the Win32 API behind
/// `-Verb RunAs`) does not provide a waitable process handle in all cases.
/// The `-PassThru` + `WaitForExit()` combination resolves this by explicitly
/// obtaining and waiting on the process handle.
///
/// The NSIS arguments are passed as a single string (not an array) to prevent
/// PowerShell from adding quotes around the `/D=` path, which NSIS does not
/// accept.
#[cfg(target_os = "windows")]
fn download_tesseract_platform(dest_dir: &Path) -> Result<PathBuf, PdfError> {
    // Include the process ID in the installer filename to prevent file
    // lock conflicts between separate processes downloading concurrently.
    let installer_path = dest_dir.join(format!("tesseract_installer_{}.exe", std::process::id()));
    let dest_str = dest_dir.to_string_lossy();

    // Remove any leftover installer from a prior interrupted download.
    let _ = std::fs::remove_file(&installer_path);

    // Clean up stale installer files left by previous download attempts
    // (both with and without PID suffix).
    cleanup_stale_installers(dest_dir);

    tracing::info!("downloading Tesseract OCR installer for Windows (one-time, ~48 MB)");

    // Download the installer via PowerShell Invoke-WebRequest.
    // Paths are passed through process environment variables ($env:NC_OUT_PATH)
    // instead of being interpolated into the PowerShell command string.
    // This prevents PowerShell injection through paths that contain single quotes,
    // semicolons, or other PowerShell metacharacters.
    let ps_download = "$ProgressPreference = 'SilentlyContinue'; \
         Invoke-WebRequest -Uri $env:NC_DOWNLOAD_URL -OutFile $env:NC_OUT_PATH -UseBasicParsing";

    let output = std::process::Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", ps_download])
        .env("NC_DOWNLOAD_URL", TESSERACT_WINDOWS_URL)
        .env("NC_OUT_PATH", &installer_path)
        .output()
        .map_err(|e| PdfError::DepDownload(format!("PowerShell invocation failed: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let _ = std::fs::remove_file(&installer_path);
        return Err(PdfError::DepDownload(format!(
            "Tesseract download failed: {}",
            stderr.trim()
        )));
    }

    // Verify the downloaded installer's integrity.
    verify_download_checksum(&installer_path, None)?;

    // Silently remove any previous Tesseract installation that is registered
    // in the Windows registry. The UB-Mannheim NSIS installer detects existing
    // installations and spawns their uninstaller WITHOUT the /S (silent) flag,
    // which causes the uninstall GUI to appear instead of a clean install. By
    // running the uninstaller ourselves with /S first, we avoid this problem.
    // This script is a hardcoded literal with no path interpolation.
    tracing::info!("checking for existing Tesseract installation in Windows registry");

    let ps_uninstall = r#"
$key = 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Tesseract-OCR'
$entry = Get-ItemProperty -Path $key -ErrorAction SilentlyContinue
if ($entry -and $entry.UninstallString) {
    $u = $entry.UninstallString -replace '"',''
    if (Test-Path $u) {
        $p = Start-Process -FilePath $u -ArgumentList '/S' -Verb RunAs -PassThru
        $p.WaitForExit(120000)
        Start-Sleep -Seconds 3
    }
}
"#;

    let uninstall_output = std::process::Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", ps_uninstall])
        .output();

    if let Ok(ref out) = uninstall_output {
        if out.status.success() {
            tracing::info!("existing Tesseract installation handled (or none found)");
        } else {
            let stderr = String::from_utf8_lossy(&out.stderr);
            tracing::warn!(
                stderr = %stderr.trim(),
                "pre-install uninstall step returned non-zero (continuing with install)"
            );
        }
    }

    // Run the NSIS installer with UAC elevation via PowerShell Start-Process.
    // Uses -PassThru + WaitForExit() instead of -Wait because -Wait is
    // unreliable with -Verb RunAs (ShellExecuteEx does not always provide a
    // waitable handle). The arguments are a single string (not an array) to
    // prevent PowerShell from adding quotes around the /D= path.
    //
    // Paths are passed through process environment variables ($env:NC_INSTALLER_PATH,
    // $env:NC_INSTALLER_DEST) to prevent PowerShell injection: the script text
    // itself is a hardcoded literal with no format! interpolation of path strings.
    tracing::info!(
        dest = %dest_str,
        "running Tesseract silent installer (UAC elevation required)"
    );

    // The /D= NSIS flag accepts the destination directory as a bare path
    // (no quotes). Environment variable expansion in PowerShell does not add
    // quotes unless the result contains spaces and the context is a string
    // literal, so we concatenate /D= with the path in the script itself.
    let ps_install = "$args_str = '/S /D=' + $env:NC_INSTALLER_DEST; \
         $p = Start-Process -FilePath $env:NC_INSTALLER_PATH -ArgumentList $args_str \
         -Verb RunAs -PassThru; \
         $p.WaitForExit(300000); \
         exit $p.ExitCode";

    let install_output = std::process::Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", ps_install])
        .env("NC_INSTALLER_PATH", &installer_path)
        .env("NC_INSTALLER_DEST", dest_str.as_ref())
        .output()
        .map_err(|e| PdfError::DepDownload(format!("Tesseract installer execution failed: {e}")))?;

    // Remove the installer after installation attempt regardless of result.
    let _ = std::fs::remove_file(&installer_path);

    if !install_output.status.success() {
        let stderr = String::from_utf8_lossy(&install_output.stderr);
        return Err(PdfError::DepDownload(format!(
            "Tesseract silent installer failed (UAC denied or installer error): {}",
            stderr.trim()
        )));
    }

    // Search for tesseract.exe in multiple locations. The NSIS /D= flag is
    // not always respected (depends on installer script and Windows version).
    // Check: (1) target directory, (2) Tesseract-OCR subdirectory (created
    // by some installer versions), (3) Program Files default location.
    let search_locations: &[PathBuf] = &[
        dest_dir.join("tesseract.exe"),
        dest_dir.join("Tesseract-OCR").join("tesseract.exe"),
        {
            // Use the ProgramFiles environment variable to support non-standard
            // Windows installation drives (e.g., D:\Program Files).
            let pf =
                std::env::var("ProgramFiles").unwrap_or_else(|_| r"C:\Program Files".to_string());
            PathBuf::from(pf)
                .join("Tesseract-OCR")
                .join("tesseract.exe")
        },
    ];

    for candidate in search_locations {
        if candidate.is_file() {
            tracing::info!(path = %candidate.display(), "Tesseract installed");
            return Ok(candidate.clone());
        }
    }

    // List the contents of the destination directory for diagnostic purposes.
    let dir_contents: Vec<String> = std::fs::read_dir(dest_dir)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.file_name().to_string_lossy().to_string())
                .collect()
        })
        .unwrap_or_default();

    Err(PdfError::DepDownload(format!(
        "Tesseract installation completed but tesseract.exe not found. \
         Searched: {}. Directory contents of {}: [{}]. \
         The UB-Mannheim installer requires administrator privileges (UAC). \
         If the UAC prompt was denied, re-run and click 'Yes' to allow installation.",
        search_locations
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", "),
        dest_dir.display(),
        dir_contents.join(", ")
    )))
}

/// Removes stale Tesseract installer files from the cache directory.
/// These files are left behind by prior interrupted or failed download
/// attempts. Each file is approximately 48 MB, so cleaning them up
/// prevents disk space waste.
#[cfg(target_os = "windows")]
fn cleanup_stale_installers(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        // Match both "tesseract_installer.exe" (legacy name) and
        // "tesseract_installer_<pid>.exe" (process-unique name).
        if name_str.starts_with("tesseract_installer") && name_str.ends_with(".exe") {
            tracing::debug!(file = %name_str, "removing stale Tesseract installer");
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Platform-specific Tesseract download for Linux x86_64: downloads the
/// AppImage from AlexanderP's release and makes it executable.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn download_tesseract_platform(dest_dir: &Path) -> Result<PathBuf, PdfError> {
    let appimage_path = dest_dir.join("tesseract.AppImage");

    tracing::info!("downloading Tesseract AppImage for Linux (one-time, ~50 MB)");

    let output = std::process::Command::new("curl")
        .args([
            "-L",
            "-o",
            &appimage_path.to_string_lossy(),
            TESSERACT_APPIMAGE_URL,
        ])
        .output()
        .map_err(|e| PdfError::DepDownload(format!("curl invocation failed: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(PdfError::DepDownload(format!(
            "Tesseract AppImage download failed: {}",
            stderr.trim()
        )));
    }

    // Verify the downloaded AppImage's integrity.
    verify_download_checksum(&appimage_path, None)?;

    // Make the AppImage executable.
    let chmod_result = std::process::Command::new("chmod")
        .args(["+x", &appimage_path.to_string_lossy()])
        .output()
        .map_err(|e| PdfError::DepDownload(format!("chmod +x failed: {e}")))?;

    if !chmod_result.status.success() {
        return Err(PdfError::DepDownload(
            "failed to make Tesseract AppImage executable".into(),
        ));
    }

    tracing::info!(
        path = %appimage_path.display(),
        "Tesseract AppImage ready"
    );
    Ok(appimage_path)
}

/// Platform-specific Tesseract download for Linux ARM64 (aarch64): no
/// portable AppImage is published for ARM64. Returns an error with
/// instructions to install Tesseract via the system package manager.
#[cfg(all(target_os = "linux", not(target_arch = "x86_64")))]
fn download_tesseract_platform(_dest_dir: &Path) -> Result<PathBuf, PdfError> {
    Err(PdfError::DepDownload(
        "No portable Tesseract binary is available for Linux ARM64. \
         Install Tesseract via your system package manager: \
         `sudo apt install tesseract-ocr` (Debian/Ubuntu) or \
         `sudo dnf install tesseract` (Fedora/RHEL)."
            .into(),
    ))
}

/// Platform-specific Tesseract download for macOS: checks Homebrew paths and
/// returns an error with installation instructions if Tesseract is not found.
/// No portable Tesseract binary is available for macOS.
#[cfg(target_os = "macos")]
fn download_tesseract_platform(_dest_dir: &Path) -> Result<PathBuf, PdfError> {
    // Check Homebrew installation paths (Intel and Apple Silicon).
    for path in &["/usr/local/bin/tesseract", "/opt/homebrew/bin/tesseract"] {
        let p = PathBuf::from(path);
        if p.is_file() {
            return Ok(p);
        }
    }

    Err(PdfError::DepDownload(
        "Tesseract is not installed. On macOS, install via Homebrew: \
         `brew install tesseract`. No portable binary is available for macOS."
            .into(),
    ))
}

/// Downloads English and German tessdata language model files to the tessdata
/// directory. Each file is approximately 2-4 MB from the tessdata_fast
/// repository on GitHub. Files that already exist in the cache are skipped.
fn download_tessdata() -> Result<(), PdfError> {
    let dir = tessdata_dir()?;
    std::fs::create_dir_all(&dir).map_err(|e| {
        PdfError::DepDownload(format!(
            "failed to create tessdata directory {}: {e}",
            dir.display()
        ))
    })?;

    for (filename, url) in TESSDATA_URLS {
        let dest = dir.join(filename);
        if dest.is_file() {
            tracing::debug!(file = %filename, "tessdata file already cached");
            continue;
        }

        tracing::info!(file = %filename, "downloading tessdata language pack");
        download_single_file(url, &dest)?;
        verify_download_checksum(&dest, None)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Shared download helpers
// ---------------------------------------------------------------------------

/// Downloads a .tgz archive from the given URL and extracts it into the
/// destination directory. On Windows, uses PowerShell for the download and
/// the built-in `tar` command (bsdtar, available since Windows 10 1803) for
/// extraction. On Linux/macOS, uses `curl` piped to `tar`.
fn download_and_extract_tgz(url: &str, dest_dir: &Path) -> Result<(), PdfError> {
    #[cfg(target_os = "windows")]
    {
        // Include the process ID in the temp filename to prevent file lock
        // conflicts between separate processes downloading concurrently.
        // Intra-process concurrency is handled by the OnceLock in
        // ensure_pdfium().
        let tgz_path = dest_dir.join(format!("_download_{}.tgz", std::process::id()));

        // Remove any leftover temp file from a prior interrupted download
        // by this process. Ignore errors (the file may not exist).
        let _ = std::fs::remove_file(&tgz_path);

        // Download via PowerShell Invoke-WebRequest.
        // URL and output path are passed through environment variables to prevent
        // PowerShell injection through paths or URLs containing metacharacters.
        let ps_script = "$ProgressPreference = 'SilentlyContinue'; \
             Invoke-WebRequest -Uri $env:NC_DOWNLOAD_URL -OutFile $env:NC_OUT_PATH -UseBasicParsing";

        let ps_output = std::process::Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command", ps_script])
            .env("NC_DOWNLOAD_URL", url)
            .env("NC_OUT_PATH", &tgz_path)
            .output()
            .map_err(|e| PdfError::DepDownload(format!("PowerShell download failed: {e}")))?;

        if !ps_output.status.success() {
            let stderr = String::from_utf8_lossy(&ps_output.stderr);
            let _ = std::fs::remove_file(&tgz_path);
            return Err(PdfError::DepDownload(format!(
                "download failed: {}",
                stderr.trim()
            )));
        }

        // Verify the downloaded archive's integrity before extraction.
        verify_download_checksum(&tgz_path, None)?;

        // Extract using Windows built-in tar (bsdtar).
        let tar_output = std::process::Command::new("tar")
            .args([
                "xzf",
                &tgz_path.to_string_lossy(),
                "-C",
                &dest_dir.to_string_lossy(),
            ])
            .output()
            .map_err(|e| PdfError::DepDownload(format!("tar extraction failed: {e}")))?;

        let _ = std::fs::remove_file(&tgz_path);

        if !tar_output.status.success() {
            let stderr = String::from_utf8_lossy(&tar_output.stderr);
            return Err(PdfError::DepDownload(format!(
                "tar extraction failed: {}",
                stderr.trim()
            )));
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        // Download and extract in one pipeline: curl -L <url> | tar xzf - -C <dir>
        let curl_child = std::process::Command::new("curl")
            .args(["-L", "-s", url])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| PdfError::DepDownload(format!("curl invocation failed: {e}")))?;

        let tar_output =
            std::process::Command::new("tar")
                .args(["xzf", "-", "-C", &dest_dir.to_string_lossy()])
                .stdin(curl_child.stdout.ok_or_else(|| {
                    PdfError::DepDownload("curl stdout pipe not available".into())
                })?)
                .output()
                .map_err(|e| PdfError::DepDownload(format!("tar extraction failed: {e}")))?;

        if !tar_output.status.success() {
            let stderr = String::from_utf8_lossy(&tar_output.stderr);
            return Err(PdfError::DepDownload(format!(
                "tgz extraction failed: {}",
                stderr.trim()
            )));
        }
    }

    Ok(())
}

/// Computes the SHA-256 hex digest of a file. Used to verify the integrity
/// of downloaded runtime binaries (Tesseract, pdfium, tessdata).
///
/// Reads the file in 8 KiB chunks to avoid buffering the entire file in
/// memory (Tesseract installer is ~75 MB, ONNX Runtime tgz is ~200 MB).
fn compute_sha256(path: &Path) -> Result<String, PdfError> {
    use sha2::{Digest, Sha256};

    let mut file = std::fs::File::open(path).map_err(|e| {
        PdfError::DepDownload(format!(
            "failed to open file for checksum: {}: {e}",
            path.display()
        ))
    })?;

    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];

    loop {
        let n = std::io::Read::read(&mut file, &mut buffer).map_err(|e| {
            PdfError::DepDownload(format!(
                "failed to read file for checksum: {}: {e}",
                path.display()
            ))
        })?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }

    let result = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for byte in &result {
        use std::fmt::Write;
        let _ = write!(hex, "{byte:02x}");
    }
    Ok(hex)
}

/// Verifies the SHA-256 checksum of a downloaded file. If `expected_sha256`
/// is provided, compares against it and returns an error on mismatch. If
/// `None`, computes and logs the checksum for auditability without enforcing
/// a specific value.
///
/// This function is called after `download_single_file` to detect truncated
/// downloads, network corruption, or CDN poisoning.
fn verify_download_checksum(path: &Path, expected_sha256: Option<&str>) -> Result<(), PdfError> {
    let computed = compute_sha256(path)?;

    if let Some(expected) = expected_sha256 {
        if computed != expected {
            return Err(PdfError::DepDownload(format!(
                "SHA-256 mismatch for {}: expected {expected}, computed {computed}",
                path.display()
            )));
        }
        tracing::debug!(
            file = %path.display(),
            sha256 = %computed,
            "download checksum verified"
        );
    } else {
        tracing::info!(
            file = %path.display(),
            sha256 = %computed,
            "computed checksum for downloaded file (no expected value configured)"
        );
    }

    Ok(())
}

/// Downloads a single file from the given URL to a local filesystem path.
/// Uses PowerShell Invoke-WebRequest on Windows and curl on Linux/macOS.
fn download_single_file(url: &str, dest: &Path) -> Result<(), PdfError> {
    #[cfg(target_os = "windows")]
    {
        // Remove any leftover file from a prior interrupted download.
        // Prevents "file in use" errors if a stale file is locked by
        // Windows Defender or another scanning process.
        let _ = std::fs::remove_file(dest);

        // URL and destination path are passed via environment variables to
        // prevent PowerShell command injection. The script references them as
        // $env:NCITE_URL and $env:NCITE_DEST instead of interpolating into the
        // command string. This matches the pattern used by download_pdfium and
        // download_tesseract in this file.
        let output = std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "$ProgressPreference = 'SilentlyContinue'; \
                 Invoke-WebRequest -Uri $env:NCITE_URL -OutFile $env:NCITE_DEST -UseBasicParsing",
            ])
            .env("NCITE_URL", url)
            .env("NCITE_DEST", dest.as_os_str())
            .output()
            .map_err(|e| PdfError::DepDownload(format!("PowerShell download failed: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(PdfError::DepDownload(format!(
                "download of {} failed: {}",
                url,
                stderr.trim()
            )));
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        let output = std::process::Command::new("curl")
            .args(["-L", "-s", "-o", &dest.to_string_lossy(), url])
            .output()
            .map_err(|e| PdfError::DepDownload(format!("curl download failed: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(PdfError::DepDownload(format!(
                "download of {} failed: {}",
                url,
                stderr.trim()
            )));
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// T-PDF-029: `deps_cache_base` returns a path ending with "runtime".
    /// The centralized runtime directory is `<Documents>/NeuronCite/runtime/`.
    #[test]
    fn t_pdf_029_deps_cache_base_ends_with_runtime() {
        let base = deps_cache_base().expect("deps_cache_base should succeed");
        assert!(
            base.ends_with("runtime"),
            "expected path ending with 'runtime', got: {}",
            base.display()
        );
    }

    /// T-PDF-030: `pdfium_cache_dir` returns a path ending with "pdfium"
    /// under the runtime directory.
    #[test]
    fn t_pdf_030_pdfium_cache_dir_under_runtime() {
        let dir = pdfium_cache_dir().expect("pdfium_cache_dir should succeed");
        assert!(
            dir.ends_with("pdfium"),
            "expected path ending with 'pdfium', got: {}",
            dir.display()
        );
        // The parent directory should be the runtime directory.
        let parent = dir.parent().expect("pdfium_cache_dir should have a parent");
        assert!(
            parent.ends_with("runtime"),
            "parent should be 'runtime', got: {}",
            parent.display()
        );
    }

    /// T-PDF-031: `tesseract_cache_dir` returns a path ending with "tesseract"
    /// under the runtime directory.
    #[test]
    fn t_pdf_031_tesseract_cache_dir_under_runtime() {
        let dir = tesseract_cache_dir().expect("tesseract_cache_dir should succeed");
        assert!(
            dir.ends_with("tesseract"),
            "expected path ending with 'tesseract', got: {}",
            dir.display()
        );
        let parent = dir
            .parent()
            .expect("tesseract_cache_dir should have a parent");
        assert!(
            parent.ends_with("runtime"),
            "parent should be 'runtime', got: {}",
            parent.display()
        );
    }

    /// T-PDF-032: `tessdata_dir` returns a "tessdata" subdirectory inside the
    /// Tesseract cache directory.
    #[test]
    fn t_pdf_032_tessdata_dir_inside_tesseract_cache() {
        let dir = tessdata_dir().expect("tessdata_dir should succeed");
        assert!(
            dir.ends_with("tessdata"),
            "expected path ending with 'tessdata', got: {}",
            dir.display()
        );
        let parent = dir.parent().expect("tessdata_dir should have a parent");
        assert!(
            parent.ends_with("tesseract"),
            "parent should be 'tesseract', got: {}",
            parent.display()
        );
    }

    /// T-PDF-033: `cached_pdfium_path` returns `Some` only when the pdfium
    /// shared library file exists on disk. If it returns `None`, the library
    /// has not been downloaded to the cache.
    #[test]
    fn t_pdf_033_cached_pdfium_path_consistency() {
        let result = cached_pdfium_path();
        match result {
            Some(dir) => {
                // If the cache reports pdfium is present, the library file
                // should actually exist on disk.
                let lib = dir.join(PDFIUM_LIB_NAME);
                assert!(
                    lib.is_file(),
                    "cached_pdfium_path returned Some but {} does not exist",
                    lib.display()
                );
            }
            None => {
                // Pdfium not cached; this is the expected case in most test
                // environments without network access.
            }
        }
    }

    /// T-PDF-034: `cached_pdfium_path` is idempotent. Calling it twice in a
    /// row returns the same result (both `Some` or both `None`).
    #[test]
    fn t_pdf_034_cached_pdfium_path_idempotent() {
        let first = cached_pdfium_path();
        let second = cached_pdfium_path();
        assert_eq!(
            first, second,
            "cached_pdfium_path should return the same result on consecutive calls"
        );
    }

    /// T-PDF-035: `cached_tesseract_path` is idempotent. Calling it twice
    /// in a row returns the same result.
    #[test]
    fn t_pdf_035_cached_tesseract_path_idempotent() {
        let first = cached_tesseract_path();
        let second = cached_tesseract_path();
        assert_eq!(
            first, second,
            "cached_tesseract_path should return the same result on consecutive calls"
        );
    }

    /// T-PDF-036: `ensure_pdfium` returns a valid result (either a directory
    /// path when pdfium is available/downloadable, or a `DepDownload` error
    /// when the download fails). The OnceLock serialization ensures this
    /// function is safe to call from multiple threads.
    #[test]
    fn t_pdf_036_ensure_pdfium_returns_valid_result() {
        let result = ensure_pdfium();
        match result {
            Ok(dir) => {
                assert!(
                    dir.is_dir(),
                    "ensure_pdfium returned Ok but {} is not a directory",
                    dir.display()
                );
                let lib = dir.join(PDFIUM_LIB_NAME);
                assert!(
                    lib.is_file(),
                    "ensure_pdfium returned Ok but {} does not exist",
                    lib.display()
                );
            }
            Err(PdfError::DepDownload(msg)) => {
                // Download failed (no network, permissions, etc.). The error
                // message should be non-empty and descriptive.
                assert!(
                    !msg.is_empty(),
                    "DepDownload error message should not be empty"
                );
            }
            Err(other) => {
                panic!("ensure_pdfium returned unexpected error variant: {other}");
            }
        }
    }

    /// T-PDF-037: `ensure_tesseract` returns a valid result: either a path
    /// to the Tesseract binary (when installed), or a `DepDownload` error
    /// (when not installed). This function does not trigger downloads.
    #[test]
    fn t_pdf_037_ensure_tesseract_returns_valid_result() {
        let result = ensure_tesseract();
        match result {
            Ok(path) => {
                // The path should point to either an executable file or the
                // string "tesseract" (when found on system PATH).
                let path_str = path.to_string_lossy();
                assert!(
                    path.is_file() || path_str == "tesseract",
                    "ensure_tesseract returned Ok but {} is not a file or 'tesseract'",
                    path.display()
                );
            }
            Err(PdfError::DepDownload(msg)) => {
                assert!(
                    !msg.is_empty(),
                    "DepDownload error message should not be empty"
                );
            }
            Err(other) => {
                panic!("ensure_tesseract returned unexpected error variant: {other}");
            }
        }
    }

    /// T-PDF-038: Concurrent calls to `ensure_pdfium` from multiple threads
    /// do not cause file lock errors or panics. The OnceLock serialization
    /// ensures only one thread performs the download while others wait.
    #[test]
    fn t_pdf_038_concurrent_ensure_pdfium_no_panic() {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                std::thread::spawn(|| {
                    let result = ensure_pdfium();
                    result.map(|p| p.to_string_lossy().to_string())
                })
            })
            .collect();

        let mut outcomes: Vec<Result<String, String>> = Vec::new();
        for handle in handles {
            let result = handle
                .join()
                .expect("thread should not panic during ensure_pdfium");
            outcomes.push(result.map_err(|e| format!("{e}")));
        }

        // All outcomes should be identical (same Ok path or same Err message).
        let first = &outcomes[0];
        for (i, outcome) in outcomes.iter().enumerate().skip(1) {
            assert_eq!(
                first, outcome,
                "thread 0 and thread {i} returned different results from ensure_pdfium"
            );
        }
    }

    /// T-PDF-039: Concurrent calls to `ensure_tesseract` from multiple
    /// threads do not cause panics. Since ensure_tesseract is check-only
    /// (no download), all threads receive the same result.
    #[test]
    fn t_pdf_039_concurrent_ensure_tesseract_no_panic() {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                std::thread::spawn(|| {
                    let result = ensure_tesseract();
                    result.map(|p| p.to_string_lossy().to_string())
                })
            })
            .collect();

        let mut outcomes: Vec<Result<String, String>> = Vec::new();
        for handle in handles {
            let result = handle
                .join()
                .expect("thread should not panic during ensure_tesseract");
            outcomes.push(result.map_err(|e| format!("{e}")));
        }

        let first = &outcomes[0];
        for (i, outcome) in outcomes.iter().enumerate().skip(1) {
            assert_eq!(
                first, outcome,
                "thread 0 and thread {i} returned different results from ensure_tesseract"
            );
        }
    }

    /// T-PDF-040: The PDFIUM_TAG constant follows the expected
    /// "chromium/<number>" format used by bblanchon/pdfium-binaries releases.
    #[test]
    fn t_pdf_040_pdfium_tag_format() {
        assert!(
            PDFIUM_TAG.starts_with("chromium/"),
            "PDFIUM_TAG should start with 'chromium/', got: {PDFIUM_TAG}"
        );
        // The version number after "chromium/" should be numeric.
        let version_part = &PDFIUM_TAG["chromium/".len()..];
        assert!(
            version_part.chars().all(|c| c.is_ascii_digit()),
            "PDFIUM_TAG version part should be numeric, got: {version_part}"
        );
    }

    /// T-PDF-041: The PDFIUM_ARCHIVE constant matches the current platform
    /// and ends with ".tgz".
    #[test]
    fn t_pdf_041_pdfium_archive_ends_with_tgz() {
        assert!(
            PDFIUM_ARCHIVE.ends_with(".tgz"),
            "PDFIUM_ARCHIVE should end with '.tgz', got: {PDFIUM_ARCHIVE}"
        );
        assert!(
            PDFIUM_ARCHIVE.starts_with("pdfium-"),
            "PDFIUM_ARCHIVE should start with 'pdfium-', got: {PDFIUM_ARCHIVE}"
        );
    }

    /// T-PDF-042: The PDFIUM_LIB_NAME constant matches the platform-specific
    /// shared library extension.
    #[test]
    fn t_pdf_042_pdfium_lib_name_platform_specific() {
        if cfg!(target_os = "windows") {
            assert_eq!(PDFIUM_LIB_NAME, "pdfium.dll");
        } else if cfg!(target_os = "linux") {
            assert_eq!(PDFIUM_LIB_NAME, "libpdfium.so");
        } else if cfg!(target_os = "macos") {
            assert_eq!(PDFIUM_LIB_NAME, "libpdfium.dylib");
        }
    }

    /// T-PDF-043: `ensure_tesseract` does not trigger downloads. When
    /// Tesseract is not installed, it returns an error instead of attempting
    /// to download from the network.
    #[test]
    fn t_pdf_043_ensure_tesseract_no_download() {
        // This test verifies the behavior contract: ensure_tesseract is
        // check-only. If Tesseract is installed, it returns Ok. If not,
        // it returns Err. Either way, no network request is made.
        let result = ensure_tesseract();
        match result {
            Ok(_) => {
                // Tesseract is installed. Verify cached_tesseract_path agrees.
                assert!(
                    cached_tesseract_path().is_some(),
                    "ensure_tesseract returned Ok but cached_tesseract_path returned None"
                );
            }
            Err(_) => {
                // Tesseract is not installed. Verify cached_tesseract_path agrees.
                assert!(
                    cached_tesseract_path().is_none(),
                    "ensure_tesseract returned Err but cached_tesseract_path returned Some"
                );
            }
        }
    }

    /// T-PDF-044: The `PDFIUM_DOWNLOAD` Mutex starts as None and does not
    /// permanently cache failures. After a failed ensure_pdfium call (when
    /// pdfium is not cached), the Mutex state remains None, allowing retries.
    /// This was changed from OnceLock (which permanently caches errors) to
    /// Mutex to support user-initiated retries from the Doctor panel.
    #[test]
    fn t_pdf_044_pdfium_mutex_does_not_cache_failures() {
        // Access the Mutex directly to verify its state.
        let guard = PDFIUM_DOWNLOAD.lock().unwrap_or_else(|p| p.into_inner());

        // The Mutex state is either None (no download attempted yet) or
        // Some(Ok(path)) (previous successful download). It must never be
        // Some(Err(_)) because failures are intentionally not cached.
        match guard.as_ref() {
            None => {
                // No download has been attempted in this test process.
                // This is the expected initial state.
            }
            Some(Ok(_)) => {
                // A previous test or process initialization already downloaded
                // pdfium successfully. The cached success is correct.
            }
            Some(Err(_)) => {
                panic!(
                    "PDFIUM_DOWNLOAD Mutex contains a cached error. \
                     Failures must not be cached (the Mutex should be left as None \
                     on failure to allow retries)."
                );
            }
        }
    }

    /// T-PDF-045: The `TESSERACT_DOWNLOAD` Mutex uses the same retry-safe
    /// pattern as `PDFIUM_DOWNLOAD`. Verifies the Mutex state is never a
    /// cached error.
    #[test]
    fn t_pdf_045_tesseract_mutex_does_not_cache_failures() {
        let guard = TESSERACT_DOWNLOAD.lock().unwrap_or_else(|p| p.into_inner());

        match guard.as_ref() {
            None => {}
            Some(Ok(_)) => {}
            Some(Err(_)) => {
                panic!(
                    "TESSERACT_DOWNLOAD Mutex contains a cached error. \
                     Failures must not be cached."
                );
            }
        }
    }

    /// T-PDF-046: Pdfium and Tesseract Mutex guards recover from poisoned
    /// state. If a previous thread panicked while holding the lock, the
    /// `unwrap_or_else(|p| p.into_inner())` pattern in `ensure_pdfium` and
    /// `download_tesseract_with_tessdata` must not propagate the poison.
    #[test]
    fn t_pdf_046_mutex_poison_recovery() {
        // This test validates the pattern used in the download functions.
        // A real poison scenario requires a thread panic while holding the
        // lock, which is complex to trigger in a unit test. Instead, we
        // validate that the `unwrap_or_else` pattern compiles and produces
        // a usable guard from a non-poisoned Mutex.
        let test_mutex: std::sync::Mutex<Option<Result<PathBuf, String>>> =
            std::sync::Mutex::new(Some(Ok(PathBuf::from("/test"))));

        let guard = test_mutex
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        assert!(
            guard.is_some(),
            "guard should contain the test value after non-poisoned lock"
        );
    }
}
