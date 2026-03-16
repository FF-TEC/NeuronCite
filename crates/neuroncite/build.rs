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

// Build script for the neuroncite binary crate.
//
// Responsibilities:
//
// 1. Suppresses the LIBCMT linker warning on MSVC. With the `load-dynamic`
//    feature on `ort`, ONNX Runtime is loaded as a shared library at runtime
//    via `libloading`. This avoids the CRT conflict between esaxx-rs (/MT)
//    and ort-sys (/MD). The /NODEFAULTLIB:LIBCMT flag is emitted defensively.
//
// 2. Embeds the application icon (assets/icon.ico) into the Windows
//    executable via winres. This makes the .exe display the NeuronCite
//    gradient "C" icon in Windows Explorer, the taskbar, and Alt+Tab.

fn main() {
    // On Windows with the MSVC toolchain, suppress the static C runtime
    // library (LIBCMT) to prevent LNK4098 linker warnings. This conflict
    // arises when one crate links against the static CRT (/MT) while
    // another uses the dynamic CRT (/MD). The esaxx-rs crate (transitive
    // dependency via tokenizers) compiles with /MT, while ort-sys and the
    // standard Rust runtime use /MD.
    #[cfg(all(target_os = "windows", target_env = "msvc"))]
    {
        println!("cargo:rustc-link-arg=/NODEFAULTLIB:LIBCMT");
    }

    // Embed the application icon into the Windows executable resource
    // section. The icon.ico file contains multiple resolutions (16px through
    // 256px) so Windows can select the appropriate size for Explorer
    // thumbnails (large), taskbar pins (medium), and title bars (small).
    // On non-Windows targets, winres is not compiled and this block is skipped.
    #[cfg(target_os = "windows")]
    {
        let mut res = winres::WindowsResource::new();
        res.set_icon("assets/icon.ico");
        if let Err(e) = res.compile() {
            // Print the error but do not fail the build. The application
            // functions without an embedded icon -- Windows falls back to
            // the default executable icon.
            println!("cargo:warning=winres icon embedding failed: {e}");
        }
    }
}
