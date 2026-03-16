// Build script for neuroncite-web.
//
// The rust-embed derive macro on FrontendAssets requires the embedded folder
// (frontend/dist/) to exist at compile time. When the crate is compiled as
// part of a workspace-wide clippy or check pass without the web feature
// (e.g., `cargo clippy --workspace --features backend-ort`), the frontend
// has not been built and the dist/ directory does not exist. This causes a
// compile error because rust-embed cannot enumerate files in a missing folder.
//
// This build script creates the dist/ directory with a minimal index.html
// placeholder if it does not already exist. The placeholder is only used
// during development and CI lint passes -- production builds always run
// `npm run build` first to populate dist/ with the real SolidJS output.

use std::fs;
use std::path::Path;

fn main() {
    let dist_dir = Path::new("frontend/dist");
    if !dist_dir.exists() {
        fs::create_dir_all(dist_dir)
            .expect("failed to create frontend/dist directory for rust-embed");
        let index_path = dist_dir.join("index.html");
        fs::write(
            &index_path,
            "<!-- placeholder: run npm run build to generate the real frontend -->\n",
        )
        .expect("failed to write placeholder index.html for rust-embed");
    }

    // Rerun this build script only when the dist directory itself changes.
    // Individual file changes inside dist/ are tracked by rust-embed.
    println!("cargo:rerun-if-changed=frontend/dist");
}
