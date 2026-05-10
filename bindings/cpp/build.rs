use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let output_file = PathBuf::from(&crate_dir)
        .join("include")
        .join("fastlowess.h");

    // Create include directory if it doesn't exist
    std::fs::create_dir_all(PathBuf::from(&crate_dir).join("include")).unwrap();

    // Generate C header
    cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(
            cbindgen::Config::from_file(PathBuf::from(&crate_dir).join("cbindgen.toml")).unwrap(),
        )
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(&output_file);

    normalize_generated_header(&output_file);

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}

fn normalize_generated_header(output_file: &PathBuf) {
    let header = std::fs::read_to_string(output_file).expect("Unable to read generated bindings");

    let normalized = header
        .replace(
            "#include <cstdarg>\n#include <cstdint>\n#include <cstdlib>\n#include <ostream>\n#include <new>\n\n",
            "",
        )
        .replace(
            "`ptr` must be a valid CppLowess pointer. `x` and `y` must be valid arrays of length `n`.",
            "`ptr` must be a valid CppLowess pointer. `x_values` and `y_values` must be valid arrays of length `n`.",
        )
        .replace(
            "`ptr` must be valid. `x` and `y` must be valid arrays of length `n`.",
            "`ptr` must be valid. `x_values` and `y_values` must be valid arrays of length `n`.",
        )
        .replace("const double *x,", "const double *x_values,")
        .replace("const double *y,", "const double *y_values,");

    if normalized != header {
        std::fs::write(output_file, normalized).expect("Unable to write normalized bindings");
    }
}
