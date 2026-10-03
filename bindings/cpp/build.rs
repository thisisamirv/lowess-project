use std::env;
use std::fs::{create_dir_all, read_to_string, write};
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let output_file = PathBuf::from(&crate_dir)
        .join("include")
        .join("fastlowess.h");

    // Create include directory if it doesn't exist
    create_dir_all(PathBuf::from(&crate_dir).join("include")).unwrap();

    generate_version_header(&crate_dir);

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
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=cmake/fastlowess_version.h.in");
}

fn generate_version_header(crate_dir: &str) {
    let directory = PathBuf::from(crate_dir);
    let template = read_to_string(directory.join("cmake/fastlowess_version.h.in"))
        .expect("Unable to read version header template");
    let header = template
        .replace(
            "@FASTLOWESS_CPP_VERSION_MAJOR@",
            env!("CARGO_PKG_VERSION_MAJOR"),
        )
        .replace(
            "@FASTLOWESS_CPP_VERSION_MINOR@",
            env!("CARGO_PKG_VERSION_MINOR"),
        )
        .replace(
            "@FASTLOWESS_CPP_VERSION_PATCH@",
            env!("CARGO_PKG_VERSION_PATCH"),
        )
        .replace("@FASTLOWESS_CPP_VERSION_STRING@", env!("CARGO_PKG_VERSION"));
    let output = directory.join("include/fastlowess_version.h");
    if read_to_string(&output).ok().as_deref() != Some(header.as_str()) {
        write(output, header).expect("Unable to write generated version header");
    }
}

fn normalize_generated_header(output_file: &PathBuf) {
    let header = read_to_string(output_file).expect("Unable to read generated bindings");

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
        write(output_file, normalized).expect("Unable to write normalized bindings");
    }
}
