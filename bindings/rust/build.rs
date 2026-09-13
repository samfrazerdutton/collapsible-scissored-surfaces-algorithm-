// Locates the built libcsa shared library (same discovery rule as
// bindings/python/csa.py: CSA_LIB_PATH env var to an exact file, else
// ../../build/<platform name> relative to this crate), links against it,
// and -- since a dynamic library also has to be *found at runtime*, not
// just at link time -- copies the actual .dll/.so/.dylib next to the
// binaries cargo produces, so `cargo test`/`cargo run` work without the
// caller having to fix up PATH/LD_LIBRARY_PATH themselves.
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn platform_names() -> (&'static str, &'static str) {
    // (import-library-or-shared-object name for linking, runtime file name)
    if cfg!(target_os = "windows") {
        ("csa", "csa.dll") // link against csa.lib (MSVC import lib), copy csa.dll
    } else if cfg!(target_os = "macos") {
        ("csa", "libcsa.dylib")
    } else {
        ("csa", "libcsa.so")
    }
}

fn find_lib_dir_and_runtime_file(runtime_name: &str) -> (PathBuf, PathBuf) {
    if let Ok(exact) = env::var("CSA_LIB_PATH") {
        let p = PathBuf::from(&exact);
        if p.exists() {
            let dir = p.parent().unwrap_or(Path::new(".")).to_path_buf();
            return (dir, p);
        }
        panic!("CSA_LIB_PATH is set to {exact:?} but that file does not exist");
    }

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let candidate_dir = Path::new(&manifest_dir).join("..").join("..").join("build");
    let candidate_file = candidate_dir.join(runtime_name);
    if candidate_file.exists() {
        return (candidate_dir, candidate_file);
    }

    panic!(
        "Could not locate the libcsa shared library at {candidate_file:?}. \
         Build the C++ project first (cmake --build build from the repo root), \
         or set CSA_LIB_PATH to the exact .dll/.so/.dylib path."
    );
}

// cargo places build script output under
// target/<profile>/build/<pkg>-<hash>/out; the actual test/example
// binaries cargo runs live in target/<profile>/ and target/<profile>/deps/
// -- walk up from OUT_DIR to find target/<profile>/ and copy the runtime
// library into both, since Windows' DLL search order includes the
// directory the launching .exe lives in (deps/ for test binaries).
fn copy_runtime_lib_next_to_test_binaries(runtime_file: &Path, runtime_name: &str) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    // out_dir = .../target/<profile>/build/<pkg-hash>/out
    if let Some(profile_dir) = out_dir
        .ancestors()
        .nth(3) // out -> pkg-hash -> build -> <profile>
    {
        let _ = fs::create_dir_all(profile_dir);
        let _ = fs::copy(runtime_file, profile_dir.join(runtime_name));
        let deps_dir = profile_dir.join("deps");
        let _ = fs::create_dir_all(&deps_dir);
        let _ = fs::copy(runtime_file, deps_dir.join(runtime_name));
    }
}

fn main() {
    let (link_name, runtime_name) = platform_names();
    let (lib_dir, runtime_file) = find_lib_dir_and_runtime_file(runtime_name);

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib={link_name}");

    if !cfg!(target_os = "windows") {
        // Best-effort rpath so a non-Windows build can also find the
        // .so/.dylib at runtime without LD_LIBRARY_PATH gymnastics.
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    }

    copy_runtime_lib_next_to_test_binaries(&runtime_file, runtime_name);

    println!("cargo:rerun-if-env-changed=CSA_LIB_PATH");
    println!("cargo:rerun-if-changed={}", runtime_file.display());
}
