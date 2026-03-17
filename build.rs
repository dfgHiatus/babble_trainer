fn main() {
    csbindgen::Builder::default()
        .input_extern_file("src/ffi.rs")
        .csharp_dll_name("babble_model")
        .csharp_class_accessibility("public")
        .csharp_namespace("babble_model.Net.Sys")
        .generate_csharp_file("bindings/NativeMethods.g.cs")
        .unwrap();
}