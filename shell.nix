let
  sources = import ./nix/sources.nix;

  pkgs = import sources.nixpkgs {
    config = { };
    overlays = [ (import sources.rust-overlay) ];
  };

  rust-tool-chain =
    let
      rust = pkgs.rust-bin;
    in
    if builtins.pathExists ./rust-toolchain.toml then
      rust.fromRustupToolchainFile ./rust-toolchain.toml
    else if builtins.pathExists ./rust-toolchain then
      rust.fromRustupToolchainFile ./rust-toolchain
    else
      rust.stable.latest.default.override {
        extensions = [
          "rust-src"
          "rustfmt"
        ];
      };
in
pkgs.mkShell {
  packages = with pkgs; [
    rust-tool-chain

    openssl
    libopus
    onnxruntime
    pkg-config
    cargo-deny
    cargo-edit
    cargo-watch
    rust-analyzer
  ];

  env = {
    # Required by rust-analyzer
    RUST_SRC_PATH = "${rust-tool-chain}/lib/rustlib/src/rust/library";

    # Keep ort/onnxruntime Nix-friendly: do not download ONNX Runtime at build
    # time. libonnxruntime.so is discovered via LD_LIBRARY_PATH at runtime.
    ORT_SKIP_DOWNLOAD = "1";
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.onnxruntime ];
  };
}
