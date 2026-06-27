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
    pkg-config
    cargo-deny
    cargo-edit
    cargo-watch
    rust-analyzer
  ];

  env = {
    # Required by rust-analyzer
    RUST_SRC_PATH = "${rust-tool-chain}/lib/rustlib/src/rust/library";
  };
}
