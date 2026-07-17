{
  description = "lePokeRed — a LeWorldModel for playing Pokémon Red";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true; # required for CUDA
        };

        python = pkgs.python313;

        # Native libraries that Python wheels (numpy, torch, pygame/emulator
        # bindings, opencv, ...) dlopen at runtime. Extend as dependencies grow.
        libs = with pkgs; [
          stdenv.cc.cc.lib
          zlib
          glib
          libGL
          libx11
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [ python pkgs.uv ];

          env = {
            # Have uv download real CPython wheels rather than manage a Python
            # toolchain itself, and point it at the Nix python.
            UV_PYTHON_DOWNLOADS = "never";
            UV_PYTHON = python.interpreter;
          };

          # PyTorch's PyPI wheels bundle their own CUDA runtime (cuDNN, cuBLAS,
          # NCCL, ...). The only thing that must come from the host is the
          # driver's libcuda.so, which NixOS exposes at /run/opengl-driver/lib.
          # Harmlessly ignored on machines without a GPU, so it's always on.
          LD_LIBRARY_PATH =
            pkgs.lib.makeLibraryPath libs + ":/run/opengl-driver/lib";

          shellHook = ''
            echo "lePokeRed dev shell — Python ${python.version} + uv"
            # Create/refresh the local venv from pyproject.toml on entry.
            if [ ! -d .venv ]; then
              uv venv --python ${python.interpreter}
            fi
            source .venv/bin/activate

            if [ -e /run/opengl-driver/lib/libcuda.so ]; then
              echo "GPU: NVIDIA driver detected."
            else
              echo "GPU: no NVIDIA driver at /run/opengl-driver/lib —" \
                   "torch will run on CPU (on non-NixOS, use nixGL)."
            fi
          '';
        };
      });
}
