# scripts/package.ps1 — portable Windows package of tts-server (no Python).
#
# Windows counterpart of scripts/package.sh: builds with MSVC, dynamically
# loaded ggml backends and every CPU variant, then packs
#   dist\qwen-tts-<version>-windows-x64-<variant>.zip (+ .sha256)
# with tts-server.exe (web UI embedded), the CLI tools, the ggml DLLs and
# backends, the model download scripts and a README.
#
# Usage (from a "x64 Native Tools" / VS developer shell, or any shell where
# cmake finds Visual Studio):
#   pwsh scripts/package.ps1 -Variant cpu
#   pwsh scripts/package.ps1 -Variant vulkan   # needs the Vulkan SDK (VULKAN_SDK set)
#   pwsh scripts/package.ps1 -Variant cuda     # needs the CUDA toolkit (CUDA_PATH set)
param(
    [ValidateSet("cpu", "vulkan", "cuda")] [string] $Variant = "cpu",
    [string] $BuildDir = "",
    [string] $Version = "",
    [string] $CMakeArgs = "",
    [int] $Jobs = 0   # parallel compile jobs, 0 = all cores
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$Root = (Get-Location).Path

if (-not $BuildDir) { $BuildDir = "build-pkg-$Variant" }
if (-not $Version) {
    $Version = (git describe --tags --always --dirty 2>$null)
    if (-not $Version) { $Version = "dev" }
}
$Name  = "qwen-tts-$Version-windows-x64-$Variant"
$Out   = Join-Path $Root "dist"
$Stage = Join-Path $Out $Name

$backend = @()
if ($Variant -eq "vulkan") { $backend = @("-DGGML_VULKAN=ON") }
if ($Variant -eq "cuda")   { $backend = @("-DGGML_CUDA=ON", "-DGGML_CUDA_GRAPHS=ON") }

$cfg = @("-S", ".", "-B", $BuildDir,
         "-DGGML_BACKEND_DL=ON", "-DBUILD_SHARED_LIBS=ON",
         "-DGGML_CPU_ALL_VARIANTS=ON", "-DGGML_NATIVE=OFF", "-DGGML_BLAS=OFF") + $backend
if ($CMakeArgs) { $cfg += $CMakeArgs.Split(" ") }
cmake @cfg
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }
if ($Jobs -le 0) { $Jobs = [int]$env:NUMBER_OF_PROCESSORS }
cmake --build $BuildDir --config Release -j $Jobs
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }

# Multi-config generators put everything under <build>\Release.
$Bin = Join-Path $BuildDir "Release"
if (-not (Test-Path (Join-Path $Bin "tts-server.exe"))) { $Bin = $BuildDir }
if (-not (Test-Path (Join-Path $Bin "tts-server.exe"))) { throw "tts-server.exe not found under $BuildDir" }

if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
New-Item -ItemType Directory -Force -Path (Join-Path $Stage "scripts") | Out-Null

foreach ($exe in "tts-server.exe", "qwen-tts.exe", "qwen-codec.exe", "quantize.exe") {
    $p = Join-Path $Bin $exe
    if (Test-Path $p) { Copy-Item $p $Stage }
}
# ggml core DLLs and every backend DLL ggml_backend_load_all() can pick up.
Get-ChildItem $Bin -Filter "ggml*.dll" | Copy-Item -Destination $Stage

# MSVC runtime + OpenMP (vcomp140.dll, used by the CPU backend): a clean
# Windows has neither unless the VC++ redistributable is installed, so
# ship them next to the binaries.
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $vs = & $vswhere -latest -property installationPath
    $redist = Get-ChildItem (Join-Path $vs "VC\Redist\MSVC") -Directory -ErrorAction SilentlyContinue |
              Sort-Object Name -Descending | Select-Object -First 1
    if ($redist) {
        foreach ($sub in "Microsoft.VC14*.CRT", "Microsoft.VC14*.OpenMP") {
            Get-ChildItem (Join-Path $redist.FullName "x64") -Directory -Filter $sub -ErrorAction SilentlyContinue |
                ForEach-Object { Get-ChildItem $_.FullName -Filter "*.dll" | Copy-Item -Destination $Stage }
        }
    } else { Write-Warning "VC redist folder not found ; the package will need the VC++ redistributable" }
} else { Write-Warning "vswhere not found ; the package will need the VC++ redistributable" }

if ($Variant -eq "cuda") {
    # CUDA runtime + cuBLAS next to the backend ; the driver comes from the system.
    $cudaBin = Join-Path $env:CUDA_PATH "bin"
    foreach ($pattern in "cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll") {
        Get-ChildItem $cudaBin -Filter $pattern | Copy-Item -Destination $Stage
    }
}

Copy-Item scripts\download-*.sh (Join-Path $Stage "scripts")
Copy-Item LICENSE $Stage
@"
qwen-tts $Version ($Variant, Windows x64)

1. Download a model and the audio codec into .\models from
   https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF
     qwen-talker-0.6b-customvoice-Q4_K_M.gguf   (talker, voices: serena, vivian, ...)
     qwen-tokenizer-12hz-Q4_K_M.gguf            (audio codec, needed by every model)
   (scripts\download-*.sh do the same from Git Bash / WSL.)

2. Start the server:
     .\tts-server.exe --model models\qwen-talker-0.6b-customvoice-Q4_K_M.gguf ^
                      --codec models\qwen-tokenizer-12hz-Q4_K_M.gguf --port 8080

3. Open http://127.0.0.1:8080/ (web UI) or call POST /v1/audio/speech.

The CPU backend for this processor is chosen at start (ggml-cpu-*.dll).
Force a backend with the environment variable GGML_BACKEND=CPU (or CUDA0 / Vulkan0).
.\tts-server.exe --help lists all options.
"@ | Set-Content -Encoding UTF8 (Join-Path $Stage "README.txt")

# Smoke check: the binary starts from the package folder.
& (Join-Path $Stage "tts-server.exe") --help | Out-Null
if ($LASTEXITCODE -ne 0) { throw "tts-server.exe does not start from the package folder" }

$Zip = Join-Path $Out "$Name.zip"
if (Test-Path $Zip) { Remove-Item $Zip }
Compress-Archive -Path $Stage -DestinationPath $Zip
$hash = (Get-FileHash -Algorithm SHA256 $Zip).Hash.ToLower()
"$hash  $Name.zip" | Set-Content -Encoding ASCII "$Zip.sha256"
Write-Host "Package: $Zip"
