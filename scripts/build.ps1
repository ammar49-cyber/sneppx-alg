param(
    [ValidateSet("Debug", "Release", "RelWithDebInfo")]
    [string]$Config = "Release",
    [switch]$LTO,
    [switch]$Asan,
    [switch]$Tests = $true,
    [switch]$Benchmarks = $true,
    [switch]$Python
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $RootDir "build"

# Ensure build directory
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
}

# Configure
$cmakeArgs = @(
    "-B", $BuildDir,
    "-G", "Visual Studio 17 2022",
    "-A", "x64"
)

if ($Python) {
    $cmakeArgs += "-DARIX_BUILD_PYTHON=ON"
} else {
    $cmakeArgs += "-DARIX_BUILD_PYTHON=OFF"
}

$cmakeArgs += "-DARIX_BUILD_TESTS=$(if ($Tests) { 'ON' } else { 'OFF' })"
$cmakeArgs += "-DARIX_BUILD_BENCHMARKS=$(if ($Benchmarks) { 'ON' } else { 'OFF' })"
$cmakeArgs += "-DARIX_USE_LTO=$(if ($LTO) { 'ON' } else { 'OFF' })"
$cmakeArgs += "-DARIX_USE_ASAN=$(if ($Asan) { 'ON' } else { 'OFF' })"

Write-Host "Configuring..." -ForegroundColor Cyan
& cmake @cmakeArgs @RootDir
if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed" }

# Build
Write-Host "Building ($Config)..." -ForegroundColor Cyan
& cmake --build $BuildDir --config $Config -j
if ($LASTEXITCODE -ne 0) { throw "Build failed" }

Write-Host "Build complete ($Config)" -ForegroundColor Green
