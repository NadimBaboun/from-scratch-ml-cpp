param(
    [string]$Solution = "MachineLearningLab.sln",
    [string]$Configuration = "Debug",
    [string]$Platform = "x64"
)

$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$solutionPath = Join-Path $workspace $Solution

if (-not (Test-Path $solutionPath)) {
    Write-Error "Solution not found: $solutionPath"
}

$msbuildCandidates = @(
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\MSBuild.exe",
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\MSBuild\Current\Bin\MSBuild.exe",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\MSBuild.exe",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\Enterprise\MSBuild\Current\Bin\MSBuild.exe",
    "C:\Windows\Microsoft.NET\Framework64\v4.0.30319\MSBuild.exe"
)

$msbuild = $msbuildCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $msbuild) {
    Write-Error "MSBuild not found. Install Visual Studio 2022 Build Tools with C++ workload (v143)."
}

$vcTargetsCandidates = @(
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Microsoft\VC\v170\",
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\MSBuild\Microsoft\VC\v170\",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\Professional\MSBuild\Microsoft\VC\v170\",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\Enterprise\MSBuild\Microsoft\VC\v170\"
)

$vcTargetsPath = $vcTargetsCandidates | Where-Object { Test-Path (Join-Path $_ 'Microsoft.Cpp.Default.props') } | Select-Object -First 1
if (-not $vcTargetsPath) {
    Write-Error @"
Visual C++ targets are missing (Microsoft.Cpp.Default.props not found).
Install Visual Studio 2022 Build Tools with:
- Desktop development with C++ (v143)
- C++/CLI support
- Windows 10/11 SDK
- .NET Framework 4.7.2 targeting pack
"@
}

$vcTargetsPath = $vcTargetsPath.TrimEnd('\\')
$vcTargetsPathForMsbuild = $vcTargetsPath + '\\'

Write-Host "Using MSBuild: $msbuild"
Write-Host "Using VCTargetsPath: $vcTargetsPath"

& $msbuild $solutionPath /m /t:Build /p:Configuration=$Configuration /p:Platform=$Platform "/p:VCTargetsPath=$vcTargetsPathForMsbuild" /v:m
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Build succeeded: $Solution ($Configuration|$Platform)"
