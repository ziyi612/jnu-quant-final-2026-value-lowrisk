param(
    [string]$PythonPath = 'python',
    [string]$ConfigPath = (Join-Path $PSScriptRoot 'code/config.json'),
    [string]$DataDir,
    [string]$BenchmarkPath,
    [string]$OutputDir = (Join-Path $PSScriptRoot 'results'),
    [switch]$WithFigures
)

$ErrorActionPreference = 'Stop'
$codeDir = Join-Path $PSScriptRoot 'code'
$researchArgs = @('--config', $ConfigPath, '--output-dir', $OutputDir)
if ($DataDir) { $researchArgs += @('--data-dir', $DataDir) }
if ($BenchmarkPath) { $researchArgs += @('--benchmark-path', $BenchmarkPath) }

& $PythonPath -B (Join-Path $codeDir 'research.py') @researchArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonPath -B (Join-Path $codeDir 'validate_results.py') --results $OutputDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($WithFigures) {
    $figureDir = Join-Path $OutputDir 'figures'
    & $PythonPath -B (Join-Path $codeDir 'figures.py') --results $OutputDir --output $figureDir
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
