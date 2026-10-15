# ===========================================================================
# run.ps1 - Levanta el servicio SVHN con un solo comando.
#
# Uso rapido:
#   .\run.ps1                 -> inicia la API en http://127.0.0.1:8000
#   .\run.ps1 -Port 9000      -> cambia el puerto
#   .\run.ps1 -Full           -> ejecuta TODO el pipeline y luego inicia la API
#
# Si da error de permisos, ejecuta una vez:
#   Set-ExecutionPolicy -Scope Process RemoteSigned
# ===========================================================================
param(
    [int]$Port = 8000,
    [string]$Host_ = "127.0.0.1",
    [switch]$Full
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

$env:PYTHONUTF8 = "1"
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Host "No se encontro el entorno virtual; usando 'python' del sistema." -ForegroundColor Yellow
    $Python = "python"
}

function Run-Step($Title, $Args) {
    Write-Host ""
    Write-Host "==> $Title" -ForegroundColor Cyan
    & $Python @Args
    if ($LASTEXITCODE -ne 0) { throw "Fallo el paso: $Title" }
}

if ($Full) {
    Write-Host "Ejecutando pipeline completo (O.E.1 .. O.E.6)..." -ForegroundColor Green
    Run-Step "O.E.1 Negocio"      @("-m", "src.business")
    Run-Step "O.E.2 Descarga"     @("-m", "src.data.download")
    Run-Step "O.E.2 EDA"          @("-m", "src.data.eda")
    Run-Step "O.E.3 Preparacion"  @("-m", "src.data.preprocess")
    Run-Step "O.E.4 Modelado"     @("-m", "src.train")
    Run-Step "O.E.5 Evaluacion"   @("-m", "src.evaluate")
    Run-Step "O.E.6 Export ONNX"  @("-m", "src.predict", "--export")
    Run-Step "Tests"              @("-m", "pytest", "-q")
}

Write-Host ""
Write-Host "Iniciando API SVHN en http://${Host_}:$Port" -ForegroundColor Green
Write-Host "Documentacion interactiva: http://${Host_}:$Port/docs" -ForegroundColor Green
Write-Host "Detener con Ctrl+C" -ForegroundColor DarkGray
Write-Host ""

& $Python -m uvicorn deployment.app:app --host $Host_ --port $Port
