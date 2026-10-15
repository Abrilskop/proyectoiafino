# ===========================================================================
# sync.ps1 - Copia el codigo y el modelo del proyecto al folder del Space.
#
# Ejecutar ANTES de hacer `git push` al Hugging Face Space, para asegurar que
# la copia desplegada esta actualizada. Las carpetas copiadas estan ignoradas
# por el .gitignore del proyecto (solo viven en el repo del Space).
#
# Uso:
#   .\deployment\hf_space\sync.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"

$Space = $PSScriptRoot
$Root = (Resolve-Path (Join-Path $Space "..\..")).Path

Write-Host "Proyecto : $Root" -ForegroundColor Cyan
Write-Host "Space    : $Space" -ForegroundColor Cyan

function Copy-Dir($Name) {
    $src = Join-Path $Root $Name
    $dst = Join-Path $Space $Name
    if (-not (Test-Path $src)) { throw "No existe: $src" }
    if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
    Copy-Item -Recurse -Force $src $dst
    # Limpia caches que no deben desplegarse
    Get-ChildItem -Recurse -Force -Directory $dst -Filter __pycache__ |
        Remove-Item -Recurse -Force
    Write-Host "  + $Name" -ForegroundColor Green
}

Copy-Dir "src"
Copy-Dir "monitoring"
Copy-Dir "configs"

# deployment/ sin el propio folder del Space
$depDst = Join-Path $Space "deployment"
if (Test-Path $depDst) { Remove-Item -Recurse -Force $depDst }
New-Item -ItemType Directory -Force -Path $depDst | Out-Null
foreach ($f in "app.py", "gradio_app.py", "__init__.py") {
    Copy-Item -Force (Join-Path $Root "deployment\$f") $depDst
}
Write-Host "  + deployment/{app.py,gradio_app.py,__init__.py}" -ForegroundColor Green

# Modelo entrenado
$modelSrc = Join-Path $Root "models\best.pt"
$modelDst = Join-Path $Space "models"
New-Item -ItemType Directory -Force -Path $modelDst | Out-Null
if (Test-Path $modelSrc) {
    Copy-Item -Force $modelSrc $modelDst
    Write-Host "  + models/best.pt" -ForegroundColor Green
} else {
    Write-Host "  ! No se encontro models/best.pt (entrena antes de desplegar)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Listo. El folder del Space esta actualizado." -ForegroundColor Green
