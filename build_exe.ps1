# Gera o executável Windows (.exe) da aplicação Streamlit.
# Uso (PowerShell, na pasta do projeto):
#     .\build_exe.ps1
#
# Requisitos: ter o ambiente virtual .venv criado com as dependencias
# (pip install -r requirements.txt). O script instala o PyInstaller se faltar.

$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Ambiente virtual nao encontrado. Crie com: python -m venv .venv; pip install -r requirements.txt"
}

# Garante o PyInstaller instalado no venv.
& $python -m pip show pyinstaller *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Instalando PyInstaller..."
    & $python -m pip install pyinstaller
}

Write-Host "Compilando (isso pode levar alguns minutos)..."
& $python -m PyInstaller --noconfirm --clean InvestmentAdvisorsML.spec

Write-Host ""
Write-Host "Pronto! Executavel gerado em:"
Write-Host "  dist\InvestmentAdvisorsML\InvestmentAdvisorsML.exe"
