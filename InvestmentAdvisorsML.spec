# -*- mode: python ; coding: utf-8 -*-
# Spec do PyInstaller para empacotar o app Streamlit como executável Windows.
# Build:  pyinstaller InvestmentAdvisorsML.spec  (com o .venv ativado)
# Saida:  dist/InvestmentAdvisorsML/InvestmentAdvisorsML.exe

from PyInstaller.utils.hooks import collect_all, copy_metadata

datas = []
binaries = []
hiddenimports = []

# Coleta código, dados (arquivos estáticos) e dependências dos pacotes pesados.
for pkg in ("streamlit", "yfinance", "sklearn", "scipy", "talib", "plotly", "pandas"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Metadados (dist-info) que o Streamlit e outras libs consultam em runtime.
for pkg in ("streamlit", "yfinance", "plotly", "scikit-learn", "numpy", "pandas"):
    datas += copy_metadata(pkg)

# Arquivos da própria aplicação (precisam estar no bundle).
datas += [
    ("streamlit_app.py", "."),
    ("parameters.json", "."),
    ("dataset", "dataset"),
    ("assets", "assets"),
]

a = Analysis(
    ["run_app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="InvestmentAdvisorsML",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="InvestmentAdvisorsML",
)
