"""Ponto de entrada do executável (.exe).

Sobe o servidor do Streamlit para o `streamlit_app.py` empacotado e abre o
navegador padrão do usuário. Serve tanto ao rodar diretamente (`python
run_app.py`) quanto empacotado com o PyInstaller.
"""

import multiprocessing
import os
import sys
import threading
import time
import webbrowser

PORT = 8501
URL = f"http://localhost:{PORT}"


def resource_path(relative: str) -> str:
    """Resolve um caminho tanto em desenvolvimento quanto dentro do bundle."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative)


def open_browser_when_ready() -> None:
    """Abre o navegador assim que o servidor estiver aceitando conexões."""
    import socket

    deadline = time.time() + 60
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            if sock.connect_ex(("127.0.0.1", PORT)) == 0:
                break
        time.sleep(0.5)
    webbrowser.open(URL)


def main() -> None:
    app_path = resource_path("streamlit_app.py")

    # No bundle do PyInstaller o Streamlit interpreta que está em "development
    # mode" (o default vira True quando não detecta instalação normal), o que
    # conflita com server.port. Forçamos via env antes de importar o Streamlit.
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"

    sys.argv = [
        "streamlit",
        "run",
        app_path,
        f"--server.port={PORT}",
        "--server.headless=true",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
    ]

    threading.Thread(target=open_browser_when_ready, daemon=True).start()

    from streamlit.web import cli as stcli

    sys.exit(stcli.main())


if __name__ == "__main__":
    # Necessário para PyInstaller no Windows (evita re-spawn infinito de processos).
    multiprocessing.freeze_support()
    main()
