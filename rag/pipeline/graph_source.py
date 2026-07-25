"""
Fuente Microsoft Graph (OneDrive / SharePoint) para la ingesta RAG.
Ruta: Backend-RAGs/pipeline/graph_source.py

Base del connector: autentica vía DEVICE CODE (cuenta personal o work) y lista /
descarga PDFs por Microsoft Graph. La MISMA API (DriveItem) sirve para:
  - OneDrive personal  → /me/drive           (test gratis, ahora)
  - SharePoint cliente → /drives/{drive_id}   (después; solo cambia el drive)

Requiere una App registrada en Entra ID (client_id) con permiso DELEGADO
`Files.Read` y "public client flows" habilitado (para device code).
"""

from __future__ import annotations

from typing import Any

import requests

GRAPH = "https://graph.microsoft.com/v1.0"
# Cuentas personales (outlook.com/hotmail). Para work/school: usar el tenant o 'organizations'.
AUTHORITY_CONSUMERS = "https://login.microsoftonline.com/consumers"
DEFAULT_SCOPES = ("Files.Read",)


def device_login(
    client_id: str,
    authority: str = AUTHORITY_CONSUMERS,
    scopes: tuple[str, ...] = DEFAULT_SCOPES,
) -> str:
    """Login interactivo por DEVICE CODE (no necesita servidor de redirect).
    Imprime la URL + código; bloquea hasta que completes el login en el navegador.
    Devuelve el access_token."""
    import msal

    app = msal.PublicClientApplication(client_id, authority=authority)
    flow = app.initiate_device_flow(scopes=list(scopes))
    if "user_code" not in flow:
        raise RuntimeError(f"No se pudo iniciar el device flow: {flow}")
    print("\n" + flow["message"] + "\n", flush=True)  # "Ve a https://microsoft.com/devicelogin e ingresa CODE"
    result = app.acquire_token_by_device_flow(flow)  # bloquea hasta login
    if "access_token" not in result:
        raise RuntimeError(f"Auth falló: {result.get('error_description', result)}")
    return result["access_token"]


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _drive_base(drive: str) -> str:
    return "/me/drive" if drive == "me" else f"/drives/{drive}"


def list_pdfs(token: str, drive: str = "me") -> list[dict[str, Any]]:
    """Lista los PDFs del drive (OneDrive del usuario por defecto). Pagina con
    @odata.nextLink. Devuelve [{id, name, size, path}]."""
    url: str | None = f"{GRAPH}{_drive_base(drive)}/root/search(q='.pdf')"
    out: list[dict[str, Any]] = []
    while url:
        r = requests.get(url, headers=_headers(token), timeout=30)
        r.raise_for_status()
        data = r.json()
        for it in data.get("value", []):
            if "file" in it and it.get("name", "").lower().endswith(".pdf"):
                out.append(
                    {
                        "id": it["id"],
                        "name": it["name"],
                        "size": it.get("size", 0),
                        "path": it.get("parentReference", {}).get("path", ""),
                    }
                )
        url = data.get("@odata.nextLink")
    return out


def download(token: str, item_id: str, drive: str = "me") -> bytes:
    """Descarga el contenido de un item (sigue el redirect a la URL de descarga)."""
    r = requests.get(
        f"{GRAPH}{_drive_base(drive)}/items/{item_id}/content",
        headers=_headers(token),
        timeout=120,
    )
    r.raise_for_status()
    return r.content
