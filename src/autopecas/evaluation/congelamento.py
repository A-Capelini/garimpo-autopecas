"""Verifica que a regra de ordenacao por veiculo nao mudou desde uma tag (regra do holdout: nao ajustar a
regra antes de medir). Compara o ARQUIVO NO DISCO (inclui alteracoes ainda nao commitadas) com a tag."""
from __future__ import annotations

import subprocess
from pathlib import Path

TAG_REGRA = "etapa-4"
ARQUIVOS_REGRA = ("src/autopecas/retrieval/veiculo.py", "src/autopecas/veiculos.py")


def regra_congelada(
    raiz: Path | str, tag: str = TAG_REGRA, arquivos: tuple[str, ...] = ARQUIVOS_REGRA
) -> tuple[bool, str]:
    """(True, resumo) se os arquivos sao identicos aos da tag; (False, motivo) se diferem ou se nao deu para verificar."""
    try:
        r = subprocess.run(
            ["git", "diff", "--name-only", tag, "--", *arquivos],
            cwd=raiz, capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"não consegui executar o git: {e}"
    if r.returncode != 0:
        return False, f"git diff falhou (a tag '{tag}' existe neste clone?): {r.stderr.strip() or r.stdout.strip()}"
    diferentes = [l for l in r.stdout.splitlines() if l.strip()]
    if diferentes:
        return False, "diferem da tag " + tag + ": " + ", ".join(diferentes)
    return True, f"{', '.join(arquivos)} idênticos à tag {tag}"


def commit_atual(raiz: Path | str) -> str:
    """Hash curto do HEAD (+ '+alterações' se ha mudanca nao commitada em arquivos rastreados); '?' se nao deu."""
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=raiz, capture_output=True, text=True, timeout=30)
        s = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=raiz,
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return "?"
    if h.returncode != 0:
        return "?"
    return h.stdout.strip() + ("+alterações" if s.stdout.strip() else "")
