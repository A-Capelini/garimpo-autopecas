"""Congelamento da regra de veiculo (src/autopecas/evaluation/congelamento.py), com um repositorio git temporario."""
import shutil
import subprocess

import pytest

from autopecas.evaluation.congelamento import commit_atual, regra_congelada

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git indisponivel")
GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false"]


def repo(tmp_path):
    (tmp_path / "regra.py").write_text("A = 1\n")
    (tmp_path / "outro.py").write_text("B = 1\n")
    for cmd in (["init", "-q"], ["add", "."], ["commit", "-q", "-m", "x"], ["tag", "etapa-4"]):
        subprocess.run(GIT + cmd, cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


def test_identico_a_tag(tmp_path):
    ok, msg = regra_congelada(repo(tmp_path), "etapa-4", ("regra.py",))
    assert ok and "idênticos" in msg


def test_alteracao_nao_commitada_e_detectada_mas_arquivo_fora_da_regra_nao_conta(tmp_path):
    r = repo(tmp_path)
    (r / "outro.py").write_text("B = 2\n")
    assert regra_congelada(r, "etapa-4", ("regra.py",))[0] is True
    (r / "regra.py").write_text("A = 2\n")
    ok, msg = regra_congelada(r, "etapa-4", ("regra.py",))
    assert not ok and "regra.py" in msg


def test_alteracao_commitada_depois_da_tag_tambem_e_detectada(tmp_path):
    r = repo(tmp_path)
    (r / "regra.py").write_text("A = 3\n")
    subprocess.run(GIT + ["commit", "-q", "-am", "muda"], cwd=r, check=True, capture_output=True)
    assert not regra_congelada(r, "etapa-4", ("regra.py",))[0]


def test_tag_inexistente_recusa_em_vez_de_aprovar(tmp_path):
    ok, msg = regra_congelada(repo(tmp_path), "nao-existe", ("regra.py",))
    assert not ok and "tag" in msg


def test_commit_atual_marca_alteracoes(tmp_path):
    r = repo(tmp_path)
    limpo = commit_atual(r)
    assert limpo != "?" and "+alterações" not in limpo
    (r / "regra.py").write_text("A = 9\n")
    assert commit_atual(r) == limpo + "+alterações"
