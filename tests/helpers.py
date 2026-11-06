"""Utilitarios compartilhados pelos testes (importados como `from helpers import ...`)."""
import pandas as pd


def linha(forn, cod, desc, preco=10.0, gtin=None, ncm="87089990", texto=None):
    return {
        "codigo_interno": cod, "gtin": gtin, "gtin_status": "valido" if gtin else "ausente",
        "gtin_chave": gtin, "descricao": desc, "preco": preco, "ncm": ncm, "fornecedor": forn,
        "texto_busca": texto if texto is not None else desc.lower(),
    }


def df_de(*linhas):
    return pd.DataFrame(list(linhas))


class _Resp:
    status_code = 200

    def __init__(self, d):
        self._d = d
        self.text = ""

    def json(self):
        return self._d


class _Sessao:
    def __init__(self):
        self.pedidos = []

    def post(self, url, json, timeout):
        self.pedidos.append(list(json["input"]))
        return _Resp({"embeddings": [[float(len(t)), 1.0, 2.0] for t in json["input"]]})
