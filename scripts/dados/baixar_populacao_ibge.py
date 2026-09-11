from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


ANOS = tuple(range(2013, 2022))
UFS_SUL = {"PR", "RS", "SC"}
URL_MODELO = (
    "https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/9324/"
    "p/{ano}?formato=json"
)
ARQUIVO_SAIDA = Path("data/processed/referencias/populacao_municipal_ibge.csv")
RELATORIO_SAIDA = Path("reports/dados/resumo_populacao_ibge.txt")


def normalizar_nome(valor: str) -> str:
    sem_acentos = "".join(
        caractere
        for caractere in unicodedata.normalize("NFKD", valor)
        if not unicodedata.combining(caractere)
    )
    return " ".join(sem_acentos.upper().replace("'", " ").split())


def baixar_ano(ano: int) -> list[dict[str, object]]:
    with urlopen(URL_MODELO.format(ano=ano), timeout=120) as resposta:
        dados = json.loads(resposta.read().decode("utf-8"))

    registros: list[dict[str, object]] = []
    for item in dados[1:]:
        nome_com_uf = str(item["D1N"])
        if " - " not in nome_com_uf:
            continue
        municipio, uf = nome_com_uf.rsplit(" - ", maxsplit=1)
        if uf not in UFS_SUL or str(item["V"]) == "...":
            continue
        registros.append(
            {
                "codigo_ibge": str(item["D1C"]),
                "uf": uf,
                "municipio_ibge": municipio,
                "municipio_normalizado": normalizar_nome(municipio),
                "ano_populacao": ano,
                "populacao": int(item["V"]),
            }
        )
    return registros


def main() -> None:
    registros: list[dict[str, object]] = []
    for ano in ANOS:
        print(f"Baixando populacao municipal de {ano}", flush=True)
        registros.extend(baixar_ano(ano))

    populacao = pd.DataFrame(registros).sort_values(
        ["uf", "municipio_normalizado", "ano_populacao"]
    )
    duplicados = populacao.duplicated(
        ["uf", "municipio_normalizado", "ano_populacao"]
    )
    if duplicados.any():
        raise ValueError("A API retornou municipios duplicados depois da normalizacao.")

    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    populacao.to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8")
    RELATORIO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    with RELATORIO_SAIDA.open("w", encoding="utf-8") as relatorio:
        relatorio.write("Populacao municipal - IBGE SIDRA, tabela 6579\n\n")
        relatorio.write("Variavel 9324: populacao residente estimada.\n")
        relatorio.write(f"Anos: {min(ANOS)} a {max(ANOS)}.\n")
        relatorio.write(f"Registros da Regiao Sul: {len(populacao)}.\n")
        relatorio.write(f"Municipios distintos: {populacao['codigo_ibge'].nunique()}.\n")
        relatorio.write(
            "Fonte: https://apisidra.ibge.gov.br/values/t/6579/\n"
        )

    print(f"Dados salvos em: {ARQUIVO_SAIDA}")
    print(f"Resumo salvo em: {RELATORIO_SAIDA}")


if __name__ == "__main__":
    main()
