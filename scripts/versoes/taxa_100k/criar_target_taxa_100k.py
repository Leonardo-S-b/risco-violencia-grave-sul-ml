from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.experimentos.comparar_variaveis_populacionais import (
        ARQUIVO_POPULACAO,
        adicionar_populacao_disponivel,
    )
except ModuleNotFoundError:
    from comparar_variaveis_populacionais import (  # type: ignore[no-redef]
        ARQUIVO_POPULACAO,
        adicionar_populacao_disponivel,
    )


ARQUIVO_ENTRADA = Path("data/processed/violencia_grave_sul_municipal_mensal.csv")
ARQUIVO_SAIDA = Path(
    "data/processed/versoes/taxa_100k/violencia_grave_sul_com_target_taxa_100k.csv"
)
RELATORIO = Path("reports/versoes/taxa_100k/resumo_target_taxa_100k.txt")
COLUNA_TARGET_TAXA = "alerta_taxa_violencia_grave_proximos_6m"
HORIZONTE_MESES = 6
HISTORICO_MINIMO_MESES = 12
CORTE_TAXA_100K = 25.0
MESES_COM_OCORRENCIA_MINIMOS = 2


def criar_target_taxa_100k(base: pd.DataFrame) -> pd.DataFrame:
    """Cria alerta proporcional e exclui episodios concentrados em um unico mes."""
    base = base.copy()
    base["data_referencia"] = pd.to_datetime(base["data_referencia"])
    base = base.sort_values(["uf", "municipio", "data_referencia"])
    grupo = base.groupby(["uf", "municipio"], sort=False)[
        "total_violencia_grave"
    ]
    futuros = [grupo.shift(-mes) for mes in range(1, HORIZONTE_MESES + 1)]
    base["meses_historico"] = base.groupby(
        ["uf", "municipio"], sort=False
    ).cumcount() + 1
    base["maximo_12m_ate_data_consulta"] = grupo.transform(
        lambda valores: valores.rolling(
            HISTORICO_MINIMO_MESES,
            min_periods=HISTORICO_MINIMO_MESES,
        ).max()
    )
    base["soma_violencia_grave_proximos_6m"] = sum(futuros)
    base["meses_com_violencia_grave_proximos_6m"] = sum(
        (valores > 0).astype(int) for valores in futuros
    )
    base["taxa_violencia_grave_proximos_6m_100k"] = (
        base["soma_violencia_grave_proximos_6m"]
        / base["populacao"]
        * 100_000
    )
    base["data_fim_horizonte"] = base["data_referencia"] + pd.DateOffset(
        months=HORIZONTE_MESES
    )
    base[COLUNA_TARGET_TAXA] = (
        (
            base["taxa_violencia_grave_proximos_6m_100k"]
            > CORTE_TAXA_100K
        )
        & (
            base["meses_com_violencia_grave_proximos_6m"]
            >= MESES_COM_OCORRENCIA_MINIMOS
        )
    ).astype("Int64")
    sem_contexto = (
        (base["meses_historico"] < HISTORICO_MINIMO_MESES)
        | base["soma_violencia_grave_proximos_6m"].isna()
        | base["populacao"].isna()
    )
    base.loc[sem_contexto, COLUNA_TARGET_TAXA] = pd.NA
    return base


def salvar_relatorio(base: pd.DataFrame) -> None:
    elegiveis = base.dropna(subset=[COLUNA_TARGET_TAXA]).copy()
    distribuicao = elegiveis[COLUNA_TARGET_TAXA].value_counts().sort_index()
    por_uf = (
        elegiveis.groupby(["uf", COLUNA_TARGET_TAXA])
        .size()
        .unstack(fill_value=0)
    )
    por_ano = (
        elegiveis.groupby(["ano", COLUNA_TARGET_TAXA])
        .size()
        .unstack(fill_value=0)
    )
    RELATORIO.parent.mkdir(parents=True, exist_ok=True)
    with RELATORIO.open("w", encoding="utf-8") as relatorio:
        relatorio.write("Resumo do target proporcional por 100 mil habitantes\n\n")
        relatorio.write(
            "Target 1 quando, nos proximos seis meses, a taxa acumulada supera "
            f"{CORTE_TAXA_100K:.0f} por 100 mil habitantes e existem ocorrencias "
            f"em pelo menos {MESES_COM_OCORRENCIA_MINIMOS} meses diferentes.\n"
        )
        relatorio.write(
            "A populacao usada e a ultima estimativa do IBGE disponivel antes "
            "do ano da consulta.\n"
        )
        relatorio.write(
            "Os meses futuros sao usados somente para construir a resposta; "
            "nunca entram nas variaveis do modelo.\n\n"
        )
        relatorio.write(f"Linhas totais: {len(base)}\n")
        relatorio.write(f"Linhas elegiveis: {len(elegiveis)}\n")
        relatorio.write(f"Linhas sem target: {base[COLUNA_TARGET_TAXA].isna().sum()}\n")
        relatorio.write(
            f"Percentual positivo: {elegiveis[COLUNA_TARGET_TAXA].mean() * 100:.4f}%\n\n"
        )
        relatorio.write(f"Distribuicao:\n{distribuicao.to_string()}\n\n")
        relatorio.write(f"Por UF:\n{por_uf.to_string()}\n\n")
        relatorio.write(f"Por ano:\n{por_ano.to_string()}\n")


def main() -> None:
    base = pd.read_csv(ARQUIVO_ENTRADA)
    populacao = pd.read_csv(ARQUIVO_POPULACAO, dtype={"codigo_ibge": str})
    base = adicionar_populacao_disponivel(base, populacao)
    base = criar_target_taxa_100k(base)
    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8")
    salvar_relatorio(base)
    print(f"Target gerado em: {ARQUIVO_SAIDA}")
    print(f"Relatorio gerado em: {RELATORIO}")


if __name__ == "__main__":
    main()
