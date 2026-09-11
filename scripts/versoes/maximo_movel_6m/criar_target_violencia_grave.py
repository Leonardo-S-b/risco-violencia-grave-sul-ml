from __future__ import annotations

from pathlib import Path

import pandas as pd


ARQUIVO_ENTRADA = Path("data/processed/violencia_grave_sul_municipal_mensal.csv")
ARQUIVO_SAIDA = Path("data/processed/versoes/maximo_movel_6m/violencia_grave_sul_com_target.csv")
RELATORIO = Path("reports/versoes/maximo_movel_6m/resumo_target_violencia_grave.txt")
JANELA_MAXIMO_MESES = 12
HORIZONTE_PREVISAO_MESES = 6
COLUNA_TARGET = "alerta_violencia_grave_proximos_6m"


def remover_espacos_no_fim_das_linhas(texto: str) -> str:
    return "\n".join(linha.rstrip() for linha in texto.splitlines())


def criar_target_por_maximo_historico(base: pd.DataFrame) -> pd.DataFrame:
    """Mantido para reproduzir o primeiro experimento, com maximo absoluto."""
    base = base.copy()
    base["data_referencia"] = pd.to_datetime(base["data_referencia"])
    base = base.sort_values(["uf", "municipio", "data_referencia"])

    grupo_municipio = base.groupby(["uf", "municipio"], sort=False)
    base["meses_historico"] = grupo_municipio.cumcount()
    base["maximo_historico_ate_mes_anterior"] = grupo_municipio[
        "total_violencia_grave"
    ].transform(lambda coluna: coluna.shift(1).cummax())

    base["alto_risco_violencia_grave"] = (
        base["total_violencia_grave"] > base["maximo_historico_ate_mes_anterior"]
    ).astype("Int64")

    sem_historico = base["maximo_historico_ate_mes_anterior"].isna()
    base.loc[sem_historico, "alto_risco_violencia_grave"] = pd.NA

    return base


def criar_target_por_maximo_movel(
    base: pd.DataFrame, janela_meses: int = JANELA_MAXIMO_MESES
) -> pd.DataFrame:
    """Compara cada mes com o maior total dos meses anteriores da janela."""
    base = base.copy()
    base["data_referencia"] = pd.to_datetime(base["data_referencia"])
    base = base.sort_values(["uf", "municipio", "data_referencia"])

    grupo_municipio = base.groupby(["uf", "municipio"], sort=False)
    base["meses_historico"] = grupo_municipio.cumcount()
    coluna_maximo = f"maximo_movel_{janela_meses}m_ate_mes_anterior"
    base[coluna_maximo] = grupo_municipio["total_violencia_grave"].transform(
        lambda coluna: coluna.shift(1).rolling(
            janela_meses, min_periods=janela_meses
        ).max()
    )
    base["alto_risco_violencia_grave"] = (
        base["total_violencia_grave"] > base[coluna_maximo]
    ).astype("Int64")
    base.loc[base[coluna_maximo].isna(), "alto_risco_violencia_grave"] = pd.NA
    return base


def criar_target_alerta_proximos_meses(
    base: pd.DataFrame,
    janela_historica: int = JANELA_MAXIMO_MESES,
    horizonte_meses: int = HORIZONTE_PREVISAO_MESES,
) -> pd.DataFrame:
    """Indica se algum dos proximos meses superara o maximo recente conhecido."""
    base = base.copy()
    base["data_referencia"] = pd.to_datetime(base["data_referencia"])
    base = base.sort_values(["uf", "municipio", "data_referencia"])
    grupo_municipio = base.groupby(["uf", "municipio"], sort=False)
    base["meses_historico"] = grupo_municipio.cumcount() + 1

    coluna_maximo_passado = f"maximo_{janela_historica}m_ate_data_consulta"
    coluna_maximo_futuro = f"maximo_real_proximos_{horizonte_meses}m"
    base[coluna_maximo_passado] = grupo_municipio[
        "total_violencia_grave"
    ].transform(
        lambda coluna: coluna.rolling(
            janela_historica, min_periods=janela_historica
        ).max()
    )
    base[coluna_maximo_futuro] = grupo_municipio[
        "total_violencia_grave"
    ].transform(
        lambda coluna: coluna.shift(-1)
        .rolling(horizonte_meses, min_periods=horizonte_meses)
        .max()
        .shift(-(horizonte_meses - 1))
    )
    base["data_fim_horizonte"] = base["data_referencia"] + pd.DateOffset(
        months=horizonte_meses
    )
    base[COLUNA_TARGET] = (
        base[coluna_maximo_futuro] > base[coluna_maximo_passado]
    ).astype("Int64")
    sem_contexto_completo = (
        base[coluna_maximo_passado].isna() | base[coluna_maximo_futuro].isna()
    )
    base.loc[sem_contexto_completo, COLUNA_TARGET] = pd.NA
    return base


def salvar_relatorio(base_com_target: pd.DataFrame) -> None:
    base_modelagem = base_com_target.dropna(subset=[COLUNA_TARGET])
    distribuicao_target = (
        base_modelagem[COLUNA_TARGET]
        .value_counts(dropna=False)
        .sort_index()
    )
    distribuicao_por_uf = (
        base_modelagem.groupby(["uf", COLUNA_TARGET])
        .size()
        .unstack(fill_value=0)
    )
    distribuicao_por_ano = (
        base_modelagem.groupby(["ano", COLUNA_TARGET])
        .size()
        .unstack(fill_value=0)
    )
    percentual_positivo = base_modelagem[COLUNA_TARGET].mean() * 100

    RELATORIO.parent.mkdir(parents=True, exist_ok=True)
    with RELATORIO.open("w", encoding="utf-8") as relatorio:
        relatorio.write("Resumo do target - violencia grave municipal\n")
        relatorio.write(
            f"Target: {COLUNA_TARGET} = 1 quando pelo menos um dos proximos "
            f"{HORIZONTE_PREVISAO_MESES} meses supera o maximo conhecido na "
            f"data da consulta, calculado sobre {JANELA_MAXIMO_MESES} meses.\n"
        )
        relatorio.write(
            "A data da linha representa o momento da consulta. As variaveis usam "
            "dados ate essa data; os meses futuros sao usados somente no target.\n\n"
        )
        relatorio.write(f"Linhas totais: {len(base_com_target)}\n")
        relatorio.write(f"Linhas para modelagem: {len(base_modelagem)}\n")
        relatorio.write(
            f"Linhas sem target por falta de historico anterior: "
            f"{base_com_target[COLUNA_TARGET].isna().sum()}\n\n"
        )
        relatorio.write("Distribuicao do target:\n")
        relatorio.write(remover_espacos_no_fim_das_linhas(distribuicao_target.to_string()))
        relatorio.write(f"\nPercentual positivo: {percentual_positivo:.4f}%")
        relatorio.write("\n\nDistribuicao do target por UF:\n")
        relatorio.write(remover_espacos_no_fim_das_linhas(distribuicao_por_uf.to_string()))
        relatorio.write("\n\nDistribuicao do target por ano:\n")
        relatorio.write(
            remover_espacos_no_fim_das_linhas(distribuicao_por_ano.to_string())
        )
        relatorio.write(
            "\n\nObservacao: a base ainda usa volume absoluto de vitimas, "
            "sem ajuste populacional.\n"
        )


def main() -> None:
    base = pd.read_csv(ARQUIVO_ENTRADA)
    base_com_target = criar_target_alerta_proximos_meses(base)
    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    base_com_target.to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8")
    salvar_relatorio(base_com_target)

    print(f"Arquivo gerado: {ARQUIVO_SAIDA}")
    print(f"Relatorio gerado: {RELATORIO}")


if __name__ == "__main__":
    main()
