from __future__ import annotations

from pathlib import Path

import pandas as pd


ARQUIVO_ENTRADA = Path("data/processed/violencia_grave_sul_municipal_mensal.csv")
ARQUIVO_SAIDA = Path("data/processed/violencia_grave_sul_com_target.csv")
RELATORIO = Path("reports/resumo_target_violencia_grave.txt")


def criar_target_por_media_historica(base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    base["data_referencia"] = pd.to_datetime(base["data_referencia"])
    base = base.sort_values(["uf", "municipio", "data_referencia"])

    grupo_municipio = base.groupby(["uf", "municipio"], sort=False)
    base["meses_historico"] = grupo_municipio.cumcount()
    base["media_historica_ate_mes_anterior"] = grupo_municipio[
        "total_violencia_grave"
    ].transform(lambda coluna: coluna.expanding().mean().shift(1))

    base["alto_risco_violencia_grave"] = (
        base["total_violencia_grave"] > base["media_historica_ate_mes_anterior"]
    ).astype("Int64")

    sem_historico = base["media_historica_ate_mes_anterior"].isna()
    base.loc[sem_historico, "alto_risco_violencia_grave"] = pd.NA

    return base


def salvar_relatorio(base_com_target: pd.DataFrame) -> None:
    base_modelagem = base_com_target.dropna(subset=["alto_risco_violencia_grave"])
    distribuicao_target = (
        base_modelagem["alto_risco_violencia_grave"]
        .value_counts(dropna=False)
        .sort_index()
    )
    distribuicao_por_uf = (
        base_modelagem.groupby(["uf", "alto_risco_violencia_grave"])
        .size()
        .unstack(fill_value=0)
    )

    RELATORIO.parent.mkdir(parents=True, exist_ok=True)
    with RELATORIO.open("w", encoding="utf-8") as relatorio:
        relatorio.write("Resumo do target - violencia grave municipal\n")
        relatorio.write(
            "Target: alto_risco_violencia_grave = 1 quando o total mensal "
            "fica acima da media historica do proprio municipio ate o mes anterior.\n"
        )
        relatorio.write(
            "Cuidado com vazamento: a media historica foi calculada apenas com meses anteriores.\n\n"
        )
        relatorio.write(f"Linhas totais: {len(base_com_target)}\n")
        relatorio.write(f"Linhas para modelagem: {len(base_modelagem)}\n")
        relatorio.write(
            f"Linhas sem target por falta de historico anterior: "
            f"{base_com_target['alto_risco_violencia_grave'].isna().sum()}\n\n"
        )
        relatorio.write("Distribuicao do target:\n")
        relatorio.write(distribuicao_target.to_string())
        relatorio.write("\n\nDistribuicao do target por UF:\n")
        relatorio.write(distribuicao_por_uf.to_string())
        relatorio.write(
            "\n\nObservacao: a base ainda usa volume absoluto de vitimas, "
            "sem ajuste populacional.\n"
        )


def main() -> None:
    base = pd.read_csv(ARQUIVO_ENTRADA)
    base_com_target = criar_target_por_media_historica(base)
    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    base_com_target.to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8")
    salvar_relatorio(base_com_target)

    print(f"Arquivo gerado: {ARQUIVO_SAIDA}")
    print(f"Relatorio gerado: {RELATORIO}")


if __name__ == "__main__":
    main()
