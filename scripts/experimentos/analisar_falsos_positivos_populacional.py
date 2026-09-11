from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.experimentos.comparar_variaveis_populacionais import (
        ARQUIVO_POPULACAO,
        adicionar_populacao_disponivel,
        criar_variaveis_populacionais,
    )
    from scripts.versoes.maximo_movel_6m.otimizar_gradient_boosting import CONFIGURACOES, criar_modelo
    from scripts.versoes.maximo_movel_6m.treinar_modelo_risco import (
        ANOS_VALIDACAO,
        COLUNAS_CATEGORICAS,
        COLUNA_MAXIMO,
        COLUNA_TARGET,
        LIMITE_VALIDACAO,
        criar_particoes_temporais,
        criar_variaveis_historicas,
    )
except ModuleNotFoundError:
    from comparar_variaveis_populacionais import (  # type: ignore[no-redef]
        ARQUIVO_POPULACAO,
        adicionar_populacao_disponivel,
        criar_variaveis_populacionais,
    )
    from otimizar_gradient_boosting import CONFIGURACOES, criar_modelo
    from treinar_modelo_risco import (  # type: ignore[no-redef]
        ANOS_VALIDACAO,
        COLUNAS_CATEGORICAS,
        COLUNA_MAXIMO,
        COLUNA_TARGET,
        LIMITE_VALIDACAO,
        criar_particoes_temporais,
        criar_variaveis_historicas,
    )


ARQUIVO_ENTRADA = Path("data/processed/versoes/maximo_movel_6m/violencia_grave_sul_com_target.csv")
ARQUIVO_PREVISOES = Path(
    "data/processed/experimentos/previsoes_validacao_modelo_populacional.csv"
)
RELATORIO_TEXTO = Path(
    "reports/experimentos/resumo_diagnostico_falsos_positivos_populacional.txt"
)
RELATORIO_JSON = Path(
    "reports/experimentos/diagnostico_falsos_positivos_populacional.json"
)
NOME_CONFIGURACAO = "folhas_100"
LIMIAR = 0.182510
CONFIGURACAO = next(
    configuracao
    for configuracao in CONFIGURACOES
    if configuracao["nome"] == NOME_CONFIGURACAO
)


def resumir_grupo(grupo: pd.DataFrame) -> pd.Series:
    real = grupo[COLUNA_TARGET].astype(int)
    previsto = grupo["alerta_previsto"].astype(int)
    tp = int(((real == 1) & (previsto == 1)).sum())
    fp = int(((real == 0) & (previsto == 1)).sum())
    fn = int(((real == 1) & (previsto == 0)).sum())
    tn = int(((real == 0) & (previsto == 0)).sum())
    alertas = tp + fp
    positivos = tp + fn
    negativos = tn + fp
    return pd.Series(
        {
            "observacoes": len(grupo),
            "positivos_reais": positivos,
            "taxa_positivos_reais": positivos / len(grupo),
            "alertas": alertas,
            "taxa_alertas": alertas / len(grupo),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precisao": tp / alertas if alertas else np.nan,
            "recall": tp / positivos if positivos else np.nan,
            "taxa_falso_positivo": fp / negativos if negativos else np.nan,
            "falsos_por_acerto": fp / tp if tp else np.nan,
        }
    )


def tabela_por(base: pd.DataFrame, coluna: str) -> list[dict[str, object]]:
    tabela = (
        base.groupby(coluna, observed=True, dropna=False)
        .apply(resumir_grupo, include_groups=False)
        .reset_index()
    )
    for coluna_num in tabela.select_dtypes(include="number").columns:
        tabela[coluna_num] = tabela[coluna_num].round(6)
    return tabela.to_dict(orient="records")


def main() -> None:
    base_original = pd.read_csv(ARQUIVO_ENTRADA)
    populacao = pd.read_csv(ARQUIVO_POPULACAO, dtype={"codigo_ibge": str})
    base, colunas_historicas = criar_variaveis_historicas(base_original)
    base = adicionar_populacao_disponivel(base, populacao)
    base, colunas_populacionais = criar_variaveis_populacionais(base)
    base = base[base["data_referencia"] <= LIMITE_VALIDACAO].copy()
    colunas_numericas = colunas_historicas + colunas_populacionais
    colunas_modelo = colunas_numericas + COLUNAS_CATEGORICAS

    previsoes: list[pd.DataFrame] = []
    for ano in ANOS_VALIDACAO:
        print(f"Gerando previsoes fora do treino para {ano}", flush=True)
        treino, validacao = criar_particoes_temporais(base, ano)
        modelo = criar_modelo(CONFIGURACAO, colunas_numericas)
        modelo.fit(treino[colunas_modelo], treino[COLUNA_TARGET])
        bloco = validacao[
            [
                "data_referencia",
                "ano",
                "mes",
                "uf",
                "municipio",
                "populacao",
                "total_violencia_grave",
                "contagem_violencia_12m",
                COLUNA_MAXIMO,
                COLUNA_TARGET,
            ]
        ].copy()
        bloco["probabilidade_alerta"] = modelo.predict_proba(
            validacao[colunas_modelo]
        )[:, 1]
        bloco["alerta_previsto"] = (
            bloco["probabilidade_alerta"] >= LIMIAR
        ).astype(int)
        previsoes.append(bloco)

    diagnostico = pd.concat(previsoes, ignore_index=True)
    diagnostico["tipo_resultado"] = np.select(
        [
            (diagnostico[COLUNA_TARGET] == 1)
            & (diagnostico["alerta_previsto"] == 1),
            (diagnostico[COLUNA_TARGET] == 0)
            & (diagnostico["alerta_previsto"] == 1),
            (diagnostico[COLUNA_TARGET] == 1)
            & (diagnostico["alerta_previsto"] == 0),
        ],
        ["verdadeiro_positivo", "falso_positivo", "falso_negativo"],
        default="verdadeiro_negativo",
    )
    diagnostico["faixa_populacao"] = pd.cut(
        diagnostico["populacao"],
        bins=[0, 10_000, 25_000, 50_000, 100_000, 500_000, np.inf],
        labels=[
            "01_ate_10_mil",
            "02_10_a_25_mil",
            "03_25_a_50_mil",
            "04_50_a_100_mil",
            "05_100_a_500_mil",
            "06_mais_de_500_mil",
        ],
        include_lowest=True,
        right=False,
    )
    diagnostico["faixa_maximo_historico"] = pd.cut(
        diagnostico[COLUNA_MAXIMO],
        bins=[-np.inf, 0, 1, 2, 3, 5, np.inf],
        labels=["00", "01", "02", "03", "04_a_05", "06_ou_mais"],
        right=True,
    )
    diagnostico["faixa_ocorrencias_12m"] = pd.cut(
        diagnostico["contagem_violencia_12m"],
        bins=[-np.inf, 0, 1, 3, 6, 12, np.inf],
        labels=["00", "01", "02_a_03", "04_a_06", "07_a_12", "13_ou_mais"],
        right=True,
    )
    ARQUIVO_PREVISOES.parent.mkdir(parents=True, exist_ok=True)
    diagnostico.to_csv(ARQUIVO_PREVISOES, index=False)

    dimensoes = {
        "ano": tabela_por(diagnostico, "ano"),
        "uf": tabela_por(diagnostico, "uf"),
        "faixa_populacao": tabela_por(diagnostico, "faixa_populacao"),
        "faixa_maximo_historico": tabela_por(
            diagnostico, "faixa_maximo_historico"
        ),
        "faixa_ocorrencias_12m": tabela_por(
            diagnostico, "faixa_ocorrencias_12m"
        ),
    }
    municipios = (
        diagnostico.groupby(["uf", "municipio"], observed=True)
        .apply(resumir_grupo, include_groups=False)
        .reset_index()
        .sort_values(["fp", "falsos_por_acerto"], ascending=[False, False])
    )
    municipios_mais_fp = municipios.head(20).round(6).to_dict(orient="records")
    saida = {
        "origem": "Previsoes temporais fora do treino de 2020 a 2023.",
        "configuracao": CONFIGURACAO,
        "limiar": LIMIAR,
        "teste_final_usado": False,
        "dimensoes": dimensoes,
        "municipios_com_mais_falsos_positivos": municipios_mais_fp,
    }
    RELATORIO_JSON.parent.mkdir(parents=True, exist_ok=True)
    RELATORIO_JSON.write_text(
        json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with RELATORIO_TEXTO.open("w", encoding="utf-8") as relatorio:
        relatorio.write("Diagnostico dos falsos positivos - modelo populacional\n\n")
        relatorio.write("Previsoes fora do treino: 2020 a 2023.\n")
        relatorio.write("Teste final nao utilizado.\n")
        relatorio.write(f"Limiar unico: {LIMIAR:.6f}.\n\n")
        for nome, registros in dimensoes.items():
            relatorio.write(f"Por {nome}:\n")
            for registro in registros:
                chave = registro[nome]
                relatorio.write(
                    f"- {chave}: n={int(registro['observacoes'])}, "
                    f"taxa_real={registro['taxa_positivos_reais']:.4f}, "
                    f"precisao={registro['precisao']:.4f}, "
                    f"recall={registro['recall']:.4f}, "
                    f"FP={int(registro['fp'])}, "
                    f"taxa_FP={registro['taxa_falso_positivo']:.4f}, "
                    f"falsos_por_acerto={registro['falsos_por_acerto']:.3f}\n"
                )
            relatorio.write("\n")
        relatorio.write("20 municipios com mais falsos positivos:\n")
        for registro in municipios_mais_fp:
            relatorio.write(
                f"- {registro['municipio']}/{registro['uf']}: "
                f"FP={int(registro['fp'])}, TP={int(registro['tp'])}, "
                f"precisao={registro['precisao']:.4f}, "
                f"recall={registro['recall']:.4f}\n"
            )

    print(f"Diagnostico salvo em: {RELATORIO_TEXTO} e {RELATORIO_JSON}")
    print(f"Previsoes auditaveis salvas em: {ARQUIVO_PREVISOES}")


if __name__ == "__main__":
    main()
