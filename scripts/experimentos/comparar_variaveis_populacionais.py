from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.versoes.maximo_movel_6m.analisar_limiares_modelo import (
        complementar_metricas,
        escolher_limiar_recall_minimo,
    )
    from scripts.dados.baixar_populacao_ibge import normalizar_nome
    from scripts.versoes.maximo_movel_6m.otimizar_gradient_boosting import CONFIGURACOES, criar_modelo
    from scripts.versoes.maximo_movel_6m.treinar_modelo_risco import (
        ANOS_VALIDACAO,
        COLUNAS_CATEGORICAS,
        COLUNA_TARGET,
        LIMITE_VALIDACAO,
        calcular_metricas,
        criar_particoes_temporais,
        criar_variaveis_historicas,
        escolher_limiar,
    )
except ModuleNotFoundError:
    from analisar_limiares_modelo import (  # type: ignore[no-redef]
        complementar_metricas,
        escolher_limiar_recall_minimo,
    )
    from baixar_populacao_ibge import normalizar_nome
    from otimizar_gradient_boosting import CONFIGURACOES, criar_modelo
    from treinar_modelo_risco import (  # type: ignore[no-redef]
        ANOS_VALIDACAO,
        COLUNAS_CATEGORICAS,
        COLUNA_TARGET,
        LIMITE_VALIDACAO,
        calcular_metricas,
        criar_particoes_temporais,
        criar_variaveis_historicas,
        escolher_limiar,
    )


ARQUIVO_ENTRADA = Path("data/processed/versoes/maximo_movel_6m/violencia_grave_sul_com_target.csv")
ARQUIVO_POPULACAO = Path("data/processed/referencias/populacao_municipal_ibge.csv")
RELATORIO_TEXTO = Path("reports/experimentos/resumo_comparacao_variaveis_populacionais.txt")
RELATORIO_JSON = Path("reports/experimentos/metricas_comparacao_variaveis_populacionais.json")
NOME_CONFIGURACAO = "conservador_15"
PESOS_SUAVIZACAO = (5_000, 20_000, 50_000)
CONFIGURACAO = next(
    configuracao
    for configuracao in CONFIGURACOES
    if configuracao["nome"] == NOME_CONFIGURACAO
)


def adicionar_populacao_disponivel(
    base: pd.DataFrame, populacao: pd.DataFrame
) -> pd.DataFrame:
    """Associa a estimativa mais recente publicada antes do ano da consulta."""
    base = base.copy()
    populacao = populacao.copy()
    base["municipio_normalizado"] = base["municipio"].map(normalizar_nome)
    maximo_ano_populacao = int(populacao["ano_populacao"].max())
    base["ano_populacao"] = np.minimum(
        base["ano"].astype(int) - 1, maximo_ano_populacao
    )
    colunas_populacao = [
        "uf",
        "municipio_normalizado",
        "ano_populacao",
        "codigo_ibge",
        "populacao",
    ]
    resultado = base.merge(
        populacao[colunas_populacao],
        on=["uf", "municipio_normalizado", "ano_populacao"],
        how="left",
        validate="many_to_one",
    )
    return resultado


def criar_variaveis_populacionais(
    base: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    base = base.copy()
    if base["populacao"].isna().any():
        faltantes = base.loc[
            base["populacao"].isna(), ["uf", "municipio"]
        ].drop_duplicates()
        exemplos = faltantes.head(10).to_dict(orient="records")
        raise ValueError(
            f"Existem {len(faltantes)} municipios sem populacao: {exemplos}"
        )
    if (base["populacao"] <= 0).any():
        raise ValueError("A populacao precisa ser positiva.")

    base["log_populacao"] = np.log1p(base["populacao"])
    base["taxa_total_atual_100k"] = (
        base["total_violencia_grave"] / base["populacao"] * 100_000
    )
    base["taxa_total_lag_1m_100k"] = (
        base["total_lag_1m"] / base["populacao"] * 100_000
    )
    colunas = [
        "log_populacao",
        "taxa_total_atual_100k",
        "taxa_total_lag_1m_100k",
    ]

    for janela in (3, 6, 12):
        coluna_contagem = f"contagem_violencia_{janela}m"
        coluna_taxa = f"taxa_violencia_{janela}m_100k"
        base[coluna_contagem] = base[f"media_movel_{janela}m"] * janela
        base[coluna_taxa] = (
            base[coluna_contagem] / base["populacao"] * 100_000
        )
        colunas.append(coluna_taxa)

    agrupamento_uf_data = base.groupby(["uf", "data_referencia"], sort=False)
    base["contagem_12m_uf"] = agrupamento_uf_data[
        "contagem_violencia_12m"
    ].transform("sum")
    base["populacao_uf"] = agrupamento_uf_data["populacao"].transform("sum")
    base["taxa_violencia_12m_uf_100k"] = (
        base["contagem_12m_uf"] / base["populacao_uf"] * 100_000
    )
    colunas.append("taxa_violencia_12m_uf_100k")

    taxa_uf_por_pessoa = base["taxa_violencia_12m_uf_100k"] / 100_000
    for peso in PESOS_SUAVIZACAO:
        coluna = f"taxa_bayes_12m_peso_{peso}_100k"
        base[coluna] = (
            (
                base["contagem_violencia_12m"]
                + taxa_uf_por_pessoa * peso
            )
            / (base["populacao"] + peso)
            * 100_000
        )
        colunas.append(coluna)

    base["razao_taxa_municipio_uf"] = np.divide(
        base["taxa_bayes_12m_peso_20000_100k"],
        base["taxa_violencia_12m_uf_100k"],
        out=np.zeros(len(base), dtype=float),
        where=base["taxa_violencia_12m_uf_100k"] > 0,
    )
    colunas.append("razao_taxa_municipio_uf")
    return base, colunas


def main() -> None:
    base_original = pd.read_csv(ARQUIVO_ENTRADA)
    populacao = pd.read_csv(ARQUIVO_POPULACAO, dtype={"codigo_ibge": str})
    base, colunas_numericas_originais = criar_variaveis_historicas(base_original)
    base = adicionar_populacao_disponivel(base, populacao)
    base, colunas_populacionais = criar_variaveis_populacionais(base)
    base = base[base["data_referencia"] <= LIMITE_VALIDACAO]

    colunas_modelo_original = colunas_numericas_originais + COLUNAS_CATEGORICAS
    colunas_numericas_populacao = (
        colunas_numericas_originais + colunas_populacionais
    )
    colunas_modelo_populacao = (
        colunas_numericas_populacao + COLUNAS_CATEGORICAS
    )
    respostas: list[np.ndarray] = []
    probabilidades_por_modelo: dict[str, list[np.ndarray]] = {
        "modelo_original": [],
        "modelo_com_populacao": [],
    }

    for ano_validacao in ANOS_VALIDACAO:
        print(f"Comparacao temporal de {ano_validacao}", flush=True)
        treino, validacao = criar_particoes_temporais(base, ano_validacao)
        respostas.append(validacao[COLUNA_TARGET].to_numpy())

        modelo_original = criar_modelo(CONFIGURACAO, colunas_numericas_originais)
        modelo_original.fit(treino[colunas_modelo_original], treino[COLUNA_TARGET])
        probabilidades_por_modelo["modelo_original"].append(
            modelo_original.predict_proba(validacao[colunas_modelo_original])[:, 1]
        )

        modelo_populacao = criar_modelo(CONFIGURACAO, colunas_numericas_populacao)
        modelo_populacao.fit(
            treino[colunas_modelo_populacao], treino[COLUNA_TARGET]
        )
        probabilidades_por_modelo["modelo_com_populacao"].append(
            modelo_populacao.predict_proba(validacao[colunas_modelo_populacao])[:, 1]
        )

    y_real = pd.Series(np.concatenate(respostas))
    resultados: dict[str, dict[str, object]] = {}
    for nome, blocos in probabilidades_por_modelo.items():
        probabilidades = np.concatenate(blocos)
        limiar_f2 = escolher_limiar(y_real, probabilidades)
        limiar_recall_70 = escolher_limiar_recall_minimo(
            y_real, probabilidades, 0.70
        )
        resultados[nome] = {
            "limiar_otimizado_f2": complementar_metricas(
                calcular_metricas(y_real, probabilidades, limiar_f2)
            ),
            "limiar_com_recall_minimo_70": complementar_metricas(
                calcular_metricas(y_real, probabilidades, limiar_recall_70)
            ),
        }

    saida = {
        "etapa": "Variaveis populacionais e taxas suavizadas no classificador.",
        "fonte_populacao": "IBGE SIDRA, tabela 6579, variavel 9324.",
        "regra_temporal_populacao": (
            "Ultima estimativa disponivel anterior ao ano da consulta."
        ),
        "target_alterado": False,
        "tipo_modelo": "classificacao_binaria",
        "teste_final_usado": False,
        "colunas_populacionais": colunas_populacionais,
        "resultados": resultados,
    }
    RELATORIO_TEXTO.parent.mkdir(parents=True, exist_ok=True)
    RELATORIO_JSON.write_text(
        json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with RELATORIO_TEXTO.open("w", encoding="utf-8") as relatorio:
        relatorio.write("Comparacao de variaveis populacionais\n\n")
        relatorio.write("Modelo: classificacao binaria. Target inalterado.\n")
        relatorio.write("Fonte: IBGE SIDRA, tabela 6579, variavel 9324.\n")
        relatorio.write(
            "Regra temporal: populacao publicada antes do ano da consulta.\n"
        )
        relatorio.write("O teste final nao foi usado neste experimento.\n\n")
        for nome, cenarios in resultados.items():
            relatorio.write(f"{nome}:\n")
            for cenario, metricas in cenarios.items():
                matriz = metricas["matriz_confusao"]
                relatorio.write(
                    f"- {cenario}: AP={metricas['average_precision']}, "
                    f"limiar={metricas['limiar']}, "
                    f"precisao={metricas['precision']}, "
                    f"recall={metricas['recall']}, F2={metricas['f2']}, "
                    f"FP={matriz['fp']}, FN={matriz['fn']}, "
                    "falsos_por_acerto="
                    f"{metricas['falsos_positivos_por_verdadeiro_positivo']}\n"
                )
            relatorio.write("\n")

    print(f"Comparacao salva em: {RELATORIO_TEXTO} e {RELATORIO_JSON}")


if __name__ == "__main__":
    main()
