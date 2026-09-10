from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    from scripts.treinar_modelo_risco import (
        ANOS_VALIDACAO,
        COLUNAS_CATEGORICAS,
        COLUNA_TARGET,
        LIMITE_VALIDACAO,
        SEMENTE,
        calcular_metricas,
        criar_particoes_temporais,
        criar_preprocessador,
        criar_variaveis_historicas,
        escolher_limiar,
        resumir_validacoes,
    )
except ModuleNotFoundError:
    from treinar_modelo_risco import (  # type: ignore[no-redef]
        ANOS_VALIDACAO,
        COLUNAS_CATEGORICAS,
        COLUNA_TARGET,
        LIMITE_VALIDACAO,
        SEMENTE,
        calcular_metricas,
        criar_particoes_temporais,
        criar_preprocessador,
        criar_variaveis_historicas,
        escolher_limiar,
        resumir_validacoes,
    )


ARQUIVO_ENTRADA = Path("data/processed/violencia_grave_sul_com_target.csv")
RELATORIO_TEXTO = Path("reports/resumo_comparacao_balanceamento.txt")
RELATORIO_JSON = Path("reports/metricas_comparacao_balanceamento.json")

ALGORITMOS = ("regressao_logistica", "gradient_boosting", "floresta_aleatoria")
ESTRATEGIAS = (
    "original",
    "peso_classes",
    "undersampling_1_3",
    "undersampling_1_2",
)


def aplicar_estrategia_treino(
    treino: pd.DataFrame, estrategia: str, semente: int
) -> pd.DataFrame:
    """Reamostra apenas o treino; validacao e teste nunca passam por esta funcao."""
    if not estrategia.startswith("undersampling_"):
        return treino

    proporcao_negativos = int(estrategia.rsplit("_", maxsplit=1)[-1])
    positivos = treino[treino[COLUNA_TARGET] == 1]
    negativos = treino[treino[COLUNA_TARGET] == 0]
    quantidade_negativos = min(
        len(negativos), len(positivos) * proporcao_negativos
    )
    negativos_amostrados = negativos.sample(
        n=quantidade_negativos, random_state=semente
    )
    return pd.concat([positivos, negativos_amostrados]).sample(
        frac=1, random_state=semente
    )


def criar_modelo(
    algoritmo: str, estrategia: str, colunas_numericas: list[str]
) -> Pipeline:
    usar_peso = estrategia == "peso_classes"

    if algoritmo == "regressao_logistica":
        estimador = LogisticRegression(
            class_weight="balanced" if usar_peso else None,
            max_iter=2_000,
            random_state=SEMENTE,
        )
    elif algoritmo == "gradient_boosting":
        estimador = HistGradientBoostingClassifier(
            class_weight="balanced" if usar_peso else None,
            learning_rate=0.08,
            max_iter=150,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=SEMENTE,
        )
    elif algoritmo == "floresta_aleatoria":
        estimador = RandomForestClassifier(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=5,
            class_weight="balanced_subsample" if usar_peso else None,
            n_jobs=-1,
            random_state=SEMENTE,
        )
    else:
        raise ValueError(f"Algoritmo desconhecido: {algoritmo}")

    return Pipeline(
        [
            ("preprocessamento", criar_preprocessador(colunas_numericas)),
            ("modelo", estimador),
        ]
    )


def main() -> None:
    base_original = pd.read_csv(ARQUIVO_ENTRADA)
    base, colunas_numericas = criar_variaveis_historicas(base_original)
    base = base[base["data_referencia"] <= LIMITE_VALIDACAO]
    colunas_modelo = colunas_numericas + COLUNAS_CATEGORICAS

    combinacoes = [
        (algoritmo, estrategia)
        for algoritmo in ALGORITMOS
        for estrategia in ESTRATEGIAS
    ]
    resultados: dict[str, list[dict[str, object]]] = {
        f"{algoritmo}__{estrategia}": []
        for algoritmo, estrategia in combinacoes
    }

    for ano_validacao in ANOS_VALIDACAO:
        treino_original, validacao = criar_particoes_temporais(
            base, ano_validacao
        )
        print(f"Validacao temporal de {ano_validacao}", flush=True)

        for algoritmo, estrategia in combinacoes:
            chave = f"{algoritmo}__{estrategia}"
            treino = aplicar_estrategia_treino(
                treino_original, estrategia, SEMENTE + ano_validacao
            )
            print(
                f"  {algoritmo} / {estrategia}: {len(treino)} linhas",
                flush=True,
            )
            inicio = time.perf_counter()
            modelo = criar_modelo(algoritmo, estrategia, colunas_numericas)
            modelo.fit(treino[colunas_modelo], treino[COLUNA_TARGET])
            probabilidades = modelo.predict_proba(validacao[colunas_modelo])[:, 1]
            limiar = escolher_limiar(validacao[COLUNA_TARGET], probabilidades)
            metricas = calcular_metricas(
                validacao[COLUNA_TARGET], probabilidades, limiar
            )
            metricas["ano_validacao"] = ano_validacao
            metricas["linhas_treino"] = len(treino)
            metricas["tempo_segundos"] = round(time.perf_counter() - inicio, 3)
            resultados[chave].append(metricas)

    resumos = {
        chave: resumir_validacoes(resultados_anuais)
        for chave, resultados_anuais in resultados.items()
    }
    ranking = sorted(
        resumos,
        key=lambda chave: resumos[chave]["average_precision_media"],
        reverse=True,
    )

    saida = {
        "etapa": "Comparacao de balanceamento com validacao temporal progressiva.",
        "teste_bloqueado": "consultas de 2024-07 a 2025-06",
        "criterio_ranking": "Media da average precision entre 2020 e 2023.",
        "ranking": ranking,
        "resumos": resumos,
        "resultados_por_ano": resultados,
    }
    RELATORIO_TEXTO.parent.mkdir(parents=True, exist_ok=True)
    RELATORIO_JSON.write_text(
        json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with RELATORIO_TEXTO.open("w", encoding="utf-8") as relatorio:
        relatorio.write("Comparacao de estrategias de balanceamento\n\n")
        relatorio.write("Validacoes: 2020, 2021, 2022 e 2023.\n")
        relatorio.write("Teste bloqueado: consultas de 2024-07 a 2025-06.\n")
        relatorio.write("Nenhuma reamostragem foi aplicada nas validacoes.\n\n")
        relatorio.write("Ranking por average precision media:\n")
        for posicao, chave in enumerate(ranking, start=1):
            metricas = resumos[chave]
            relatorio.write(
                f"{posicao}. {chave}: AP={metricas['average_precision_media']}, "
                f"precisao={metricas['precision_media']}, "
                f"recall={metricas['recall_media']}, "
                f"F2={metricas['f2_media']}, "
                f"tempo={metricas['tempo_total_segundos']}s\n"
            )

    print(f"Melhor combinacao na validacao: {ranking[0]}")
    print("O conjunto de teste nao foi acessado.")
    print(f"Relatorios salvos em: {RELATORIO_TEXTO} e {RELATORIO_JSON}")


if __name__ == "__main__":
    main()
