from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    from scripts.comparar_balanceamento import aplicar_estrategia_treino
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
    from comparar_balanceamento import aplicar_estrategia_treino
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
RELATORIO_TEXTO = Path("reports/resumo_otimizacao_regressao_logistica.txt")
RELATORIO_JSON = Path("reports/metricas_otimizacao_regressao_logistica.json")

ESTRATEGIA = "undersampling_1_5"
VALORES_C = (0.01, 0.1, 1.0, 10.0, 100.0)
PENALIDADES = ("l1", "l2")


def criar_modelo(
    penalidade: str, valor_c: float, colunas_numericas: list[str]
) -> Pipeline:
    return Pipeline(
        [
            ("preprocessamento", criar_preprocessador(colunas_numericas)),
            (
                "modelo",
                LogisticRegression(
                    C=valor_c,
                    l1_ratio=1.0 if penalidade == "l1" else 0.0,
                    solver="liblinear",
                    max_iter=2_000,
                    random_state=SEMENTE,
                ),
            ),
        ]
    )


def nome_configuracao(penalidade: str, valor_c: float) -> str:
    return f"penalidade_{penalidade}__C_{valor_c:g}"


def main() -> None:
    base_original = pd.read_csv(ARQUIVO_ENTRADA)
    base, colunas_numericas = criar_variaveis_historicas(base_original)
    base = base[base["data_referencia"] <= LIMITE_VALIDACAO]
    colunas_modelo = colunas_numericas + COLUNAS_CATEGORICAS
    configuracoes = [
        (penalidade, valor_c)
        for penalidade in PENALIDADES
        for valor_c in VALORES_C
    ]
    resultados: dict[str, list[dict[str, object]]] = {
        nome_configuracao(penalidade, valor_c): []
        for penalidade, valor_c in configuracoes
    }

    for ano_validacao in ANOS_VALIDACAO:
        treino_original, validacao = criar_particoes_temporais(
            base, ano_validacao
        )
        treino = aplicar_estrategia_treino(
            treino_original, ESTRATEGIA, SEMENTE + ano_validacao
        )
        print(
            f"Validacao {ano_validacao}: {len(treino)} linhas no treino 1:5",
            flush=True,
        )

        for penalidade, valor_c in configuracoes:
            chave = nome_configuracao(penalidade, valor_c)
            inicio = time.perf_counter()
            modelo = criar_modelo(penalidade, valor_c, colunas_numericas)
            modelo.fit(treino[colunas_modelo], treino[COLUNA_TARGET])
            probabilidades = modelo.predict_proba(validacao[colunas_modelo])[:, 1]
            limiar = escolher_limiar(validacao[COLUNA_TARGET], probabilidades)
            metricas = calcular_metricas(
                validacao[COLUNA_TARGET], probabilidades, limiar
            )
            metricas["ano_validacao"] = ano_validacao
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
    melhor_chave = ranking[0]

    saida = {
        "etapa": "Otimizacao da regressao logistica com undersampling 1:5.",
        "teste_bloqueado": "2024-01 a 2025-12",
        "criterio_ranking": "Media da average precision entre 2020 e 2023.",
        "melhor_configuracao": melhor_chave,
        "ranking": ranking,
        "resumos": resumos,
        "resultados_por_ano": resultados,
    }
    RELATORIO_TEXTO.parent.mkdir(parents=True, exist_ok=True)
    RELATORIO_JSON.write_text(
        json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with RELATORIO_TEXTO.open("w", encoding="utf-8") as relatorio:
        relatorio.write("Otimizacao da regressao logistica\n\n")
        relatorio.write("Treinamento com undersampling 1:5.\n")
        relatorio.write("Validacoes temporais: 2020, 2021, 2022 e 2023.\n")
        relatorio.write("Teste bloqueado: 2024-01 a 2025-12.\n\n")
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

    print(f"Melhor configuracao: {melhor_chave}")
    print("O conjunto de teste nao foi acessado.")
    print(f"Relatorios salvos em: {RELATORIO_TEXTO} e {RELATORIO_JSON}")


if __name__ == "__main__":
    main()
