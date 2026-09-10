from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
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
RELATORIO_TEXTO = Path("reports/resumo_otimizacao_gradient_boosting.txt")
RELATORIO_JSON = Path("reports/metricas_otimizacao_gradient_boosting.json")

# Grade curta e explicita para manter o experimento reproduzivel.
CONFIGURACOES = (
    {"nome": "base", "learning_rate": 0.08, "max_iter": 150, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.1},
    {"nome": "aprendizado_lento", "learning_rate": 0.05, "max_iter": 250, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.1},
    {"nome": "aprendizado_muito_lento", "learning_rate": 0.03, "max_iter": 350, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.1},
    {"nome": "aprendizado_rapido", "learning_rate": 0.1, "max_iter": 120, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.1},
    {"nome": "arvores_menores", "learning_rate": 0.08, "max_iter": 150, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 0.1},
    {"nome": "arvores_maiores", "learning_rate": 0.08, "max_iter": 150, "max_leaf_nodes": 63, "min_samples_leaf": 20, "l2_regularization": 0.1},
    {"nome": "folhas_50", "learning_rate": 0.08, "max_iter": 150, "max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 0.1},
    {"nome": "folhas_100", "learning_rate": 0.08, "max_iter": 150, "max_leaf_nodes": 31, "min_samples_leaf": 100, "l2_regularization": 0.1},
    {"nome": "sem_l2", "learning_rate": 0.08, "max_iter": 150, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.0},
    {"nome": "l2_1", "learning_rate": 0.08, "max_iter": 150, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 1.0},
    {"nome": "l2_10", "learning_rate": 0.08, "max_iter": 150, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 10.0},
    {"nome": "conservador_15", "learning_rate": 0.05, "max_iter": 250, "max_leaf_nodes": 15, "min_samples_leaf": 50, "l2_regularization": 1.0},
    {"nome": "conservador_31", "learning_rate": 0.05, "max_iter": 250, "max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 1.0},
)


def criar_modelo(
    configuracao: dict[str, object], colunas_numericas: list[str]
) -> Pipeline:
    return Pipeline(
        [
            ("preprocessamento", criar_preprocessador(colunas_numericas)),
            (
                "modelo",
                HistGradientBoostingClassifier(
                    learning_rate=float(configuracao["learning_rate"]),
                    max_iter=int(configuracao["max_iter"]),
                    max_leaf_nodes=int(configuracao["max_leaf_nodes"]),
                    min_samples_leaf=int(configuracao["min_samples_leaf"]),
                    l2_regularization=float(configuracao["l2_regularization"]),
                    early_stopping=False,
                    random_state=SEMENTE,
                ),
            ),
        ]
    )


def main() -> None:
    base_original = pd.read_csv(ARQUIVO_ENTRADA)
    base, colunas_numericas = criar_variaveis_historicas(base_original)
    base = base[base["data_referencia"] <= LIMITE_VALIDACAO]
    colunas_modelo = colunas_numericas + COLUNAS_CATEGORICAS
    resultados: dict[str, list[dict[str, object]]] = {
        str(configuracao["nome"]): [] for configuracao in CONFIGURACOES
    }

    for ano_validacao in ANOS_VALIDACAO:
        treino, validacao = criar_particoes_temporais(base, ano_validacao)
        print(f"Validacao temporal de {ano_validacao}", flush=True)
        for configuracao in CONFIGURACOES:
            nome = str(configuracao["nome"])
            print(f"  {nome}", flush=True)
            inicio = time.perf_counter()
            modelo = criar_modelo(configuracao, colunas_numericas)
            modelo.fit(treino[colunas_modelo], treino[COLUNA_TARGET])
            probabilidades = modelo.predict_proba(validacao[colunas_modelo])[:, 1]
            limiar = escolher_limiar(validacao[COLUNA_TARGET], probabilidades)
            metricas = calcular_metricas(
                validacao[COLUNA_TARGET], probabilidades, limiar
            )
            metricas["ano_validacao"] = ano_validacao
            metricas["tempo_segundos"] = round(time.perf_counter() - inicio, 3)
            resultados[nome].append(metricas)

    resumos = {
        nome: resumir_validacoes(resultados_anuais)
        for nome, resultados_anuais in resultados.items()
    }
    ranking = sorted(
        resumos,
        key=lambda nome: resumos[nome]["average_precision_media"],
        reverse=True,
    )
    configuracoes_por_nome = {
        str(configuracao["nome"]): configuracao for configuracao in CONFIGURACOES
    }
    melhor_nome = ranking[0]
    saida = {
        "etapa": "Otimizacao do HistGradientBoosting sem balanceamento.",
        "teste_bloqueado": "consultas de 2024-07 a 2025-06",
        "criterio_ranking": "Media da average precision entre 2020 e 2023.",
        "melhor_configuracao": configuracoes_por_nome[melhor_nome],
        "ranking": ranking,
        "configuracoes": configuracoes_por_nome,
        "resumos": resumos,
        "resultados_por_ano": resultados,
    }
    RELATORIO_TEXTO.parent.mkdir(parents=True, exist_ok=True)
    RELATORIO_JSON.write_text(
        json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with RELATORIO_TEXTO.open("w", encoding="utf-8") as relatorio:
        relatorio.write("Otimizacao do Gradient Boosting\n\n")
        relatorio.write("Estrategia: dados originais, sem balanceamento.\n")
        relatorio.write("Validacoes temporais: 2020, 2021, 2022 e 2023.\n")
        relatorio.write("Teste bloqueado: consultas de 2024-07 a 2025-06.\n\n")
        relatorio.write("Ranking por average precision media:\n")
        for posicao, nome in enumerate(ranking, start=1):
            metricas = resumos[nome]
            relatorio.write(
                f"{posicao}. {nome}: AP={metricas['average_precision_media']}, "
                f"precisao={metricas['precision_media']}, "
                f"recall={metricas['recall_media']}, "
                f"F2={metricas['f2_media']}, "
                f"tempo={metricas['tempo_total_segundos']}s\n"
            )
        relatorio.write("\nMelhor configuracao:\n")
        relatorio.write(f"{configuracoes_por_nome[melhor_nome]}\n")

    print(f"Melhor configuracao: {melhor_nome}")
    print("O conjunto de teste nao foi acessado.")
    print(f"Relatorios salvos em: {RELATORIO_TEXTO} e {RELATORIO_JSON}")


if __name__ == "__main__":
    main()
