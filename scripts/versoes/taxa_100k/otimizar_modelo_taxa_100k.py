from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

try:
    from scripts.versoes.maximo_movel_6m.analisar_limiares_modelo import (
        complementar_metricas,
        escolher_limiar_recall_minimo,
    )
    from scripts.experimentos.comparar_variaveis_populacionais import criar_variaveis_populacionais
    from scripts.versoes.taxa_100k.criar_target_taxa_100k import ARQUIVO_SAIDA, COLUNA_TARGET_TAXA
    from scripts.versoes.maximo_movel_6m.otimizar_gradient_boosting import CONFIGURACOES, criar_modelo
    from scripts.versoes.maximo_movel_6m.treinar_modelo_risco import (
        ANOS_VALIDACAO,
        COLUNAS_CATEGORICAS,
        COLUNA_TARGET,
        LIMITE_VALIDACAO,
        SEMENTE,
        calcular_metricas,
        criar_particoes_temporais,
        criar_preprocessador,
        criar_variaveis_historicas,
    )
except ModuleNotFoundError:
    from analisar_limiares_modelo import (  # type: ignore[no-redef]
        complementar_metricas,
        escolher_limiar_recall_minimo,
    )
    from comparar_variaveis_populacionais import (  # type: ignore[no-redef]
        criar_variaveis_populacionais,
    )
    from criar_target_taxa_100k import (  # type: ignore[no-redef]
        ARQUIVO_SAIDA,
        COLUNA_TARGET_TAXA,
    )
    from otimizar_gradient_boosting import CONFIGURACOES, criar_modelo
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
    )


RELATORIO_TEXTO = Path("reports/versoes/taxa_100k/resumo_otimizacao_modelo_taxa_100k.txt")
RELATORIO_JSON = Path("reports/versoes/taxa_100k/metricas_otimizacao_modelo_taxa_100k.json")
RECALL_MINIMO = 0.70
NOMES_HGB = ("base", "folhas_50", "folhas_100", "conservador_15")
CONFIGURACOES_HGB = {
    str(configuracao["nome"]): configuracao
    for configuracao in CONFIGURACOES
    if configuracao["nome"] in NOMES_HGB
}
CONFIGURACOES_RF = {
    "rf_base": {"max_depth": 14, "min_samples_leaf": 5},
    "rf_folhas_10": {"max_depth": 14, "min_samples_leaf": 10},
    "rf_folhas_20": {"max_depth": 14, "min_samples_leaf": 20},
}


def criar_random_forest(
    configuracao: dict[str, int], colunas_numericas: list[str]
) -> Pipeline:
    return Pipeline(
        [
            ("preprocessamento", criar_preprocessador(colunas_numericas)),
            (
                "modelo",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=configuracao["max_depth"],
                    min_samples_leaf=configuracao["min_samples_leaf"],
                    n_jobs=-1,
                    random_state=SEMENTE,
                ),
            ),
        ]
    )


def main() -> None:
    original = pd.read_csv(ARQUIVO_SAIDA)
    original[COLUNA_TARGET] = original[COLUNA_TARGET_TAXA]
    base, colunas_historicas = criar_variaveis_historicas(original)
    base, colunas_populacionais = criar_variaveis_populacionais(base)
    base = base[base["data_referencia"] <= LIMITE_VALIDACAO]
    colunas_numericas = colunas_historicas + colunas_populacionais
    colunas_modelo = colunas_numericas + COLUNAS_CATEGORICAS
    nomes = [f"hgb_{nome}" for nome in CONFIGURACOES_HGB] + list(
        CONFIGURACOES_RF
    )
    respostas: list[np.ndarray] = []
    anos: list[np.ndarray] = []
    probabilidades: dict[str, list[np.ndarray]] = {nome: [] for nome in nomes}
    tempos = {nome: 0.0 for nome in nomes}

    for ano in ANOS_VALIDACAO:
        print(f"Validacao temporal de {ano}", flush=True)
        treino, validacao = criar_particoes_temporais(base, ano)
        respostas.append(validacao[COLUNA_TARGET].to_numpy())
        anos.append(np.full(len(validacao), ano))
        for nome_hgb, configuracao in CONFIGURACOES_HGB.items():
            nome = f"hgb_{nome_hgb}"
            print(f"  {nome}", flush=True)
            inicio = time.perf_counter()
            modelo = criar_modelo(configuracao, colunas_numericas)
            modelo.fit(treino[colunas_modelo], treino[COLUNA_TARGET])
            probabilidades[nome].append(
                modelo.predict_proba(validacao[colunas_modelo])[:, 1]
            )
            tempos[nome] += time.perf_counter() - inicio
        for nome, configuracao in CONFIGURACOES_RF.items():
            print(f"  {nome}", flush=True)
            inicio = time.perf_counter()
            modelo = criar_random_forest(configuracao, colunas_numericas)
            modelo.fit(treino[colunas_modelo], treino[COLUNA_TARGET])
            probabilidades[nome].append(
                modelo.predict_proba(validacao[colunas_modelo])[:, 1]
            )
            tempos[nome] += time.perf_counter() - inicio

    y_real = pd.Series(np.concatenate(respostas))
    anos_agrupados = np.concatenate(anos)
    resultados: dict[str, dict[str, object]] = {}
    for nome, blocos in probabilidades.items():
        probs = np.concatenate(blocos)
        limiar = escolher_limiar_recall_minimo(y_real, probs, RECALL_MINIMO)
        anuais = {}
        for ano in ANOS_VALIDACAO:
            mascara = anos_agrupados == ano
            anuais[str(ano)] = complementar_metricas(
                calcular_metricas(y_real[mascara], probs[mascara], limiar)
            )
        resultados[nome] = {
            "metricas_agrupadas": complementar_metricas(
                calcular_metricas(y_real, probs, limiar)
            ),
            "metricas_por_ano": anuais,
            "tempo_total_segundos": round(tempos[nome], 3),
        }

    ranking = sorted(
        nomes,
        key=lambda nome: (
            int(
                resultados[nome]["metricas_agrupadas"]["matriz_confusao"][
                    "fp"
                ]
            ),
            -float(
                resultados[nome]["metricas_agrupadas"]["average_precision"]
            ),
        ),
    )
    saida = {
        "etapa": "Otimizacao dos finalistas para o target proporcional.",
        "target": COLUNA_TARGET_TAXA,
        "teste_final_usado": False,
        "criterio": "Menor FP com recall agrupado minimo de 70%.",
        "ranking": ranking,
        "melhor_modelo": ranking[0],
        "configuracoes_hgb": CONFIGURACOES_HGB,
        "configuracoes_rf": CONFIGURACOES_RF,
        "resultados": resultados,
    }
    RELATORIO_TEXTO.parent.mkdir(parents=True, exist_ok=True)
    RELATORIO_JSON.write_text(
        json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with RELATORIO_TEXTO.open("w", encoding="utf-8") as relatorio:
        relatorio.write("Otimizacao do modelo para o target proporcional\n\n")
        relatorio.write("Teste final nao utilizado.\n")
        relatorio.write("Mesmo limiar aplicado a 2020, 2021, 2022 e 2023.\n")
        relatorio.write("Ranking: menor FP com recall minimo agrupado de 70%.\n\n")
        for posicao, nome in enumerate(ranking, start=1):
            resultado = resultados[nome]
            metricas = resultado["metricas_agrupadas"]
            matriz = metricas["matriz_confusao"]
            relatorio.write(
                f"{posicao}. {nome}: AP={metricas['average_precision']}, "
                f"precisao={metricas['precision']}, "
                f"recall={metricas['recall']}, F2={metricas['f2']}, "
                f"FP={matriz['fp']}, FN={matriz['fn']}, "
                f"limiar={metricas['limiar']}, "
                f"tempo={resultado['tempo_total_segundos']}s\n"
            )
        vencedor = resultados[ranking[0]]
        relatorio.write(f"\nMelhor modelo: {ranking[0]}.\n")
        relatorio.write("Estabilidade anual do vencedor:\n")
        for ano, metricas in vencedor["metricas_por_ano"].items():
            matriz = metricas["matriz_confusao"]
            relatorio.write(
                f"- {ano}: precisao={metricas['precision']}, "
                f"recall={metricas['recall']}, FP={matriz['fp']}, "
                f"FN={matriz['fn']}\n"
            )

    print(f"Melhor modelo: {ranking[0]}")
    print("O teste final nao foi acessado.")
    print(f"Relatorios salvos em: {RELATORIO_TEXTO} e {RELATORIO_JSON}")


if __name__ == "__main__":
    main()
