from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.versoes.maximo_movel_6m.analisar_limiares_modelo import (
        complementar_metricas,
        escolher_limiar_recall_minimo,
    )
    from scripts.experimentos.comparar_variaveis_populacionais import criar_variaveis_populacionais
    from scripts.versoes.taxa_100k.criar_target_taxa_100k import ARQUIVO_SAIDA, COLUNA_TARGET_TAXA
    from scripts.versoes.maximo_movel_6m.treinar_modelo_risco import (
        ANOS_VALIDACAO,
        COLUNAS_CATEGORICAS,
        COLUNA_TARGET,
        LIMITE_VALIDACAO,
        calcular_metricas,
        criar_modelos,
        criar_particoes_temporais,
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
    from treinar_modelo_risco import (  # type: ignore[no-redef]
        ANOS_VALIDACAO,
        COLUNAS_CATEGORICAS,
        COLUNA_TARGET,
        LIMITE_VALIDACAO,
        calcular_metricas,
        criar_modelos,
        criar_particoes_temporais,
        criar_variaveis_historicas,
    )


RELATORIO_TEXTO = Path("reports/versoes/taxa_100k/resumo_treinamento_modelo_taxa_100k.txt")
RELATORIO_JSON = Path("reports/versoes/taxa_100k/metricas_treinamento_modelo_taxa_100k.json")
RECALL_MINIMO = 0.70


def main() -> None:
    original = pd.read_csv(ARQUIVO_SAIDA)
    original[COLUNA_TARGET] = original[COLUNA_TARGET_TAXA]
    base, colunas_historicas = criar_variaveis_historicas(original)
    base, colunas_populacionais = criar_variaveis_populacionais(base)
    base = base[base["data_referencia"] <= LIMITE_VALIDACAO]
    colunas_numericas = colunas_historicas + colunas_populacionais
    colunas_modelo = colunas_numericas + COLUNAS_CATEGORICAS
    nomes = list(criar_modelos(colunas_numericas))
    respostas: list[np.ndarray] = []
    probabilidades: dict[str, list[np.ndarray]] = {nome: [] for nome in nomes}
    tempos = {nome: 0.0 for nome in nomes}

    for ano in ANOS_VALIDACAO:
        print(f"Validacao temporal de {ano}", flush=True)
        treino, validacao = criar_particoes_temporais(base, ano)
        respostas.append(validacao[COLUNA_TARGET].to_numpy())
        for nome, modelo in criar_modelos(colunas_numericas).items():
            print(f"  {nome}", flush=True)
            inicio = time.perf_counter()
            modelo.fit(treino[colunas_modelo], treino[COLUNA_TARGET])
            probabilidades[nome].append(
                modelo.predict_proba(validacao[colunas_modelo])[:, 1]
            )
            tempos[nome] += time.perf_counter() - inicio

    y_real = pd.Series(np.concatenate(respostas))
    resultados: dict[str, dict[str, object]] = {}
    for nome, blocos in probabilidades.items():
        probs = np.concatenate(blocos)
        limiar = escolher_limiar_recall_minimo(y_real, probs, RECALL_MINIMO)
        resultados[nome] = {
            "metricas": complementar_metricas(
                calcular_metricas(y_real, probs, limiar)
            ),
            "tempo_total_segundos": round(tempos[nome], 3),
        }

    ranking = sorted(
        nomes,
        key=lambda nome: (
            int(resultados[nome]["metricas"]["matriz_confusao"]["fp"]),
            -float(resultados[nome]["metricas"]["average_precision"]),
        ),
    )
    saida = {
        "etapa": "Comparacao inicial de algoritmos para o target proporcional.",
        "tipo_modelo": "classificacao_binaria",
        "target": COLUNA_TARGET_TAXA,
        "teste_final_usado": False,
        "criterio": "Menor FP com recall agrupado minimo de 70%.",
        "ranking": ranking,
        "melhor_modelo": ranking[0],
        "resultados": resultados,
    }
    RELATORIO_TEXTO.parent.mkdir(parents=True, exist_ok=True)
    RELATORIO_JSON.write_text(
        json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with RELATORIO_TEXTO.open("w", encoding="utf-8") as relatorio:
        relatorio.write("Treinamento inicial - target proporcional por 100 mil\n\n")
        relatorio.write("Classificacao binaria sem balanceamento artificial.\n")
        relatorio.write("Validacoes temporais: 2020 a 2023.\n")
        relatorio.write("Teste final nao utilizado.\n")
        relatorio.write("Ranking: menor FP com recall minimo de 70%.\n\n")
        for posicao, nome in enumerate(ranking, start=1):
            resultado = resultados[nome]
            metricas = resultado["metricas"]
            matriz = metricas["matriz_confusao"]
            relatorio.write(
                f"{posicao}. {nome}: AP={metricas['average_precision']}, "
                f"precisao={metricas['precision']}, "
                f"recall={metricas['recall']}, F2={metricas['f2']}, "
                f"FP={matriz['fp']}, FN={matriz['fn']}, "
                f"limiar={metricas['limiar']}, "
                f"tempo={resultado['tempo_total_segundos']}s\n"
            )
        relatorio.write(f"\nMelhor modelo: {ranking[0]}.\n")

    print(f"Melhor modelo: {ranking[0]}")
    print("O teste final nao foi acessado.")
    print(f"Relatorios salvos em: {RELATORIO_TEXTO} e {RELATORIO_JSON}")


if __name__ == "__main__":
    main()
