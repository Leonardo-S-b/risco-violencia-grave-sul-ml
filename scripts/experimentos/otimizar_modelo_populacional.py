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
    from scripts.experimentos.comparar_variaveis_populacionais import (
        ARQUIVO_POPULACAO,
        adicionar_populacao_disponivel,
        criar_variaveis_populacionais,
    )
    from scripts.versoes.maximo_movel_6m.otimizar_gradient_boosting import CONFIGURACOES, criar_modelo
    from scripts.versoes.maximo_movel_6m.treinar_modelo_risco import (
        ANOS_VALIDACAO,
        COLUNAS_CATEGORICAS,
        COLUNA_TARGET,
        LIMITE_VALIDACAO,
        calcular_metricas,
        criar_particoes_temporais,
        criar_variaveis_historicas,
    )
except ModuleNotFoundError:
    from analisar_limiares_modelo import (  # type: ignore[no-redef]
        complementar_metricas,
        escolher_limiar_recall_minimo,
    )
    from comparar_variaveis_populacionais import (  # type: ignore[no-redef]
        ARQUIVO_POPULACAO,
        adicionar_populacao_disponivel,
        criar_variaveis_populacionais,
    )
    from otimizar_gradient_boosting import CONFIGURACOES, criar_modelo
    from treinar_modelo_risco import (  # type: ignore[no-redef]
        ANOS_VALIDACAO,
        COLUNAS_CATEGORICAS,
        COLUNA_TARGET,
        LIMITE_VALIDACAO,
        calcular_metricas,
        criar_particoes_temporais,
        criar_variaveis_historicas,
    )


ARQUIVO_ENTRADA = Path("data/processed/versoes/maximo_movel_6m/violencia_grave_sul_com_target.csv")
RELATORIO_TEXTO = Path("reports/experimentos/resumo_otimizacao_modelo_populacional.txt")
RELATORIO_JSON = Path("reports/experimentos/metricas_otimizacao_modelo_populacional.json")
RECALL_MINIMO = 0.70


def main() -> None:
    base_original = pd.read_csv(ARQUIVO_ENTRADA)
    populacao = pd.read_csv(ARQUIVO_POPULACAO, dtype={"codigo_ibge": str})
    base, colunas_numericas_originais = criar_variaveis_historicas(base_original)
    base = adicionar_populacao_disponivel(base, populacao)
    base, colunas_populacionais = criar_variaveis_populacionais(base)
    base = base[base["data_referencia"] <= LIMITE_VALIDACAO]
    colunas_numericas = colunas_numericas_originais + colunas_populacionais
    colunas_modelo = colunas_numericas + COLUNAS_CATEGORICAS

    respostas: list[np.ndarray] = []
    probabilidades: dict[str, list[np.ndarray]] = {
        str(configuracao["nome"]): [] for configuracao in CONFIGURACOES
    }
    tempos: dict[str, float] = {nome: 0.0 for nome in probabilidades}

    for ano_validacao in ANOS_VALIDACAO:
        treino, validacao = criar_particoes_temporais(base, ano_validacao)
        respostas.append(validacao[COLUNA_TARGET].to_numpy())
        print(f"Validacao temporal de {ano_validacao}", flush=True)
        for configuracao in CONFIGURACOES:
            nome = str(configuracao["nome"])
            print(f"  {nome}", flush=True)
            inicio = time.perf_counter()
            modelo = criar_modelo(configuracao, colunas_numericas)
            modelo.fit(treino[colunas_modelo], treino[COLUNA_TARGET])
            probabilidades[nome].append(
                modelo.predict_proba(validacao[colunas_modelo])[:, 1]
            )
            tempos[nome] += time.perf_counter() - inicio

    y_real = pd.Series(np.concatenate(respostas))
    resultados: dict[str, dict[str, object]] = {}
    configuracoes_por_nome = {
        str(configuracao["nome"]): configuracao for configuracao in CONFIGURACOES
    }
    for nome, blocos in probabilidades.items():
        probabilidades_agrupadas = np.concatenate(blocos)
        limiar = escolher_limiar_recall_minimo(
            y_real, probabilidades_agrupadas, RECALL_MINIMO
        )
        resultados[nome] = {
            "configuracao": configuracoes_por_nome[nome],
            "metricas": complementar_metricas(
                calcular_metricas(y_real, probabilidades_agrupadas, limiar)
            ),
            "tempo_total_segundos": round(tempos[nome], 3),
        }

    ranking = sorted(
        resultados,
        key=lambda nome: (
            int(resultados[nome]["metricas"]["matriz_confusao"]["fp"]),
            -float(resultados[nome]["metricas"]["average_precision"]),
        ),
    )
    vencedor = ranking[0]
    saida = {
        "etapa": "Otimizacao da versao 2 com variaveis populacionais.",
        "tipo_modelo": "classificacao_binaria",
        "target_alterado": False,
        "teste_final_usado": False,
        "criterio": (
            "Menor quantidade de falsos positivos com recall minimo de 70%."
        ),
        "melhor_configuracao": resultados[vencedor],
        "ranking": ranking,
        "resultados": resultados,
    }
    RELATORIO_TEXTO.parent.mkdir(parents=True, exist_ok=True)
    RELATORIO_JSON.write_text(
        json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with RELATORIO_TEXTO.open("w", encoding="utf-8") as relatorio:
        relatorio.write("Otimizacao do classificador populacional - versao 2\n\n")
        relatorio.write("Target binario inalterado. Teste final nao utilizado.\n")
        relatorio.write(
            "Ranking: menor quantidade de falsos positivos mantendo recall "
            "minimo de 70%.\n\n"
        )
        for posicao, nome in enumerate(ranking, start=1):
            resultado = resultados[nome]
            metricas = resultado["metricas"]
            matriz = metricas["matriz_confusao"]
            relatorio.write(
                f"{posicao}. {nome}: AP={metricas['average_precision']}, "
                f"precisao={metricas['precision']}, "
                f"recall={metricas['recall']}, F2={metricas['f2']}, "
                f"limiar={metricas['limiar']}, FP={matriz['fp']}, "
                f"FN={matriz['fn']}, tempo={resultado['tempo_total_segundos']}s\n"
            )
        relatorio.write(f"\nMelhor configuracao: {vencedor}.\n")
        relatorio.write(f"{resultados[vencedor]['configuracao']}\n")

    print(f"Melhor configuracao: {vencedor}")
    print(f"Relatorios salvos em: {RELATORIO_TEXTO} e {RELATORIO_JSON}")


if __name__ == "__main__":
    main()
