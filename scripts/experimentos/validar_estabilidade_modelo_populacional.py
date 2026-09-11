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
RELATORIO_TEXTO = Path(
    "reports/experimentos/resumo_estabilidade_modelo_populacional.txt"
)
RELATORIO_JSON = Path(
    "reports/experimentos/metricas_estabilidade_modelo_populacional.json"
)
NOME_CONFIGURACAO = "folhas_100"
RECALL_MINIMO = 0.70
CONFIGURACAO = next(
    configuracao
    for configuracao in CONFIGURACOES
    if configuracao["nome"] == NOME_CONFIGURACAO
)


def main() -> None:
    base_original = pd.read_csv(ARQUIVO_ENTRADA)
    populacao = pd.read_csv(ARQUIVO_POPULACAO, dtype={"codigo_ibge": str})
    base, colunas_historicas = criar_variaveis_historicas(base_original)
    base = adicionar_populacao_disponivel(base, populacao)
    base, colunas_populacionais = criar_variaveis_populacionais(base)
    base = base[base["data_referencia"] <= LIMITE_VALIDACAO]

    colunas_numericas = colunas_historicas + colunas_populacionais
    colunas_modelo = colunas_numericas + COLUNAS_CATEGORICAS
    respostas_por_ano: dict[int, pd.Series] = {}
    probabilidades_por_ano: dict[int, np.ndarray] = {}

    for ano_validacao in ANOS_VALIDACAO:
        print(f"Validacao temporal de {ano_validacao}", flush=True)
        treino, validacao = criar_particoes_temporais(base, ano_validacao)
        modelo = criar_modelo(CONFIGURACAO, colunas_numericas)
        modelo.fit(treino[colunas_modelo], treino[COLUNA_TARGET])
        respostas_por_ano[ano_validacao] = validacao[COLUNA_TARGET]
        probabilidades_por_ano[ano_validacao] = modelo.predict_proba(
            validacao[colunas_modelo]
        )[:, 1]

    respostas_agrupadas = pd.Series(
        np.concatenate(
            [respostas_por_ano[ano].to_numpy() for ano in ANOS_VALIDACAO]
        )
    )
    probabilidades_agrupadas = np.concatenate(
        [probabilidades_por_ano[ano] for ano in ANOS_VALIDACAO]
    )
    limiar_unico = escolher_limiar_recall_minimo(
        respostas_agrupadas, probabilidades_agrupadas, RECALL_MINIMO
    )
    metricas_agrupadas = complementar_metricas(
        calcular_metricas(
            respostas_agrupadas, probabilidades_agrupadas, limiar_unico
        )
    )

    resultados_anuais: list[dict[str, object]] = []
    for ano in ANOS_VALIDACAO:
        metricas = complementar_metricas(
            calcular_metricas(
                respostas_por_ano[ano],
                probabilidades_por_ano[ano],
                limiar_unico,
            )
        )
        resultados_anuais.append({"ano_validacao": ano, "metricas": metricas})

    saida = {
        "etapa": "Estabilidade temporal da versao 2 populacional.",
        "tipo_modelo": "classificacao_binaria",
        "target_alterado": False,
        "teste_final_usado": False,
        "configuracao": CONFIGURACAO,
        "recall_minimo_agrupado": RECALL_MINIMO,
        "limiar_unico": round(limiar_unico, 6),
        "metricas_agrupadas": metricas_agrupadas,
        "resultados_anuais_com_mesmo_limiar": resultados_anuais,
    }
    RELATORIO_TEXTO.parent.mkdir(parents=True, exist_ok=True)
    RELATORIO_JSON.write_text(
        json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with RELATORIO_TEXTO.open("w", encoding="utf-8") as relatorio:
        relatorio.write("Estabilidade temporal da versao 2 populacional\n\n")
        relatorio.write("Classificacao binaria e target inalterado.\n")
        relatorio.write(f"Configuracao: {CONFIGURACAO}.\n")
        relatorio.write("Teste final nao utilizado.\n")
        relatorio.write(
            "O limiar e calculado uma vez nas validacoes agrupadas e aplicado "
            "igualmente a todos os anos.\n\n"
        )
        relatorio.write(f"Limiar unico: {limiar_unico:.6f}\n")
        matriz = metricas_agrupadas["matriz_confusao"]
        relatorio.write(
            "Resultado agrupado: "
            f"AP={metricas_agrupadas['average_precision']}, "
            f"precisao={metricas_agrupadas['precision']}, "
            f"recall={metricas_agrupadas['recall']}, "
            f"F2={metricas_agrupadas['f2']}, FP={matriz['fp']}, "
            f"FN={matriz['fn']}\n\n"
        )
        relatorio.write("Resultado por ano com o mesmo limiar:\n")
        for resultado in resultados_anuais:
            metricas = resultado["metricas"]
            matriz = metricas["matriz_confusao"]
            relatorio.write(
                f"- {resultado['ano_validacao']}: "
                f"AP={metricas['average_precision']}, "
                f"precisao={metricas['precision']}, "
                f"recall={metricas['recall']}, F2={metricas['f2']}, "
                f"FP={matriz['fp']}, FN={matriz['fn']}\n"
            )

    print(f"Limiar unico: {limiar_unico:.6f}")
    print("O conjunto de teste final nao foi acessado.")
    print(f"Relatorios salvos em: {RELATORIO_TEXTO} e {RELATORIO_JSON}")


if __name__ == "__main__":
    main()
