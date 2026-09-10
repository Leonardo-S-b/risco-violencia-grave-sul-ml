from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.otimizar_gradient_boosting import CONFIGURACOES, criar_modelo
    from scripts.treinar_modelo_risco import (
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


ARQUIVO_ENTRADA = Path("data/processed/violencia_grave_sul_com_target.csv")
RELATORIO_TEXTO = Path("reports/resumo_estabilidade_modelo.txt")
RELATORIO_JSON = Path("reports/metricas_estabilidade_modelo.json")

NOME_CONFIGURACAO = "conservador_15"
CONFIGURACAO = next(
    configuracao
    for configuracao in CONFIGURACOES
    if configuracao["nome"] == NOME_CONFIGURACAO
)


def main() -> None:
    base_original = pd.read_csv(ARQUIVO_ENTRADA)
    base, colunas_numericas = criar_variaveis_historicas(base_original)
    base = base[base["data_referencia"] <= LIMITE_VALIDACAO]
    colunas_modelo = colunas_numericas + COLUNAS_CATEGORICAS

    respostas_por_ano: dict[int, pd.Series] = {}
    probabilidades_por_ano: dict[int, np.ndarray] = {}
    respostas_agrupadas: list[np.ndarray] = []
    probabilidades_agrupadas: list[np.ndarray] = []

    for ano_validacao in ANOS_VALIDACAO:
        print(f"Validacao temporal de {ano_validacao}", flush=True)
        treino, validacao = criar_particoes_temporais(base, ano_validacao)
        modelo = criar_modelo(CONFIGURACAO, colunas_numericas)
        modelo.fit(treino[colunas_modelo], treino[COLUNA_TARGET])

        respostas = validacao[COLUNA_TARGET]
        probabilidades = modelo.predict_proba(validacao[colunas_modelo])[:, 1]
        respostas_por_ano[ano_validacao] = respostas
        probabilidades_por_ano[ano_validacao] = probabilidades
        respostas_agrupadas.append(respostas.to_numpy())
        probabilidades_agrupadas.append(probabilidades)

    respostas_fora_treino = pd.Series(np.concatenate(respostas_agrupadas))
    probabilidades_fora_treino = np.concatenate(probabilidades_agrupadas)
    limiar_unico = escolher_limiar(
        respostas_fora_treino, probabilidades_fora_treino
    )
    metricas_agrupadas = calcular_metricas(
        respostas_fora_treino, probabilidades_fora_treino, limiar_unico
    )

    resultados_anuais: list[dict[str, object]] = []
    for ano_validacao in ANOS_VALIDACAO:
        metricas = calcular_metricas(
            respostas_por_ano[ano_validacao],
            probabilidades_por_ano[ano_validacao],
            limiar_unico,
        )
        resultados_anuais.append(
            {"ano_validacao": ano_validacao, "metricas": metricas}
        )

    resumo_estabilidade: dict[str, float] = {}
    for nome_metrica in ("average_precision", "precision", "recall", "f2"):
        valores = [
            float(resultado["metricas"][nome_metrica])
            for resultado in resultados_anuais
        ]
        resumo_estabilidade[f"{nome_metrica}_media_entre_anos"] = round(
            float(np.mean(valores)), 6
        )
        resumo_estabilidade[f"{nome_metrica}_desvio_entre_anos"] = round(
            float(np.std(valores)), 6
        )

    parametros = {
        chave: valor for chave, valor in CONFIGURACAO.items() if chave != "nome"
    }
    saida = {
        "configuracao": {
            "algoritmo": "hist_gradient_boosting",
            "nome": NOME_CONFIGURACAO,
            "parametros": parametros,
            "balanceamento": "nenhum",
        },
        "validacoes": list(ANOS_VALIDACAO),
        "teste_bloqueado": "consultas de 2024-07 a 2025-06",
        "limiar_recomendado": round(limiar_unico, 6),
        "metricas_agrupadas": metricas_agrupadas,
        "resultados_anuais_com_mesmo_limiar": resultados_anuais,
        "resumo_estabilidade": resumo_estabilidade,
    }

    RELATORIO_TEXTO.parent.mkdir(parents=True, exist_ok=True)
    RELATORIO_JSON.write_text(
        json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with RELATORIO_TEXTO.open("w", encoding="utf-8") as relatorio:
        relatorio.write("Estabilidade do Gradient Boosting selecionado\n\n")
        relatorio.write(f"Configuracao: {NOME_CONFIGURACAO}, sem balanceamento.\n")
        relatorio.write(f"Parametros: {parametros}.\n")
        relatorio.write("Validacoes temporais: 2020, 2021, 2022 e 2023.\n")
        relatorio.write("O mesmo limiar e usado para todos os anos.\n")
        relatorio.write("Teste bloqueado: consultas de 2024-07 a 2025-06.\n\n")
        relatorio.write(f"Limiar recomendado: {limiar_unico:.6f}\n")
        relatorio.write(
            "Metricas agrupadas: "
            f"AP={metricas_agrupadas['average_precision']}, "
            f"precisao={metricas_agrupadas['precision']}, "
            f"recall={metricas_agrupadas['recall']}, "
            f"F2={metricas_agrupadas['f2']}\n\n"
        )
        relatorio.write("Resultado por ano com o limiar unico:\n")
        for resultado in resultados_anuais:
            metricas = resultado["metricas"]
            relatorio.write(
                f"- {resultado['ano_validacao']}: "
                f"AP={metricas['average_precision']}, "
                f"precisao={metricas['precision']}, "
                f"recall={metricas['recall']}, F2={metricas['f2']}\n"
            )
        relatorio.write("\nVariacao entre os anos:\n")
        for metrica, valor in resumo_estabilidade.items():
            relatorio.write(f"- {metrica}: {valor}\n")

    print(f"Limiar recomendado: {limiar_unico:.6f}")
    print("O conjunto de teste final nao foi acessado.")
    print(f"Relatorios salvos em: {RELATORIO_TEXTO} e {RELATORIO_JSON}")


if __name__ == "__main__":
    main()
