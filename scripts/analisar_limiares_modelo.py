from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

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
    )


ARQUIVO_ENTRADA = Path("data/processed/violencia_grave_sul_com_target.csv")
RELATORIO_TEXTO = Path("reports/resumo_analise_limiares.txt")
RELATORIO_JSON = Path("reports/metricas_analise_limiares.json")
NOME_CONFIGURACAO = "conservador_15"
LIMIAR_ATUAL = 0.108100
CONFIGURACAO = next(
    configuracao
    for configuracao in CONFIGURACOES
    if configuracao["nome"] == NOME_CONFIGURACAO
)


def escolher_limiar_recall_minimo(
    respostas: pd.Series, probabilidades: np.ndarray, recall_minimo: float
) -> float:
    precisoes, recalls, limiares = precision_recall_curve(respostas, probabilidades)
    candidatos = np.flatnonzero(recalls[:-1] >= recall_minimo)
    if len(candidatos) == 0:
        raise ValueError(f"Nao existe limiar com recall minimo de {recall_minimo}.")
    melhor = candidatos[int(np.argmax(precisoes[:-1][candidatos]))]
    return float(limiares[melhor])


def escolher_limiar_f1(respostas: pd.Series, probabilidades: np.ndarray) -> float:
    precisoes, recalls, limiares = precision_recall_curve(respostas, probabilidades)
    denominador = precisoes[:-1] + recalls[:-1]
    valores_f1 = np.divide(
        2 * precisoes[:-1] * recalls[:-1],
        denominador,
        out=np.zeros_like(denominador),
        where=denominador > 0,
    )
    return float(limiares[int(np.argmax(valores_f1))])


def complementar_metricas(metricas: dict[str, object]) -> dict[str, object]:
    matriz = metricas["matriz_confusao"]
    quantidade_alertas = int(matriz["tp"]) + int(matriz["fp"])
    verdadeiros = int(matriz["tp"])
    return {
        **metricas,
        "quantidade_alertas": quantidade_alertas,
        "taxa_alertas": round(quantidade_alertas / int(metricas["quantidade"]), 6),
        "falsos_positivos_por_verdadeiro_positivo": round(
            int(matriz["fp"]) / verdadeiros, 6
        )
        if verdadeiros
        else None,
    }


def main() -> None:
    base_original = pd.read_csv(ARQUIVO_ENTRADA)
    base, colunas_numericas = criar_variaveis_historicas(base_original)
    base = base[base["data_referencia"] <= LIMITE_VALIDACAO]
    colunas_modelo = colunas_numericas + COLUNAS_CATEGORICAS

    respostas: list[np.ndarray] = []
    probabilidades: list[np.ndarray] = []
    for ano_validacao in ANOS_VALIDACAO:
        print(f"Gerando previsoes fora do treino para {ano_validacao}", flush=True)
        treino, validacao = criar_particoes_temporais(base, ano_validacao)
        modelo = criar_modelo(CONFIGURACAO, colunas_numericas)
        modelo.fit(treino[colunas_modelo], treino[COLUNA_TARGET])
        respostas.append(validacao[COLUNA_TARGET].to_numpy())
        probabilidades.append(
            modelo.predict_proba(validacao[colunas_modelo])[:, 1]
        )

    y_real = pd.Series(np.concatenate(respostas))
    probabilidades_oof = np.concatenate(probabilidades)
    politicas = [("f2_atual", LIMIAR_ATUAL)]
    politicas.append(("melhor_f1", escolher_limiar_f1(y_real, probabilidades_oof)))
    for recall_minimo in (0.85, 0.80, 0.75, 0.70, 0.60):
        politicas.append(
            (
                f"recall_minimo_{int(recall_minimo * 100)}",
                escolher_limiar_recall_minimo(
                    y_real, probabilidades_oof, recall_minimo
                ),
            )
        )

    resultados = {
        nome: complementar_metricas(
            calcular_metricas(y_real, probabilidades_oof, limiar)
        )
        for nome, limiar in politicas
    }
    saida = {
        "origem": "previsoes fora do treino das validacoes 2020-2023",
        "teste_final_usado_para_escolher_limiar": False,
        "resultados": resultados,
    }
    RELATORIO_TEXTO.parent.mkdir(parents=True, exist_ok=True)
    RELATORIO_JSON.write_text(
        json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with RELATORIO_TEXTO.open("w", encoding="utf-8") as relatorio:
        relatorio.write("Analise de limites de alerta nas validacoes 2020-2023\n\n")
        relatorio.write("O teste final nao foi usado para escolher estes limites.\n")
        relatorio.write(
            "Limites maiores reduzem falsos alertas, mas deixam passar mais casos.\n\n"
        )
        for nome, metricas in resultados.items():
            matriz = metricas["matriz_confusao"]
            relatorio.write(
                f"- {nome}: limiar={metricas['limiar']}, "
                f"precisao={metricas['precision']}, recall={metricas['recall']}, "
                f"F1={metricas['f1']}, F2={metricas['f2']}, "
                f"alertas={metricas['quantidade_alertas']}, "
                f"FP={matriz['fp']}, FN={matriz['fn']}, "
                "falsos_por_acerto="
                f"{metricas['falsos_positivos_por_verdadeiro_positivo']}\n"
            )

    print(f"Analise salva em: {RELATORIO_TEXTO} e {RELATORIO_JSON}")


if __name__ == "__main__":
    main()
