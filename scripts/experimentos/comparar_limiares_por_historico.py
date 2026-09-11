from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.experimentos.analisar_falsos_positivos_populacional import (
        ARQUIVO_PREVISOES,
        LIMIAR,
        resumir_grupo,
    )
    from scripts.versoes.maximo_movel_6m.treinar_modelo_risco import COLUNA_MAXIMO, COLUNA_TARGET
except ModuleNotFoundError:
    from analisar_falsos_positivos_populacional import (  # type: ignore[no-redef]
        ARQUIVO_PREVISOES,
        LIMIAR,
        resumir_grupo,
    )
    from treinar_modelo_risco import (  # type: ignore[no-redef]
        COLUNA_MAXIMO,
        COLUNA_TARGET,
    )


RELATORIO_TEXTO = Path("reports/experimentos/resumo_limiares_por_historico.txt")
RELATORIO_JSON = Path("reports/experimentos/metricas_limiares_por_historico.json")
RECALL_MINIMO = 0.70


def candidatos_limiar(grupo: pd.DataFrame) -> pd.DataFrame:
    ordenado = grupo.sort_values("probabilidade_alerta", ascending=False)
    probabilidades = ordenado["probabilidade_alerta"].to_numpy()
    reais = ordenado[COLUNA_TARGET].astype(int).to_numpy()
    tp_acumulado = np.cumsum(reais)
    fp_acumulado = np.cumsum(1 - reais)
    finais_empate = np.r_[probabilidades[:-1] != probabilidades[1:], True]
    candidatos = pd.DataFrame(
        {
            "limiar": probabilidades[finais_empate],
            "tp": tp_acumulado[finais_empate],
            "fp": fp_acumulado[finais_empate],
        }
    )
    nenhum = pd.DataFrame(
        {
            "limiar": [float(np.nextafter(probabilidades.max(), np.inf))],
            "tp": [0],
            "fp": [0],
        }
    )
    return pd.concat([nenhum, candidatos], ignore_index=True)


def escolher_dois_limiares(base: pd.DataFrame) -> dict[str, float | int]:
    grupo_zero = base[base[COLUNA_MAXIMO] == 0]
    grupo_positivo = base[base[COLUNA_MAXIMO] > 0]
    candidatos_zero = candidatos_limiar(grupo_zero)
    candidatos_positivo = candidatos_limiar(grupo_positivo)
    minimo_tp = math.ceil(RECALL_MINIMO * int(base[COLUNA_TARGET].sum()))
    tp_positivo = candidatos_positivo["tp"].to_numpy()

    melhor: dict[str, float | int] | None = None
    for candidato_zero in candidatos_zero.itertuples(index=False):
        necessario = max(0, minimo_tp - int(candidato_zero.tp))
        indice = int(np.searchsorted(tp_positivo, necessario, side="left"))
        if indice >= len(candidatos_positivo):
            continue
        candidato_positivo = candidatos_positivo.iloc[indice]
        resultado = {
            "limiar_maximo_zero": float(candidato_zero.limiar),
            "limiar_maximo_positivo": float(candidato_positivo["limiar"]),
            "tp": int(candidato_zero.tp + candidato_positivo["tp"]),
            "fp": int(candidato_zero.fp + candidato_positivo["fp"]),
        }
        if melhor is None or (resultado["fp"], resultado["tp"]) < (
            melhor["fp"],
            melhor["tp"],
        ):
            melhor = resultado
    if melhor is None:
        raise ValueError("Nao foi possivel encontrar os dois limiares.")
    return melhor


def metricas(base: pd.DataFrame) -> dict[str, object]:
    resumo = resumir_grupo(base)
    return {
        chave: (None if pd.isna(valor) else round(float(valor), 6))
        for chave, valor in resumo.items()
    }


def main() -> None:
    base = pd.read_csv(ARQUIVO_PREVISOES)
    escolha = escolher_dois_limiares(base)
    base["alerta_limiar_unico"] = (
        base["probabilidade_alerta"] >= LIMIAR
    ).astype(int)
    base["alerta_dois_limiares"] = np.where(
        base[COLUNA_MAXIMO] == 0,
        base["probabilidade_alerta"] >= escolha["limiar_maximo_zero"],
        base["probabilidade_alerta"] >= escolha["limiar_maximo_positivo"],
    ).astype(int)

    comparacao: dict[str, dict[str, object]] = {}
    for nome, coluna in (
        ("limiar_unico", "alerta_limiar_unico"),
        ("dois_limiares", "alerta_dois_limiares"),
    ):
        avaliada = base.copy()
        avaliada["alerta_previsto"] = avaliada[coluna]
        por_contexto = {}
        for contexto, grupo in avaliada.groupby(
            np.where(avaliada[COLUNA_MAXIMO] == 0, "maximo_zero", "maximo_positivo")
        ):
            por_contexto[str(contexto)] = metricas(grupo)
        por_ano = {}
        for ano, grupo in avaliada.groupby("ano"):
            por_ano[str(int(ano))] = metricas(grupo)
        comparacao[nome] = {
            "total": metricas(avaliada),
            "por_contexto": por_contexto,
            "por_ano": por_ano,
        }

    saida = {
        "origem": "Previsoes fora do treino de 2020 a 2023.",
        "tipo_modelo": "classificacao_binaria",
        "target_alterado": False,
        "teste_final_usado": False,
        "recall_minimo_agrupado": RECALL_MINIMO,
        "limiar_unico": LIMIAR,
        "limiares_contextuais": escolha,
        "comparacao": comparacao,
    }
    RELATORIO_JSON.parent.mkdir(parents=True, exist_ok=True)
    RELATORIO_JSON.write_text(
        json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with RELATORIO_TEXTO.open("w", encoding="utf-8") as relatorio:
        relatorio.write("Comparacao de limiar unico e limiares por historico\n\n")
        relatorio.write("Classificacao binaria e target inalterado.\n")
        relatorio.write("Teste final nao utilizado.\n")
        relatorio.write(
            f"Limiar quando maximo historico = 0: "
            f"{escolha['limiar_maximo_zero']:.6f}\n"
        )
        relatorio.write(
            f"Limiar quando maximo historico > 0: "
            f"{escolha['limiar_maximo_positivo']:.6f}\n\n"
        )
        for nome, resultado in comparacao.items():
            total = resultado["total"]
            relatorio.write(
                f"{nome}: precisao={total['precisao']:.6f}, "
                f"recall={total['recall']:.6f}, FP={int(total['fp'])}, "
                f"FN={int(total['fn'])}, "
                f"falsos_por_acerto={total['falsos_por_acerto']:.6f}\n"
            )
            relatorio.write("  Por contexto:\n")
            for contexto, valores in resultado["por_contexto"].items():
                relatorio.write(
                    f"  - {contexto}: precisao={valores['precisao']:.6f}, "
                    f"recall={valores['recall']:.6f}, FP={int(valores['fp'])}\n"
                )
            relatorio.write("  Por ano:\n")
            for ano, valores in resultado["por_ano"].items():
                relatorio.write(
                    f"  - {ano}: precisao={valores['precisao']:.6f}, "
                    f"recall={valores['recall']:.6f}, FP={int(valores['fp'])}\n"
                )
            relatorio.write("\n")

    print(f"Comparacao salva em: {RELATORIO_TEXTO} e {RELATORIO_JSON}")


if __name__ == "__main__":
    main()
