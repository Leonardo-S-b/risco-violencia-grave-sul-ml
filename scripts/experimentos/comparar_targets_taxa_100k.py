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
RELATORIO_TEXTO = Path("reports/experimentos/resumo_comparacao_targets_taxa_100k.txt")
RELATORIO_JSON = Path("reports/experimentos/metricas_comparacao_targets_taxa_100k.json")
NOME_CONFIGURACAO = "folhas_100"
PERCENTIL_CORTE = 0.80
RECALL_MINIMO = 0.70
CONFIGURACAO = next(
    configuracao
    for configuracao in CONFIGURACOES
    if configuracao["nome"] == NOME_CONFIGURACAO
)


def adicionar_taxa_futura_6m(base: pd.DataFrame) -> pd.DataFrame:
    """Calcula a incidencia futura usada apenas como resposta supervisionada."""
    base = base.copy()
    grupo = base.groupby(["uf", "municipio"], sort=False)[
        "total_violencia_grave"
    ]
    valores_futuros = [grupo.shift(-mes) for mes in range(1, 7)]
    base["soma_violencia_grave_proximos_6m"] = sum(valores_futuros)
    base["meses_com_violencia_grave_proximos_6m"] = sum(
        (valores > 0).astype(int) for valores in valores_futuros
    )
    base["taxa_violencia_grave_proximos_6m_100k"] = (
        base["soma_violencia_grave_proximos_6m"]
        / base["populacao"]
        * 100_000
    )
    return base


def main() -> None:
    original = pd.read_csv(ARQUIVO_ENTRADA)
    original["data_referencia"] = pd.to_datetime(original["data_referencia"])
    populacao = pd.read_csv(ARQUIVO_POPULACAO, dtype={"codigo_ibge": str})
    base = adicionar_populacao_disponivel(original, populacao)
    base = adicionar_taxa_futura_6m(base)

    referencia = base[
        (base["data_referencia"] < pd.Timestamp("2020-01-01"))
        & base["taxa_violencia_grave_proximos_6m_100k"].notna()
    ]
    corte = float(
        referencia["taxa_violencia_grave_proximos_6m_100k"].quantile(
            PERCENTIL_CORTE
        )
    )
    taxa_acima_corte = (
        base["taxa_violencia_grave_proximos_6m_100k"] > corte
    )
    regras = {
        "taxa_pura": taxa_acima_corte,
        "taxa_e_minimo_2_casos": taxa_acima_corte
        & (base["soma_violencia_grave_proximos_6m"] >= 2),
        "taxa_e_ocorrencia_em_2_meses": taxa_acima_corte
        & (base["meses_com_violencia_grave_proximos_6m"] >= 2),
    }

    resultados: dict[str, dict[str, object]] = {}
    for nome, regra in regras.items():
        print(f"Target {nome}: corte={corte:.6f}", flush=True)
        experimento = base.copy()
        elegivel = experimento[
            "taxa_violencia_grave_proximos_6m_100k"
        ].notna()
        experimento[COLUNA_TARGET] = np.where(
            elegivel,
            regra.astype(int),
            np.nan,
        )
        experimento, colunas_historicas = criar_variaveis_historicas(experimento)
        experimento, colunas_populacionais = criar_variaveis_populacionais(
            experimento
        )
        experimento = experimento[
            experimento["data_referencia"] <= LIMITE_VALIDACAO
        ]
        colunas_numericas = colunas_historicas + colunas_populacionais
        colunas_modelo = colunas_numericas + COLUNAS_CATEGORICAS
        respostas: list[np.ndarray] = []
        probabilidades: list[np.ndarray] = []
        anos: list[np.ndarray] = []

        for ano in ANOS_VALIDACAO:
            print(f"  validacao {ano}", flush=True)
            treino, validacao = criar_particoes_temporais(experimento, ano)
            modelo = criar_modelo(CONFIGURACAO, colunas_numericas)
            modelo.fit(treino[colunas_modelo], treino[COLUNA_TARGET])
            respostas.append(validacao[COLUNA_TARGET].to_numpy())
            probabilidades.append(
                modelo.predict_proba(validacao[colunas_modelo])[:, 1]
            )
            anos.append(np.full(len(validacao), ano))

        y_real = pd.Series(np.concatenate(respostas))
        probs = np.concatenate(probabilidades)
        anos_agrupados = np.concatenate(anos)
        limiar = escolher_limiar_recall_minimo(
            y_real, probs, RECALL_MINIMO
        )
        gerais = complementar_metricas(calcular_metricas(y_real, probs, limiar))
        anuais = {}
        for ano in ANOS_VALIDACAO:
            mascara = anos_agrupados == ano
            anuais[str(ano)] = complementar_metricas(
                calcular_metricas(y_real[mascara], probs[mascara], limiar)
            )
        resultados[nome] = {
            "corte_taxa_6m_por_100k": round(corte, 6),
            "regra_evidencia_minima": nome,
            "prevalencia_referencia_antes_2020": round(
                float(
                    regra.loc[referencia.index].mean()
                ),
                6,
            ),
            "limiar_modelo": round(limiar, 6),
            "metricas_agrupadas": gerais,
            "metricas_por_ano": anuais,
        }

    saida = {
        "etapa": "Experimento de targets proporcionais por 100 mil habitantes.",
        "definicao": (
            "Target 1 quando a soma de violencia grave nos proximos seis meses, "
            "dividida pela populacao conhecida na consulta, ultrapassa o corte."
        ),
        "origem_cortes": (
            "Percentil 80 da taxa futura apenas no periodo anterior a 2020."
        ),
        "tipo_modelo": "classificacao_binaria",
        "target_atual_sobrescrito": False,
        "teste_final_usado": False,
        "resultados": resultados,
    }
    RELATORIO_JSON.parent.mkdir(parents=True, exist_ok=True)
    RELATORIO_JSON.write_text(
        json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with RELATORIO_TEXTO.open("w", encoding="utf-8") as relatorio:
        relatorio.write("Comparacao de targets por 100 mil habitantes\n\n")
        relatorio.write(
            "Pergunta: a taxa acumulada nos proximos seis meses ultrapassara "
            "o corte por 100 mil habitantes, com ou sem evidencia minima?\n"
        )
        relatorio.write(
            "Corte definido pelo percentil 80 somente com dados anteriores a "
            "2020.\n"
        )
        relatorio.write("Teste final nao utilizado. Target atual preservado.\n\n")
        for nome, resultado in resultados.items():
            metricas = resultado["metricas_agrupadas"]
            matriz = metricas["matriz_confusao"]
            relatorio.write(
                f"{nome}: corte={resultado['corte_taxa_6m_por_100k']}, "
                f"prevalencia_pre2020="
                f"{resultado['prevalencia_referencia_antes_2020']}, "
                f"AP={metricas['average_precision']}, "
                f"precisao={metricas['precision']}, "
                f"recall={metricas['recall']}, F2={metricas['f2']}, "
                f"FP={matriz['fp']}, FN={matriz['fn']}\n"
            )
            for ano, anual in resultado["metricas_por_ano"].items():
                matriz_ano = anual["matriz_confusao"]
                relatorio.write(
                    f"  - {ano}: prevalencia="
                    f"{anual['positivos'] / anual['quantidade']:.6f}, "
                    f"precisao={anual['precision']}, recall={anual['recall']}, "
                    f"FP={matriz_ano['fp']}, FN={matriz_ano['fn']}\n"
                )
            relatorio.write("\n")

    print(f"Comparacao salva em: {RELATORIO_TEXTO} e {RELATORIO_JSON}")


if __name__ == "__main__":
    main()
