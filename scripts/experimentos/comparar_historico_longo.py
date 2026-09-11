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
RELATORIO_TEXTO = Path("reports/experimentos/resumo_comparacao_historico_longo.txt")
RELATORIO_JSON = Path("reports/experimentos/metricas_comparacao_historico_longo.json")
NOME_CONFIGURACAO = "folhas_100"
RECALL_MINIMO = 0.70
CONFIGURACAO = next(
    configuracao
    for configuracao in CONFIGURACOES
    if configuracao["nome"] == NOME_CONFIGURACAO
)


def criar_variaveis_historico_longo(
    base: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    base = base.copy()
    grupo = base.groupby(["uf", "municipio"], sort=False)
    colunas: list[str] = []
    ocorrencia = (base["total_violencia_grave"] > 0).astype(int)

    for janela in (24, 36):
        coluna_media = f"media_movel_{janela}m"
        coluna_desvio = f"desvio_movel_{janela}m"
        coluna_meses = f"meses_com_ocorrencia_{janela}m"
        coluna_taxa = f"taxa_violencia_{janela}m_100k"
        base[coluna_media] = grupo["total_violencia_grave"].transform(
            lambda valores: valores.rolling(janela, min_periods=janela).mean()
        )
        base[coluna_desvio] = grupo["total_violencia_grave"].transform(
            lambda valores: valores.rolling(janela, min_periods=janela).std()
        )
        base[coluna_meses] = ocorrencia.groupby(
            [base["uf"], base["municipio"]], sort=False
        ).transform(
            lambda valores: valores.rolling(janela, min_periods=janela).sum()
        )
        base[coluna_taxa] = (
            base[coluna_media] * janela / base["populacao"] * 100_000
        )
        colunas.extend(
            [coluna_media, coluna_desvio, coluna_meses, coluna_taxa]
        )

    data_evento = base["data_referencia"].where(ocorrencia == 1)
    ultima_data = data_evento.groupby(
        [base["uf"], base["municipio"]], sort=False
    ).ffill()
    meses_desde = (
        (base["data_referencia"].dt.year - ultima_data.dt.year) * 12
        + base["data_referencia"].dt.month
        - ultima_data.dt.month
    )
    base["sem_ocorrencia_anterior"] = ultima_data.isna().astype(int)
    base["meses_desde_ultima_ocorrencia"] = meses_desde.fillna(120).clip(0, 120)
    colunas.extend(
        ["sem_ocorrencia_anterior", "meses_desde_ultima_ocorrencia"]
    )
    return base, colunas


def main() -> None:
    original = pd.read_csv(ARQUIVO_ENTRADA)
    populacao = pd.read_csv(ARQUIVO_POPULACAO, dtype={"codigo_ibge": str})
    base, colunas_historicas = criar_variaveis_historicas(original)
    base = adicionar_populacao_disponivel(base, populacao)
    base, colunas_populacionais = criar_variaveis_populacionais(base)
    base, colunas_longo_prazo = criar_variaveis_historico_longo(base)
    base = base[base["data_referencia"] <= LIMITE_VALIDACAO]

    colunas_v2 = colunas_historicas + colunas_populacionais
    colunas_por_modelo = {
        "populacional_v2": colunas_v2,
        "populacional_com_historico_longo": colunas_v2 + colunas_longo_prazo,
    }
    respostas: list[np.ndarray] = []
    probabilidades: dict[str, list[np.ndarray]] = {
        nome: [] for nome in colunas_por_modelo
    }
    anos: list[np.ndarray] = []

    for ano in ANOS_VALIDACAO:
        print(f"Comparacao temporal de {ano}", flush=True)
        treino, validacao = criar_particoes_temporais(base, ano)
        respostas.append(validacao[COLUNA_TARGET].to_numpy())
        anos.append(np.full(len(validacao), ano))
        for nome, colunas_numericas in colunas_por_modelo.items():
            print(f"  {nome}", flush=True)
            colunas_modelo = colunas_numericas + COLUNAS_CATEGORICAS
            modelo = criar_modelo(CONFIGURACAO, colunas_numericas)
            modelo.fit(treino[colunas_modelo], treino[COLUNA_TARGET])
            probabilidades[nome].append(
                modelo.predict_proba(validacao[colunas_modelo])[:, 1]
            )

    y_real = pd.Series(np.concatenate(respostas))
    anos_agrupados = np.concatenate(anos)
    resultados: dict[str, dict[str, object]] = {}
    for nome, blocos in probabilidades.items():
        probs = np.concatenate(blocos)
        limiar = escolher_limiar_recall_minimo(
            y_real, probs, RECALL_MINIMO
        )
        metricas_gerais = complementar_metricas(
            calcular_metricas(y_real, probs, limiar)
        )
        metricas_anuais = {}
        for ano in ANOS_VALIDACAO:
            mascara = anos_agrupados == ano
            metricas_anuais[str(ano)] = complementar_metricas(
                calcular_metricas(y_real[mascara], probs[mascara], limiar)
            )
        resultados[nome] = {
            "limiar": round(limiar, 6),
            "metricas_agrupadas": metricas_gerais,
            "metricas_por_ano": metricas_anuais,
        }

    fp_v2 = int(
        resultados["populacional_v2"]["metricas_agrupadas"][
            "matriz_confusao"
        ]["fp"]
    )
    fp_longo = int(
        resultados["populacional_com_historico_longo"]["metricas_agrupadas"][
            "matriz_confusao"
        ]["fp"]
    )
    saida = {
        "etapa": "Comparacao da versao 2 com variaveis de 24 e 36 meses.",
        "tipo_modelo": "classificacao_binaria",
        "target_alterado": False,
        "teste_final_usado": False,
        "configuracao": CONFIGURACAO,
        "recall_minimo_agrupado": RECALL_MINIMO,
        "variaveis_adicionais": colunas_longo_prazo,
        "reducao_falsos_positivos": fp_v2 - fp_longo,
        "resultados": resultados,
    }
    RELATORIO_JSON.parent.mkdir(parents=True, exist_ok=True)
    RELATORIO_JSON.write_text(
        json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with RELATORIO_TEXTO.open("w", encoding="utf-8") as relatorio:
        relatorio.write("Comparacao com historico de 24 e 36 meses\n\n")
        relatorio.write("Classificacao binaria e target inalterado.\n")
        relatorio.write("Teste final nao utilizado.\n")
        relatorio.write(
            "Criterio: menor FP com recall agrupado minimo de 70%.\n\n"
        )
        for nome, resultado in resultados.items():
            metricas = resultado["metricas_agrupadas"]
            matriz = metricas["matriz_confusao"]
            relatorio.write(
                f"{nome}: limiar={resultado['limiar']}, "
                f"AP={metricas['average_precision']}, "
                f"precisao={metricas['precision']}, "
                f"recall={metricas['recall']}, F2={metricas['f2']}, "
                f"FP={matriz['fp']}, FN={matriz['fn']}\n"
            )
            for ano, metricas_ano in resultado["metricas_por_ano"].items():
                matriz_ano = metricas_ano["matriz_confusao"]
                relatorio.write(
                    f"  - {ano}: precisao={metricas_ano['precision']}, "
                    f"recall={metricas_ano['recall']}, FP={matriz_ano['fp']}, "
                    f"FN={matriz_ano['fn']}\n"
                )
            relatorio.write("\n")
        relatorio.write(
            f"Reducao de falsos positivos contra a v2: {fp_v2 - fp_longo}.\n"
        )

    print(f"Comparacao salva em: {RELATORIO_TEXTO} e {RELATORIO_JSON}")


if __name__ == "__main__":
    main()
