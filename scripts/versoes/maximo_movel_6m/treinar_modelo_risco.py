from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.tree import DecisionTreeClassifier


ARQUIVO_ENTRADA = Path("data/processed/versoes/maximo_movel_6m/violencia_grave_sul_com_target.csv")
RELATORIO_TEXTO = Path("reports/versoes/maximo_movel_6m/resumo_treinamento_modelo.txt")
RELATORIO_JSON = Path("reports/versoes/maximo_movel_6m/metricas_treinamento_modelo.json")

COLUNA_TARGET = "alerta_violencia_grave_proximos_6m"
COLUNAS_EVENTOS = [
    "homicidio_doloso",
    "latrocinio",
    "tentativa_homicidio",
    "lesao_corporal_seguida_morte",
    "feminicidio",
]
COLUNAS_CATEGORICAS = ["uf"]
COLUNA_MAXIMO = "maximo_12m_ate_data_consulta"
LIMITE_VALIDACAO = pd.Timestamp("2023-12-01")
ANOS_VALIDACAO = (2020, 2021, 2022, 2023)
SEMENTE = 42


def criar_variaveis_historicas(base: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Cria variaveis disponiveis na data da consulta, antes dos seis meses alvo."""
    base = base.copy()
    base["data_referencia"] = pd.to_datetime(base["data_referencia"])
    base["data_fim_horizonte"] = pd.to_datetime(base["data_fim_horizonte"])
    base = base.sort_values(["uf", "municipio", "data_referencia"])
    grupo = base.groupby(["uf", "municipio"], sort=False)

    colunas_numericas: list[str] = ["total_violencia_grave"]
    for atraso in (1, 2, 3, 6, 12):
        coluna = f"total_lag_{atraso}m"
        base[coluna] = grupo["total_violencia_grave"].shift(atraso)
        colunas_numericas.append(coluna)

    for janela in (3, 6, 12):
        coluna_media = f"media_movel_{janela}m"
        base[coluna_media] = grupo["total_violencia_grave"].transform(
            lambda valores: valores.rolling(janela, min_periods=janela).mean()
        )
        colunas_numericas.append(coluna_media)

    for janela in (6, 12):
        coluna_desvio = f"desvio_movel_{janela}m"
        base[coluna_desvio] = grupo["total_violencia_grave"].transform(
            lambda valores: valores.rolling(janela, min_periods=janela).std()
        )
        colunas_numericas.append(coluna_desvio)

    for evento in COLUNAS_EVENTOS:
        coluna_atraso = f"{evento}_lag_1m"
        coluna_soma = f"{evento}_soma_12m"
        base[coluna_atraso] = grupo[evento].shift(1)
        base[coluna_soma] = grupo[evento].transform(
            lambda valores: valores.rolling(12, min_periods=12).sum()
        )
        colunas_numericas.extend([evento, coluna_atraso, coluna_soma])

    maximo = base[COLUNA_MAXIMO]
    base["distancia_total_atual_ao_maximo"] = (
        maximo - base["total_violencia_grave"]
    )
    base["razao_total_atual_pelo_maximo"] = np.where(
        maximo > 0, base["total_violencia_grave"] / maximo, 0.0
    )
    base["mes_seno"] = np.sin(2 * np.pi * base["mes"] / 12)
    base["mes_cosseno"] = np.cos(2 * np.pi * base["mes"] / 12)
    base["indice_tempo"] = (base["ano"] - base["ano"].min()) * 12 + base["mes"]
    colunas_numericas.extend(
        [
            COLUNA_MAXIMO,
            "distancia_total_atual_ao_maximo",
            "razao_total_atual_pelo_maximo",
            "mes_seno",
            "mes_cosseno",
            "indice_tempo",
        ]
    )

    base = base.dropna(subset=[COLUNA_TARGET]).copy()
    base[COLUNA_TARGET] = base[COLUNA_TARGET].astype(int)
    return base, colunas_numericas


def criar_particoes_temporais(
    base: pd.DataFrame, ano_validacao: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa sem usar no treino targets que terminam dentro da validacao."""
    inicio_validacao = pd.Timestamp(year=ano_validacao, month=1, day=1)
    fim_validacao = pd.Timestamp(year=ano_validacao, month=12, day=1)
    treino = base[base["data_fim_horizonte"] < inicio_validacao]
    validacao = base[
        (base["data_referencia"] >= inicio_validacao)
        & (base["data_referencia"] <= fim_validacao)
    ]
    return treino, validacao


def criar_preprocessador(colunas_numericas: list[str]) -> ColumnTransformer:
    pipeline_numerico = Pipeline(
        [
            ("imputacao", SimpleImputer(strategy="median")),
            ("escala", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("numericas", pipeline_numerico, colunas_numericas),
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                COLUNAS_CATEGORICAS,
            ),
        ]
    )


def criar_modelos(colunas_numericas: list[str]) -> dict[str, Pipeline]:
    def pipeline(modelo: object) -> Pipeline:
        return Pipeline(
            [
                ("preprocessamento", criar_preprocessador(colunas_numericas)),
                ("modelo", modelo),
            ]
        )

    return {
        "baseline": pipeline(DummyClassifier(strategy="prior")),


        #melhor modelo ate o momento, mas nao e o mais rapido, deve ter alguma coisa melhor
        "regressao_logistica": pipeline(
            LogisticRegression(max_iter=2_000, random_state=SEMENTE)
        ),
        # não usarei por enquanto não gostei dos resultados
        # "arvore_decisao": pipeline(
        #     DecisionTreeClassifier(
        #         max_depth=8,
        #         min_samples_leaf=25,
        #         random_state=SEMENTE,
        #     )
        # ),

        #acho muito caro pode ate ser mais rápido mas perde performance
        "floresta_aleatoria": pipeline(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=14,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=SEMENTE,
            )
        ),
        "extra_trees": pipeline(
            ExtraTreesClassifier(
                n_estimators=300,
                max_depth=14,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=SEMENTE,
            )
        ),
        "gradient_boosting": pipeline(
            HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=150,
                max_leaf_nodes=31,
                l2_regularization=0.1,
                random_state=SEMENTE,
            )
        ),
        #Deu muuito ruim, nao sei se vale a pena tentar otimizar
        # "knn": pipeline(
        #     KNeighborsClassifier(
        #         n_neighbors=51,
        #         weights="distance",
        #         n_jobs=-1,
        #     )
        # ),
    }


def escolher_limiar(y_real: pd.Series, probabilidades: np.ndarray) -> float:
    """Prioriza recall sem ignorar precisao, usando F-beta com beta igual a 2."""
    precisoes, recalls, limiares = precision_recall_curve(y_real, probabilidades)
    if len(limiares) == 0:
        return 0.5

    precisoes = precisoes[:-1]
    recalls = recalls[:-1]
    denominador = 4 * precisoes + recalls
    valores_f2 = np.divide(
        5 * precisoes * recalls,
        denominador,
        out=np.zeros_like(denominador),
        where=denominador > 0,
    )
    return float(limiares[int(np.argmax(valores_f2))])


def calcular_metricas(
    y_real: pd.Series, probabilidades: np.ndarray, limiar: float
) -> dict[str, object]:
    previsoes = (probabilidades >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_real, previsoes, labels=[0, 1]).ravel()
    return {
        "limiar": round(limiar, 4),
        "acuracia": round(float(accuracy_score(y_real, previsoes)), 6),
        "roc_auc": round(float(roc_auc_score(y_real, probabilidades)), 6),
        "average_precision": round(
            float(average_precision_score(y_real, probabilidades)), 6
        ),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(y_real, previsoes)), 6
        ),
        "precision": round(float(precision_score(y_real, previsoes, zero_division=0)), 6),
        "recall": round(float(recall_score(y_real, previsoes, zero_division=0)), 6),
        "f1": round(float(f1_score(y_real, previsoes, zero_division=0)), 6),
        "f2": round(float(fbeta_score(y_real, previsoes, beta=2, zero_division=0)), 6),
        "matriz_confusao": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "quantidade": int(len(y_real)),
        "positivos": int(y_real.sum()),
    }


def resumir_validacoes(
    resultados_por_ano: list[dict[str, object]],
) -> dict[str, object]:
    metricas_resumidas = [
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "f2",
    ]
    resumo: dict[str, object] = {}
    for metrica in metricas_resumidas:
        valores = [float(resultado[metrica]) for resultado in resultados_por_ano]
        resumo[f"{metrica}_media"] = round(float(np.mean(valores)), 6)
        resumo[f"{metrica}_desvio"] = round(float(np.std(valores)), 6)

    resumo["tempo_total_segundos"] = round(
        sum(float(resultado["tempo_segundos"]) for resultado in resultados_por_ano),
        3,
    )
    resumo["positivos_validacao"] = sum(
        int(resultado["positivos"]) for resultado in resultados_por_ano
    )
    return resumo


def main() -> None:
    base_original = pd.read_csv(ARQUIVO_ENTRADA)
    base, colunas_numericas = criar_variaveis_historicas(base_original)
    colunas_modelo = colunas_numericas + COLUNAS_CATEGORICAS

    # O periodo posterior a 2023 permanece bloqueado nesta comparacao.
    base_comparacao = base[base["data_referencia"] <= LIMITE_VALIDACAO]
    nomes_modelos = list(criar_modelos(colunas_numericas))
    resultados_validacao: dict[str, list[dict[str, object]]] = {
        nome: [] for nome in nomes_modelos
    }

    for ano_validacao in ANOS_VALIDACAO:
        treino, validacao = criar_particoes_temporais(
            base_comparacao, ano_validacao
        )
        if min(treino[COLUNA_TARGET].sum(), validacao[COLUNA_TARGET].sum()) == 0:
            raise ValueError(
                f"Treino e validacao de {ano_validacao} precisam conter positivos."
            )

        print(
            f"Validacao {ano_validacao}: apenas targets concluidos antes do ano",
            flush=True,
        )
        for nome, modelo in criar_modelos(colunas_numericas).items():
            print(f"  Treinando {nome}...", flush=True)
            inicio = time.perf_counter()
            modelo.fit(treino[colunas_modelo], treino[COLUNA_TARGET])
            probabilidades = modelo.predict_proba(validacao[colunas_modelo])[:, 1]
            limiar = 0.5 if nome == "baseline" else escolher_limiar(
                validacao[COLUNA_TARGET], probabilidades
            )
            metricas = calcular_metricas(
                validacao[COLUNA_TARGET], probabilidades, limiar
            )
            metricas["ano_validacao"] = ano_validacao
            metricas["ultima_data_fim_target_treino"] = str(
                treino["data_fim_horizonte"].max().date()
            )
            metricas["tempo_segundos"] = round(time.perf_counter() - inicio, 3)
            resultados_validacao[nome].append(metricas)

    resumos = {
        nome: resumir_validacoes(resultados)
        for nome, resultados in resultados_validacao.items()
    }

    ranking = sorted(
        (nome for nome in nomes_modelos if nome != "baseline"),
        key=lambda nome: resumos[nome]["average_precision_media"],
        reverse=True,
    )

    resultado = {
        "objetivo": "Prever alerta de superacao do maximo nos proximos 6 meses.",
        "etapa": "Validacao temporal progressiva sem balanceamento.",
        "criterio_ranking": "Media da average precision entre 2020 e 2023.",
        "ranking": ranking,
        "periodos": {
            "validacoes": [
                "treino ate 2019, validacao 2020",
                "treino ate 2020, validacao 2021",
                "treino ate 2021, validacao 2022",
                "treino ate 2022, validacao 2023",
            ],
            "teste_bloqueado": "consultas de 2024-07 a 2025-06",
        },
        "resumo_validacoes": resumos,
        "resultados_por_ano": resultados_validacao,
        "colunas_modelo": colunas_modelo,
    }

    RELATORIO_TEXTO.parent.mkdir(parents=True, exist_ok=True)
    RELATORIO_JSON.write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with RELATORIO_TEXTO.open("w", encoding="utf-8") as relatorio:
        relatorio.write("Validacao temporal progressiva - violencia grave\n\n")
        relatorio.write("Estrategia: dados originais, sem balanceamento.\n")
        relatorio.write("Validacoes anuais: 2020, 2021, 2022 e 2023.\n")
        relatorio.write("Em cada rodada, o treino termina no ano anterior.\n")
        relatorio.write("Teste bloqueado: consultas de 2024-07 a 2025-06\n\n")
        relatorio.write("Ranking pela media da average precision:\n")
        for posicao, nome in enumerate(ranking, start=1):
            metricas = resumos[nome]
            relatorio.write(
                f"{posicao}. {nome}: AP media={metricas['average_precision_media']}, "
                f"recall medio={metricas['recall_media']}, "
                f"precisao media={metricas['precision_media']}, "
                f"F2 medio={metricas['f2_media']}, "
                f"tempo total={metricas['tempo_total_segundos']}s\n"
            )
        relatorio.write("\nBaseline:\n")
        relatorio.write(f"{resumos['baseline']}\n")
        relatorio.write("\nAverage precision por ano:\n")
        for nome in [*ranking, "baseline"]:
            valores = ", ".join(
                f"{resultado['ano_validacao']}={resultado['average_precision']}"
                for resultado in resultados_validacao[nome]
            )
            relatorio.write(f"- {nome}: {valores}\n")
        relatorio.write(
            "\nInterpretacao: o resultado e um alerta para consulta preliminar e nao "
            "uma recomendacao de investimento.\n"
        )
        relatorio.write(
            "Atencao: o alerta futuro e desbalanceado; por isso average "
            "precision, recall, F2 e a matriz de confusao devem acompanhar a acuracia.\n"
        )

    print(f"Ranking na validacao: {ranking}")
    print("O conjunto de teste nao foi acessado.")
    print(f"Relatorios salvos em: {RELATORIO_TEXTO} e {RELATORIO_JSON}")


if __name__ == "__main__":
    main()
