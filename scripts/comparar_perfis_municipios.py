from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    from scripts.analisar_limiares_modelo import (
        complementar_metricas,
        escolher_limiar_recall_minimo,
    )
    from scripts.otimizar_gradient_boosting import CONFIGURACOES, criar_modelo
    from scripts.treinar_modelo_risco import (
        ANOS_VALIDACAO,
        COLUNAS_CATEGORICAS,
        COLUNA_TARGET,
        LIMITE_VALIDACAO,
        SEMENTE,
        calcular_metricas,
        criar_particoes_temporais,
        criar_variaveis_historicas,
        escolher_limiar,
    )
except ModuleNotFoundError:
    from analisar_limiares_modelo import (  # type: ignore[no-redef]
        complementar_metricas,
        escolher_limiar_recall_minimo,
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
        criar_variaveis_historicas,
        escolher_limiar,
    )


ARQUIVO_ENTRADA = Path("data/processed/violencia_grave_sul_com_target.csv")
RELATORIO_TEXTO = Path("reports/resumo_comparacao_perfis_municipios.txt")
RELATORIO_JSON = Path("reports/metricas_comparacao_perfis_municipios.json")
NOME_CONFIGURACAO = "conservador_15"
QUANTIDADE_PERFIS = 5
CONFIGURACAO = next(
    configuracao
    for configuracao in CONFIGURACOES
    if configuracao["nome"] == NOME_CONFIGURACAO
)

COLUNAS_DESCRITORAS_PERFIL = [
    "perfil_media_24m",
    "perfil_desvio_24m",
    "perfil_proporcao_zeros_24m",
    "perfil_amplitude_24m",
    "perfil_razao_media_6m_24m",
]


def criar_descritores_perfil(base: pd.DataFrame) -> pd.DataFrame:
    """Resume o comportamento conhecido do municipio ate a data da consulta."""
    base = base.copy().sort_values(["uf", "municipio", "data_referencia"])
    grupo = base.groupby(["uf", "municipio"], sort=False)["total_violencia_grave"]
    base["perfil_media_24m"] = grupo.transform(
        lambda valores: valores.rolling(24, min_periods=12).mean()
    )
    base["perfil_desvio_24m"] = grupo.transform(
        lambda valores: valores.rolling(24, min_periods=12).std()
    )
    base["perfil_proporcao_zeros_24m"] = grupo.transform(
        lambda valores: valores.rolling(24, min_periods=12)
        .apply(lambda janela: float(np.mean(janela == 0)), raw=True)
    )
    maximo_24m = grupo.transform(
        lambda valores: valores.rolling(24, min_periods=12).max()
    )
    minimo_24m = grupo.transform(
        lambda valores: valores.rolling(24, min_periods=12).min()
    )
    base["perfil_amplitude_24m"] = maximo_24m - minimo_24m
    media_6m = grupo.transform(
        lambda valores: valores.rolling(6, min_periods=6).mean()
    )
    base["perfil_razao_media_6m_24m"] = (
        (media_6m + 0.25) / (base["perfil_media_24m"] + 0.25)
    )
    return base


def atribuir_perfis(
    treino: pd.DataFrame, validacao: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, float | int]]]:
    imputador = SimpleImputer(strategy="median")
    escala = StandardScaler()
    treino_imputado = imputador.fit_transform(treino[COLUNAS_DESCRITORAS_PERFIL])
    validacao_imputada = imputador.transform(validacao[COLUNAS_DESCRITORAS_PERFIL])
    treino_escalado = escala.fit_transform(treino_imputado)
    validacao_escalada = escala.transform(validacao_imputada)

    agrupador = KMeans(
        n_clusters=QUANTIDADE_PERFIS,
        n_init=10,
        random_state=SEMENTE,
    )
    grupos_treino = agrupador.fit_predict(treino_escalado)
    grupos_validacao = agrupador.predict(validacao_escalada)

    # Ordenar pela media historica torna os identificadores mais interpretaveis.
    medias_por_grupo = {
        grupo: float(
            treino.loc[grupos_treino == grupo, "perfil_media_24m"].median()
        )
        for grupo in range(QUANTIDADE_PERFIS)
    }
    ordem = sorted(medias_por_grupo, key=medias_por_grupo.get)
    mapa = {grupo: posicao + 1 for posicao, grupo in enumerate(ordem)}

    treino = treino.copy()
    validacao = validacao.copy()
    treino["perfil_municipio"] = [f"perfil_{mapa[grupo]}" for grupo in grupos_treino]
    validacao["perfil_municipio"] = [
        f"perfil_{mapa[grupo]}" for grupo in grupos_validacao
    ]

    centroides_originais = escala.inverse_transform(agrupador.cluster_centers_)
    centroides = []
    for grupo, centroide in enumerate(centroides_originais):
        valores = {
            coluna: round(float(valor), 4)
            for coluna, valor in zip(COLUNAS_DESCRITORAS_PERFIL, centroide)
        }
        centroides.append(
            {
                "perfil": mapa[grupo],
                "quantidade_treino": int(np.sum(grupos_treino == grupo)),
                **valores,
            }
        )
    centroides.sort(key=lambda item: int(item["perfil"]))
    return treino, validacao, centroides


def criar_modelo_com_perfil(colunas_numericas: list[str]) -> Pipeline:
    preprocessador = ColumnTransformer(
        [
            (
                "numericas",
                Pipeline(
                    [
                        ("imputacao", SimpleImputer(strategy="median")),
                        ("escala", StandardScaler()),
                    ]
                ),
                colunas_numericas,
            ),
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                COLUNAS_CATEGORICAS + ["perfil_municipio"],
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessamento", preprocessador),
            (
                "modelo",
                HistGradientBoostingClassifier(
                    learning_rate=float(CONFIGURACAO["learning_rate"]),
                    max_iter=int(CONFIGURACAO["max_iter"]),
                    max_leaf_nodes=int(CONFIGURACAO["max_leaf_nodes"]),
                    min_samples_leaf=int(CONFIGURACAO["min_samples_leaf"]),
                    l2_regularization=float(CONFIGURACAO["l2_regularization"]),
                    early_stopping=False,
                    random_state=SEMENTE,
                ),
            ),
        ]
    )


def main() -> None:
    base_original = pd.read_csv(ARQUIVO_ENTRADA)
    base, colunas_numericas_originais = criar_variaveis_historicas(base_original)
    base = criar_descritores_perfil(base)
    base = base[base["data_referencia"] <= LIMITE_VALIDACAO]
    colunas_modelo_original = colunas_numericas_originais + COLUNAS_CATEGORICAS
    colunas_numericas_perfil = (
        colunas_numericas_originais + COLUNAS_DESCRITORAS_PERFIL
    )
    colunas_modelo_perfil = (
        colunas_numericas_perfil + COLUNAS_CATEGORICAS + ["perfil_municipio"]
    )

    respostas: list[np.ndarray] = []
    probabilidades_por_modelo: dict[str, list[np.ndarray]] = {
        "modelo_original": [],
        "modelo_com_perfis": [],
    }
    centroides_por_ano: dict[str, list[dict[str, float | int]]] = {}

    for ano_validacao in ANOS_VALIDACAO:
        print(f"Comparacao temporal de {ano_validacao}", flush=True)
        treino, validacao = criar_particoes_temporais(base, ano_validacao)
        treino_perfil, validacao_perfil, centroides = atribuir_perfis(
            treino, validacao
        )
        centroides_por_ano[str(ano_validacao)] = centroides
        respostas.append(validacao[COLUNA_TARGET].to_numpy())

        modelo_original = criar_modelo(CONFIGURACAO, colunas_numericas_originais)
        modelo_original.fit(
            treino[colunas_modelo_original], treino[COLUNA_TARGET]
        )
        probabilidades_por_modelo["modelo_original"].append(
            modelo_original.predict_proba(validacao[colunas_modelo_original])[:, 1]
        )

        modelo_perfil = criar_modelo_com_perfil(colunas_numericas_perfil)
        modelo_perfil.fit(
            treino_perfil[colunas_modelo_perfil], treino_perfil[COLUNA_TARGET]
        )
        probabilidades_por_modelo["modelo_com_perfis"].append(
            modelo_perfil.predict_proba(validacao_perfil[colunas_modelo_perfil])[:, 1]
        )

    y_real = pd.Series(np.concatenate(respostas))
    resultados: dict[str, dict[str, object]] = {}
    for nome, blocos_probabilidades in probabilidades_por_modelo.items():
        probabilidades = np.concatenate(blocos_probabilidades)
        limiar_f2 = escolher_limiar(y_real, probabilidades)
        limiar_recall_70 = escolher_limiar_recall_minimo(
            y_real, probabilidades, 0.70
        )
        resultados[nome] = {
            "limiar_otimizado_f2": complementar_metricas(
                calcular_metricas(y_real, probabilidades, limiar_f2)
            ),
            "limiar_com_recall_minimo_70": complementar_metricas(
                calcular_metricas(y_real, probabilidades, limiar_recall_70)
            ),
        }

    saida = {
        "etapa": "Comparacao do modelo original com cinco perfis municipais.",
        "dados": "Somente validacoes temporais de 2020 a 2023.",
        "teste_final_usado": False,
        "descritores_perfil": COLUNAS_DESCRITORAS_PERFIL,
        "resultados": resultados,
        "centroides_por_ano": centroides_por_ano,
    }
    RELATORIO_TEXTO.parent.mkdir(parents=True, exist_ok=True)
    RELATORIO_JSON.write_text(
        json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with RELATORIO_TEXTO.open("w", encoding="utf-8") as relatorio:
        relatorio.write("Comparacao de perfis municipais\n\n")
        relatorio.write(
            "Cinco perfis foram aprendidos separadamente em cada treino temporal.\n"
        )
        relatorio.write("O teste final nao foi usado neste experimento.\n\n")
        for nome, cenarios in resultados.items():
            relatorio.write(f"{nome}:\n")
            for cenario, metricas in cenarios.items():
                matriz = metricas["matriz_confusao"]
                relatorio.write(
                    f"- {cenario}: AP={metricas['average_precision']}, "
                    f"limiar={metricas['limiar']}, "
                    f"precisao={metricas['precision']}, "
                    f"recall={metricas['recall']}, F2={metricas['f2']}, "
                    f"FP={matriz['fp']}, FN={matriz['fn']}, "
                    "falsos_por_acerto="
                    f"{metricas['falsos_positivos_por_verdadeiro_positivo']}\n"
                )
            relatorio.write("\n")

    print(f"Comparacao salva em: {RELATORIO_TEXTO} e {RELATORIO_JSON}")


if __name__ == "__main__":
    main()
