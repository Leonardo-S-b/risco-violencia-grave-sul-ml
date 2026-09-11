from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from scripts.versoes.maximo_movel_6m.analisar_limiares_modelo import (
        complementar_metricas,
        escolher_limiar_recall_minimo,
    )
    from scripts.experimentos.comparar_perfis_municipios import criar_modelo_com_perfil
    from scripts.versoes.maximo_movel_6m.otimizar_gradient_boosting import CONFIGURACOES, criar_modelo
    from scripts.versoes.maximo_movel_6m.treinar_modelo_risco import (
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
    from comparar_perfis_municipios import criar_modelo_com_perfil
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


ARQUIVO_ENTRADA = Path("data/processed/versoes/maximo_movel_6m/violencia_grave_sul_com_target.csv")
RELATORIO_TEXTO = Path("reports/experimentos/resumo_comparacao_perfis_anuais_municipios.txt")
RELATORIO_JSON = Path("reports/experimentos/metricas_comparacao_perfis_anuais_municipios.json")
NOME_CONFIGURACAO = "conservador_15"
QUANTIDADE_PERFIS = 5
CONFIGURACAO = next(
    configuracao
    for configuracao in CONFIGURACOES
    if configuracao["nome"] == NOME_CONFIGURACAO
)

COLUNAS_DESCRITORAS = [
    "municipio_media_24m_anteriores",
    "municipio_desvio_24m_anteriores",
    "municipio_proporcao_zeros_24m_anteriores",
    "municipio_maximo_24m_anteriores",
    "municipio_razao_ultimos_12m_12m_anteriores",
]
COLUNAS_LOG = [
    "municipio_media_24m_anteriores",
    "municipio_desvio_24m_anteriores",
    "municipio_maximo_24m_anteriores",
]


def criar_resumos_anuais_municipios(base: pd.DataFrame) -> pd.DataFrame:
    """Cria um retrato por municipio/ano usando somente os 24 meses anteriores."""
    base = base.copy()
    base["data_referencia"] = pd.to_datetime(base["data_referencia"])
    base = base.sort_values(["uf", "municipio", "data_referencia"])
    anos = sorted(base["data_referencia"].dt.year.unique())
    resumos: list[pd.DataFrame] = []

    for ano in anos:
        inicio_ano = pd.Timestamp(year=int(ano), month=1, day=1)
        inicio_janela = inicio_ano - pd.DateOffset(months=24)
        inicio_ultimos_12m = inicio_ano - pd.DateOffset(months=12)
        historico = base[
            (base["data_referencia"] >= inicio_janela)
            & (base["data_referencia"] < inicio_ano)
        ].copy()
        if historico.empty:
            continue

        chaves = ["uf", "municipio"]
        resumo = (
            historico.groupby(chaves)["total_violencia_grave"]
            .agg(
                municipio_media_24m_anteriores="mean",
                municipio_desvio_24m_anteriores="std",
                municipio_maximo_24m_anteriores="max",
            )
            .reset_index()
        )
        proporcao_zeros = (
            historico.assign(
                mes_zerado=(historico["total_violencia_grave"] == 0).astype(float)
            )
            .groupby(chaves)["mes_zerado"]
            .mean()
            .rename("municipio_proporcao_zeros_24m_anteriores")
            .reset_index()
        )
        media_recente = (
            historico[historico["data_referencia"] >= inicio_ultimos_12m]
            .groupby(chaves)["total_violencia_grave"]
            .mean()
            .rename("media_ultimos_12m")
            .reset_index()
        )
        media_anterior = (
            historico[historico["data_referencia"] < inicio_ultimos_12m]
            .groupby(chaves)["total_violencia_grave"]
            .mean()
            .rename("media_12m_anteriores")
            .reset_index()
        )
        resumo = resumo.merge(proporcao_zeros, on=chaves, how="left")
        resumo = resumo.merge(media_recente, on=chaves, how="left")
        resumo = resumo.merge(media_anterior, on=chaves, how="left")
        resumo["municipio_razao_ultimos_12m_12m_anteriores"] = (
            (resumo["media_ultimos_12m"] + 0.25)
            / (resumo["media_12m_anteriores"] + 0.25)
        )
        resumo["ano"] = int(ano)
        resumos.append(resumo[chaves + ["ano"] + COLUNAS_DESCRITORAS])

    if not resumos:
        return pd.DataFrame(columns=["uf", "municipio", "ano"] + COLUNAS_DESCRITORAS)
    return pd.concat(resumos, ignore_index=True)


def preparar_dados_agrupamento(dados: pd.DataFrame) -> pd.DataFrame:
    preparados = dados[COLUNAS_DESCRITORAS].copy()
    preparados[COLUNAS_LOG] = np.log1p(preparados[COLUNAS_LOG].clip(lower=0))
    preparados["municipio_razao_ultimos_12m_12m_anteriores"] = np.log1p(
        preparados["municipio_razao_ultimos_12m_12m_anteriores"].clip(lower=0)
    )
    return preparados


def atribuir_perfis_anuais(
    treino: pd.DataFrame, validacao: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    chaves = ["uf", "municipio", "ano"]
    snapshots_treino = treino[chaves + COLUNAS_DESCRITORAS].drop_duplicates(chaves)
    dados_agrupamento = preparar_dados_agrupamento(snapshots_treino)
    imputador = SimpleImputer(strategy="median")
    escala = StandardScaler()
    treino_imputado = imputador.fit_transform(dados_agrupamento)
    treino_escalado = escala.fit_transform(treino_imputado)
    agrupador = KMeans(
        n_clusters=QUANTIDADE_PERFIS,
        n_init=20,
        random_state=SEMENTE,
    )
    grupos_snapshots = agrupador.fit_predict(treino_escalado)

    medianas_volume = {
        grupo: float(
            snapshots_treino.loc[
                grupos_snapshots == grupo, "municipio_media_24m_anteriores"
            ].median()
        )
        for grupo in range(QUANTIDADE_PERFIS)
    }
    ordem = sorted(medianas_volume, key=medianas_volume.get)
    mapa_perfis = {grupo: posicao + 1 for posicao, grupo in enumerate(ordem)}

    def prever_perfil(dados: pd.DataFrame) -> pd.Series:
        preparados = preparar_dados_agrupamento(dados)
        transformados = escala.transform(imputador.transform(preparados))
        grupos = agrupador.predict(transformados)
        return pd.Series(
            [f"perfil_{mapa_perfis[grupo]}" for grupo in grupos],
            index=dados.index,
        )

    treino = treino.copy()
    validacao = validacao.copy()
    treino["perfil_municipio"] = prever_perfil(treino)
    validacao["perfil_municipio"] = prever_perfil(validacao)

    caracteristicas: list[dict[str, object]] = []
    snapshots_treino = snapshots_treino.copy()
    snapshots_treino["perfil_municipio"] = [
        f"perfil_{mapa_perfis[grupo]}" for grupo in grupos_snapshots
    ]
    for perfil, recorte in snapshots_treino.groupby("perfil_municipio", sort=True):
        caracteristicas.append(
            {
                "perfil": str(perfil),
                "snapshots_municipais": len(recorte),
                "mediana_volume_mensal": round(
                    float(recorte["municipio_media_24m_anteriores"].median()), 4
                ),
                "mediana_proporcao_zeros": round(
                    float(
                        recorte[
                            "municipio_proporcao_zeros_24m_anteriores"
                        ].median()
                    ),
                    4,
                ),
                "mediana_tendencia": round(
                    float(
                        recorte[
                            "municipio_razao_ultimos_12m_12m_anteriores"
                        ].median()
                    ),
                    4,
                ),
            }
        )
    return treino, validacao, caracteristicas


def avaliar_limiares_por_perfil(
    y_real: pd.Series,
    probabilidades: np.ndarray,
    perfis: pd.Series,
    criterio: str,
) -> dict[str, object]:
    limiares: dict[str, float] = {}
    metricas_por_perfil: dict[str, dict[str, object]] = {}
    for perfil in sorted(perfis.unique()):
        mascara = perfis == perfil
        respostas_perfil = y_real[mascara].reset_index(drop=True)
        probabilidades_perfil = probabilidades[mascara.to_numpy()]
        if criterio == "f2":
            limiar = escolher_limiar(respostas_perfil, probabilidades_perfil)
        elif criterio == "recall_minimo_70":
            limiar = escolher_limiar_recall_minimo(
                respostas_perfil, probabilidades_perfil, 0.70
            )
        else:
            raise ValueError(f"Criterio de limiar desconhecido: {criterio}")
        limiares[str(perfil)] = limiar
        metricas_por_perfil[str(perfil)] = complementar_metricas(
            calcular_metricas(respostas_perfil, probabilidades_perfil, limiar)
        )

    limites_por_linha = perfis.map(limiares).to_numpy(dtype=float)
    probabilidades_normalizadas = probabilidades / np.maximum(
        limites_por_linha, np.finfo(float).eps
    )
    metricas_gerais = complementar_metricas(
        calcular_metricas(y_real, probabilidades_normalizadas, 1.0)
    )
    # AP e ROC AUC medem o ranking original e independem do limite operacional.
    metricas_gerais["average_precision"] = round(
        float(average_precision_score(y_real, probabilidades)), 6
    )
    metricas_gerais["roc_auc"] = round(
        float(roc_auc_score(y_real, probabilidades)), 6
    )
    metricas_gerais["limiar"] = "especifico_por_perfil"
    metricas_gerais["limiares_por_perfil"] = {
        perfil: round(limiar, 6) for perfil, limiar in limiares.items()
    }
    metricas_gerais["metricas_por_perfil"] = metricas_por_perfil
    return metricas_gerais


def main() -> None:
    base_original = pd.read_csv(ARQUIVO_ENTRADA)
    resumos_anuais = criar_resumos_anuais_municipios(base_original)
    base, colunas_numericas_originais = criar_variaveis_historicas(base_original)
    base = base.merge(
        resumos_anuais,
        on=["uf", "municipio", "ano"],
        how="left",
        validate="many_to_one",
    )
    base = base[base["data_referencia"] <= LIMITE_VALIDACAO]
    colunas_modelo_original = colunas_numericas_originais + COLUNAS_CATEGORICAS
    colunas_numericas_perfil = colunas_numericas_originais + COLUNAS_DESCRITORAS
    colunas_modelo_perfil = (
        colunas_numericas_perfil + COLUNAS_CATEGORICAS + ["perfil_municipio"]
    )

    respostas: list[np.ndarray] = []
    probabilidades_por_modelo: dict[str, list[np.ndarray]] = {
        "modelo_original": [],
        "modelo_com_perfis_anuais": [],
    }
    perfis_validacao: list[np.ndarray] = []
    caracteristicas_por_ano: dict[str, list[dict[str, object]]] = {}

    for ano_validacao in ANOS_VALIDACAO:
        print(f"Comparacao temporal de {ano_validacao}", flush=True)
        treino, validacao = criar_particoes_temporais(base, ano_validacao)
        treino_perfil, validacao_perfil, caracteristicas = atribuir_perfis_anuais(
            treino, validacao
        )
        caracteristicas_por_ano[str(ano_validacao)] = caracteristicas
        respostas.append(validacao[COLUNA_TARGET].to_numpy())
        perfis_validacao.append(validacao_perfil["perfil_municipio"].to_numpy())

        modelo_original = criar_modelo(CONFIGURACAO, colunas_numericas_originais)
        modelo_original.fit(treino[colunas_modelo_original], treino[COLUNA_TARGET])
        probabilidades_por_modelo["modelo_original"].append(
            modelo_original.predict_proba(validacao[colunas_modelo_original])[:, 1]
        )

        modelo_perfil = criar_modelo_com_perfil(colunas_numericas_perfil)
        modelo_perfil.fit(
            treino_perfil[colunas_modelo_perfil], treino_perfil[COLUNA_TARGET]
        )
        probabilidades_por_modelo["modelo_com_perfis_anuais"].append(
            modelo_perfil.predict_proba(validacao_perfil[colunas_modelo_perfil])[:, 1]
        )

    y_real = pd.Series(np.concatenate(respostas))
    perfis_fora_treino = pd.Series(np.concatenate(perfis_validacao))
    resultados: dict[str, dict[str, object]] = {}
    for nome, blocos in probabilidades_por_modelo.items():
        probabilidades = np.concatenate(blocos)
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
        if nome == "modelo_com_perfis_anuais":
            resultados[nome]["limiares_por_perfil_f2"] = avaliar_limiares_por_perfil(
                y_real, probabilidades, perfis_fora_treino, "f2"
            )
            resultados[nome][
                "limiares_por_perfil_recall_minimo_70"
            ] = avaliar_limiares_por_perfil(
                y_real,
                probabilidades,
                perfis_fora_treino,
                "recall_minimo_70",
            )

    saida = {
        "etapa": "Perfis anuais por municipio com historico anterior ao ano.",
        "dados": "Somente validacoes temporais de 2020 a 2023.",
        "teste_final_usado": False,
        "resultados": resultados,
        "caracteristicas_perfis_por_validacao": caracteristicas_por_ano,
    }
    RELATORIO_TEXTO.parent.mkdir(parents=True, exist_ok=True)
    RELATORIO_JSON.write_text(
        json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with RELATORIO_TEXTO.open("w", encoding="utf-8") as relatorio:
        relatorio.write("Comparacao de perfis anuais por municipio\n\n")
        relatorio.write(
            "Cada municipio recebe um perfil fixo no ano, calculado com os "
            "24 meses anteriores.\n"
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
                if "limiares_por_perfil" in metricas:
                    relatorio.write(
                        f"  limites: {metricas['limiares_por_perfil']}\n"
                    )
            relatorio.write("\n")

    print(f"Comparacao salva em: {RELATORIO_TEXTO} e {RELATORIO_JSON}")


if __name__ == "__main__":
    main()
