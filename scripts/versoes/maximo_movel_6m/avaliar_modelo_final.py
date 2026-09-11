from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

try:
    from scripts.versoes.maximo_movel_6m.otimizar_gradient_boosting import CONFIGURACOES, criar_modelo
    from scripts.versoes.maximo_movel_6m.treinar_modelo_risco import (
        COLUNAS_CATEGORICAS,
        COLUNA_TARGET,
        calcular_metricas,
        criar_variaveis_historicas,
    )
except ModuleNotFoundError:
    from otimizar_gradient_boosting import CONFIGURACOES, criar_modelo
    from treinar_modelo_risco import (  # type: ignore[no-redef]
        COLUNAS_CATEGORICAS,
        COLUNA_TARGET,
        calcular_metricas,
        criar_variaveis_historicas,
    )


ARQUIVO_ENTRADA = Path("data/processed/versoes/maximo_movel_6m/violencia_grave_sul_com_target.csv")
ARQUIVO_PREVISOES = Path("data/processed/versoes/maximo_movel_6m/previsoes_teste_modelo_final.csv")
ARQUIVO_MODELO = Path("models/versoes/maximo_movel_6m/modelo_risco_violencia_grave.joblib")
RELATORIO_TEXTO = Path("reports/versoes/maximo_movel_6m/resumo_avaliacao_modelo_final.txt")
RELATORIO_JSON = Path("reports/versoes/maximo_movel_6m/metricas_avaliacao_modelo_final.json")

NOME_CONFIGURACAO = "conservador_15"
CONFIGURACAO = next(
    configuracao
    for configuracao in CONFIGURACOES
    if configuracao["nome"] == NOME_CONFIGURACAO
)
LIMIAR_FIXO = 0.108100
INICIO_TESTE = pd.Timestamp("2024-07-01")
FIM_TESTE = pd.Timestamp("2025-06-01")


def main() -> None:
    base_original = pd.read_csv(ARQUIVO_ENTRADA)
    base, colunas_numericas = criar_variaveis_historicas(base_original)
    colunas_modelo = colunas_numericas + COLUNAS_CATEGORICAS

    # A purga evita treinar com targets cujo horizonte invade o teste final.
    desenvolvimento = base[base["data_fim_horizonte"] < INICIO_TESTE].copy()
    teste = base[
        (base["data_referencia"] >= INICIO_TESTE)
        & (base["data_referencia"] <= FIM_TESTE)
    ].copy()

    if desenvolvimento.empty or teste.empty:
        raise ValueError("Os recortes de desenvolvimento e teste nao podem ser vazios.")
    if desenvolvimento["data_fim_horizonte"].max() >= teste["data_referencia"].min():
        raise ValueError("Foi detectada sobreposicao temporal entre treino e teste.")
    if teste["data_referencia"].min() != INICIO_TESTE:
        raise ValueError("O teste final nao comeca em 2024-07.")
    if teste["data_referencia"].max() != FIM_TESTE:
        raise ValueError("O teste final nao termina em 2025-06.")

    modelo = criar_modelo(CONFIGURACAO, colunas_numericas)
    modelo.fit(desenvolvimento[colunas_modelo], desenvolvimento[COLUNA_TARGET])
    probabilidades = modelo.predict_proba(teste[colunas_modelo])[:, 1]
    metricas_gerais = calcular_metricas(
        teste[COLUNA_TARGET], probabilidades, LIMIAR_FIXO
    )

    teste["probabilidade_alerta"] = probabilidades
    teste["alerta_previsto"] = (probabilidades >= LIMIAR_FIXO).astype(int)
    teste["periodo_teste"] = teste["data_referencia"].apply(
        lambda data: "2024-07 a 2024-12" if data.year == 2024 else "2025-01 a 2025-06"
    )

    metricas_por_periodo: dict[str, dict[str, object]] = {}
    metricas_por_uf: dict[str, dict[str, object]] = {}
    for periodo, recorte in teste.groupby("periodo_teste", sort=True):
        metricas_por_periodo[str(periodo)] = calcular_metricas(
            recorte[COLUNA_TARGET],
            recorte["probabilidade_alerta"].to_numpy(),
            LIMIAR_FIXO,
        )
    for uf, recorte in teste.groupby("uf", sort=True):
        metricas_por_uf[str(uf)] = calcular_metricas(
            recorte[COLUNA_TARGET],
            recorte["probabilidade_alerta"].to_numpy(),
            LIMIAR_FIXO,
        )

    colunas_previsoes = [
        "uf",
        "municipio",
        "data_referencia",
        "total_violencia_grave",
        "maximo_12m_ate_data_consulta",
        "maximo_real_proximos_6m",
        "data_fim_horizonte",
        COLUNA_TARGET,
        "probabilidade_alerta",
        "alerta_previsto",
    ]
    ARQUIVO_PREVISOES.parent.mkdir(parents=True, exist_ok=True)
    teste[colunas_previsoes].to_csv(
        ARQUIVO_PREVISOES, index=False, encoding="utf-8"
    )

    parametros = {
        chave: valor for chave, valor in CONFIGURACAO.items() if chave != "nome"
    }
    resultado = {
        "configuracao_final": {
            "algoritmo": "hist_gradient_boosting",
            "nome": NOME_CONFIGURACAO,
            "parametros": parametros,
            "balanceamento": "nenhum",
            "limiar": LIMIAR_FIXO,
        },
        "periodo_desenvolvimento_consultas": {
            "inicio": str(desenvolvimento["data_referencia"].min().date()),
            "fim": str(desenvolvimento["data_referencia"].max().date()),
            "ultimo_horizonte_target": str(
                desenvolvimento["data_fim_horizonte"].max().date()
            ),
        },
        "periodo_teste": "consultas de 2024-07 a 2025-06",
        "linhas_desenvolvimento": len(desenvolvimento),
        "linhas_teste": len(teste),
        "metricas_gerais": metricas_gerais,
        "metricas_por_periodo": metricas_por_periodo,
        "metricas_por_uf": metricas_por_uf,
    }
    RELATORIO_TEXTO.parent.mkdir(parents=True, exist_ok=True)
    RELATORIO_JSON.write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with RELATORIO_TEXTO.open("w", encoding="utf-8") as relatorio:
        relatorio.write("Avaliacao final - consulta preliminar de violencia grave\n\n")
        relatorio.write(
            f"Modelo: HistGradientBoosting {NOME_CONFIGURACAO}, sem balanceamento.\n"
        )
        relatorio.write(f"Parametros: {parametros}.\n")
        relatorio.write(f"Limiar fixado na validacao: {LIMIAR_FIXO:.6f}.\n")
        relatorio.write("Teste final: consultas de 2024-07 a 2025-06.\n")
        relatorio.write(
            "O treino usa apenas targets encerrados antes do inicio do teste.\n\n"
        )
        relatorio.write("Metricas gerais:\n")
        for nome, valor in metricas_gerais.items():
            relatorio.write(f"- {nome}: {valor}\n")
        relatorio.write("\nMetricas por periodo:\n")
        for periodo, metricas in metricas_por_periodo.items():
            relatorio.write(f"- {periodo}: {metricas}\n")
        relatorio.write("\nMetricas por UF:\n")
        for uf, metricas in metricas_por_uf.items():
            relatorio.write(f"- {uf}: {metricas}\n")
        relatorio.write(
            "\nUso: alerta de consulta preliminar; nao constitui recomendacao "
            "de investimento.\n"
        )

    ARQUIVO_MODELO.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "modelo": modelo,
            "limiar": LIMIAR_FIXO,
            "colunas_modelo": colunas_modelo,
            "configuracao": resultado["configuracao_final"],
            "treinado_com_consultas_ate": str(
                desenvolvimento["data_referencia"].max().date()
            ),
            "targets_de_treino_encerrados_ate": str(
                desenvolvimento["data_fim_horizonte"].max().date()
            ),
        },
        ARQUIVO_MODELO,
    )

    print(f"Metricas finais: {metricas_gerais}")
    print(f"Previsoes salvas em: {ARQUIVO_PREVISOES}")
    print(f"Modelo salvo em: {ARQUIVO_MODELO}")
    print(f"Relatorios salvos em: {RELATORIO_TEXTO} e {RELATORIO_JSON}")


if __name__ == "__main__":
    main()
