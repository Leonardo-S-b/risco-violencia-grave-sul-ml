from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from scripts.experimentos.comparar_variaveis_populacionais import (
    criar_variaveis_populacionais,
)
from scripts.versoes.maximo_movel_6m.treinar_modelo_risco import (
    COLUNAS_CATEGORICAS,
    COLUNA_TARGET,
    criar_variaveis_historicas,
)
from scripts.versoes.taxa_100k.criar_target_taxa_100k import (
    ARQUIVO_SAIDA,
    COLUNA_TARGET_TAXA,
    CORTE_TAXA_100K,
    MESES_COM_OCORRENCIA_MINIMOS,
)
from scripts.versoes.taxa_100k.otimizar_modelo_taxa_100k import (
    CONFIGURACOES_RF,
    criar_random_forest,
)


NOME_MODELO = "rf_folhas_10"
LIMIAR_ALERTA = 0.1313
ARQUIVO_METRICAS = Path(
    "reports/versoes/taxa_100k/metricas_otimizacao_modelo_taxa_100k.json"
)
PASTA_MODELO = Path("models/versoes/taxa_100k")
ARQUIVO_MODELO = PASTA_MODELO / "modelo_risco_taxa_100k.joblib"
ARQUIVO_METADATA = PASTA_MODELO / "metadata_modelo_risco_taxa_100k.json"
RELATORIO = Path("reports/versoes/taxa_100k/resumo_empacotamento_modelo.txt")


def main() -> None:
    metricas_otimizacao = json.loads(
        ARQUIVO_METRICAS.read_text(encoding="utf-8")
    )
    if metricas_otimizacao["melhor_modelo"] != NOME_MODELO:
        raise ValueError(
            "O modelo configurado nao coincide com o vencedor da otimizacao."
        )

    original = pd.read_csv(ARQUIVO_SAIDA)
    original[COLUNA_TARGET] = original[COLUNA_TARGET_TAXA]
    base, colunas_historicas = criar_variaveis_historicas(original)
    base, colunas_populacionais = criar_variaveis_populacionais(base)
    colunas_numericas = colunas_historicas + colunas_populacionais
    colunas_modelo = colunas_numericas + COLUNAS_CATEGORICAS
    configuracao = CONFIGURACOES_RF[NOME_MODELO]

    print(f"Treinando {NOME_MODELO} com {len(base)} observacoes...", flush=True)
    modelo = criar_random_forest(configuracao, colunas_numericas)
    modelo.fit(base[colunas_modelo], base[COLUNA_TARGET])

    metricas_validacao = metricas_otimizacao["resultados"][NOME_MODELO][
        "metricas_agrupadas"
    ]
    metadata = {
        "schema_artefato": "1.0",
        "gerado_em": datetime.now().astimezone().isoformat(timespec="seconds"),
        "algoritmo": "RandomForestClassifier",
        "nome_configuracao": NOME_MODELO,
        "parametros": {
            "n_estimators": 300,
            **configuracao,
            "random_state": 42,
        },
        "limiar_alerta": LIMIAR_ALERTA,
        "classes": {"0": "sem_alerta", "1": "com_alerta"},
        "target": COLUNA_TARGET_TAXA,
        "regra_target": {
            "corte_taxa_6m_por_100k": CORTE_TAXA_100K,
            "meses_com_ocorrencia_minimos": MESES_COM_OCORRENCIA_MINIMOS,
        },
        "quantidade_treinamento": len(base),
        "periodo_consultas_treinamento": {
            "inicio": str(base["data_referencia"].min().date()),
            "fim": str(base["data_referencia"].max().date()),
        },
        "colunas_numericas": colunas_numericas,
        "colunas_categoricas": COLUNAS_CATEGORICAS,
        "colunas_modelo": colunas_modelo,
        "metricas_validacao_temporal_2020_2023": metricas_validacao,
        "observacao_metricas": (
            "As metricas pertencem as validacoes temporais. O artefato foi "
            "reajustado com todas as respostas conhecidas e nao foi reavaliado."
        ),
        "versoes": {
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    pacote = {
        "modelo": modelo,
        "limiar_alerta": LIMIAR_ALERTA,
        "metadata": metadata,
    }

    PASTA_MODELO.mkdir(parents=True, exist_ok=True)
    joblib.dump(pacote, ARQUIVO_MODELO, compress=3)
    ARQUIVO_METADATA.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    RELATORIO.parent.mkdir(parents=True, exist_ok=True)
    tamanho_mb = ARQUIVO_MODELO.stat().st_size / (1024 * 1024)
    with RELATORIO.open("w", encoding="utf-8") as relatorio:
        relatorio.write("Empacotamento do modelo proporcional\n\n")
        relatorio.write(f"Artefato: {ARQUIVO_MODELO}\n")
        relatorio.write(f"Metadata: {ARQUIVO_METADATA}\n")
        relatorio.write(f"Tamanho: {tamanho_mb:.2f} MB\n")
        relatorio.write(f"Algoritmo: Random Forest ({NOME_MODELO})\n")
        relatorio.write(f"Limiar: {LIMIAR_ALERTA}\n")
        relatorio.write(f"Linhas usadas no reajuste: {len(base)}\n")
        relatorio.write(
            "O reajuste usa todos os targets conhecidos e nao representa uma "
            "nova avaliacao.\n"
        )

    print(f"Modelo salvo em: {ARQUIVO_MODELO}")
    print(f"Metadata salva em: {ARQUIVO_METADATA}")
    print(f"Tamanho do artefato: {tamanho_mb:.2f} MB")


if __name__ == "__main__":
    main()
