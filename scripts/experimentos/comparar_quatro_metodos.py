from __future__ import annotations

import json
from pathlib import Path


ARQUIVOS = {
    "original": (
        Path("reports/experimentos/metricas_comparacao_variaveis_populacionais.json"),
        "modelo_original",
    ),
    "perfis_mensais": (
        Path("reports/experimentos/metricas_comparacao_perfis_municipios.json"),
        "modelo_com_perfis",
    ),
    "perfis_anuais": (
        Path("reports/experimentos/metricas_comparacao_perfis_anuais_municipios.json"),
        "modelo_com_perfis_anuais",
    ),
    "populacao_taxas_suavizadas": (
        Path("reports/experimentos/metricas_comparacao_variaveis_populacionais.json"),
        "modelo_com_populacao",
    ),
}
RELATORIO_TEXTO = Path("reports/experimentos/resumo_comparacao_quatro_metodos.txt")
RELATORIO_JSON = Path("reports/experimentos/metricas_comparacao_quatro_metodos.json")


def main() -> None:
    resultados: dict[str, dict[str, object]] = {}
    for nome, (arquivo, chave_modelo) in ARQUIVOS.items():
        dados = json.loads(arquivo.read_text(encoding="utf-8"))
        metricas = dados["resultados"][chave_modelo]
        resultados[nome] = {
            "cenario_f2": metricas["limiar_otimizado_f2"],
            "cenario_recall_minimo_70": metricas[
                "limiar_com_recall_minimo_70"
            ],
        }

    ranking_recall_70 = sorted(
        resultados,
        key=lambda nome: (
            int(
                resultados[nome]["cenario_recall_minimo_70"][
                    "matriz_confusao"
                ]["fp"]
            ),
            -float(
                resultados[nome]["cenario_recall_minimo_70"]["precision"]
            ),
        ),
    )
    ranking_ap = sorted(
        resultados,
        key=lambda nome: -float(resultados[nome]["cenario_f2"]["average_precision"]),
    )
    vencedor = ranking_recall_70[0]
    saida = {
        "criterio_principal": (
            "Menor quantidade de falsos positivos mantendo recall minimo de 70%."
        ),
        "teste_final_usado": False,
        "ranking_recall_minimo_70": ranking_recall_70,
        "ranking_average_precision": ranking_ap,
        "metodo_mais_benefico": vencedor,
        "resultados": resultados,
    }
    RELATORIO_TEXTO.parent.mkdir(parents=True, exist_ok=True)
    RELATORIO_JSON.write_text(
        json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with RELATORIO_TEXTO.open("w", encoding="utf-8") as relatorio:
        relatorio.write("Comparacao final dos quatro metodos\n\n")
        relatorio.write(
            "Criterio principal: menor numero de falsos positivos com recall "
            "minimo de 70%.\n"
        )
        relatorio.write("Validacoes temporais: 2020 a 2023.\n")
        relatorio.write("O teste final nao foi usado nesta decisao.\n\n")
        for posicao, nome in enumerate(ranking_recall_70, start=1):
            metricas = resultados[nome]["cenario_recall_minimo_70"]
            matriz = metricas["matriz_confusao"]
            relatorio.write(
                f"{posicao}. {nome}: AP={metricas['average_precision']}, "
                f"precisao={metricas['precision']}, "
                f"recall={metricas['recall']}, F2={metricas['f2']}, "
                f"FP={matriz['fp']}, FN={matriz['fn']}\n"
            )
        relatorio.write(f"\nMetodo mais benefico: {vencedor}.\n")

    print(f"Metodo mais benefico: {vencedor}")
    print(f"Comparacao salva em: {RELATORIO_TEXTO} e {RELATORIO_JSON}")


if __name__ == "__main__":
    main()
