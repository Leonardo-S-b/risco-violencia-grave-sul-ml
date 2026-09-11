import unittest

import pandas as pd

from scripts.versoes.maximo_movel_6m.criar_target_violencia_grave import (
    criar_target_alerta_proximos_meses,
    criar_target_por_maximo_historico,
    criar_target_por_maximo_movel,
)
from scripts.versoes.maximo_movel_6m.comparar_balanceamento import aplicar_estrategia_treino
from scripts.experimentos.comparar_perfis_anuais_municipios import (
    criar_resumos_anuais_municipios,
)
from scripts.experimentos.comparar_variaveis_populacionais import adicionar_populacao_disponivel
from scripts.experimentos.comparar_historico_longo import criar_variaveis_historico_longo
from scripts.experimentos.comparar_targets_taxa_100k import adicionar_taxa_futura_6m
from scripts.versoes.taxa_100k.criar_target_taxa_100k import (
    COLUNA_TARGET_TAXA,
    criar_target_taxa_100k,
)
from scripts.versoes.maximo_movel_6m.treinar_modelo_risco import (
    criar_particoes_temporais,
    criar_variaveis_historicas,
)


class TestPipelineModelagem(unittest.TestCase):
    def test_target_compara_com_maximo_anterior(self) -> None:
        base = pd.DataFrame(
            {
                "uf": ["PR"] * 4,
                "municipio": ["EXEMPLO"] * 4,
                "data_referencia": pd.date_range("2020-01-01", periods=4, freq="MS"),
                "total_violencia_grave": [0, 2, 2, 3],
            }
        )

        resultado = criar_target_por_maximo_historico(base)

        self.assertTrue(pd.isna(resultado.iloc[0]["alto_risco_violencia_grave"]))
        self.assertEqual(resultado["alto_risco_violencia_grave"].iloc[1:].tolist(), [1, 0, 1])
        self.assertEqual(
            resultado["maximo_historico_ate_mes_anterior"].iloc[1:].tolist(),
            [0.0, 2.0, 2.0],
        )

    def test_variaveis_usam_apenas_meses_anteriores(self) -> None:
        totais = list(range(19))
        base = pd.DataFrame(
            {
                "uf": ["SC"] * 19,
                "municipio": ["EXEMPLO"] * 19,
                "data_referencia": pd.date_range("2020-01-01", periods=19, freq="MS"),
                "ano": [2020] * 12 + [2021] * 7,
                "mes": list(range(1, 13)) + list(range(1, 8)),
                "total_violencia_grave": totais,
                "homicidio_doloso": totais,
                "latrocinio": [0] * 19,
                "tentativa_homicidio": [0] * 19,
                "lesao_corporal_seguida_morte": [0] * 19,
                "feminicidio": [0] * 19,
            }
        )
        base = criar_target_alerta_proximos_meses(base)

        resultado, colunas = criar_variaveis_historicas(base)

        self.assertEqual(len(resultado), 2)
        self.assertEqual(resultado.iloc[0]["total_violencia_grave"], 11)
        self.assertEqual(resultado.iloc[0]["total_lag_1m"], 10)
        self.assertEqual(resultado.iloc[0]["media_movel_12m"], 5.5)
        self.assertIn("total_violencia_grave", colunas)

    def test_target_movel_exige_e_usa_doze_meses_anteriores(self) -> None:
        base = pd.DataFrame(
            {
                "uf": ["RS"] * 14,
                "municipio": ["EXEMPLO"] * 14,
                "data_referencia": pd.date_range("2020-01-01", periods=14, freq="MS"),
                "total_violencia_grave": list(range(12)) + [12, 5],
            }
        )

        resultado = criar_target_por_maximo_movel(base)

        self.assertEqual(resultado["alto_risco_violencia_grave"].isna().sum(), 12)
        self.assertEqual(
            resultado["alto_risco_violencia_grave"].iloc[12:].tolist(), [1, 0]
        )
        self.assertEqual(
            resultado["maximo_movel_12m_ate_mes_anterior"].iloc[12:].tolist(),
            [11.0, 12.0],
        )

    def test_target_dos_proximos_seis_meses(self) -> None:
        base = pd.DataFrame(
            {
                "uf": ["PR"] * 19,
                "municipio": ["EXEMPLO"] * 19,
                "data_referencia": pd.date_range("2020-01-01", periods=19, freq="MS"),
                "total_violencia_grave": list(range(12)) + [12] + [5] * 6,
            }
        )

        resultado = criar_target_alerta_proximos_meses(base)
        elegiveis = resultado.dropna(
            subset=["alerta_violencia_grave_proximos_6m"]
        )

        self.assertEqual(
            elegiveis["alerta_violencia_grave_proximos_6m"].tolist(),
            [1, 0],
        )
        self.assertEqual(
            elegiveis["maximo_real_proximos_6m"].tolist(),
            [12.0, 5.0],
        )

    def test_particao_temporal_respeita_fim_do_horizonte(self) -> None:
        base = pd.DataFrame(
            {
                "data_referencia": pd.to_datetime(
                    ["2019-05-01", "2019-06-01", "2020-01-01"]
                ),
                "data_fim_horizonte": pd.to_datetime(
                    ["2019-11-01", "2019-12-01", "2020-07-01"]
                ),
            }
        )

        treino, validacao = criar_particoes_temporais(base, 2020)

        self.assertEqual(len(treino), 2)
        self.assertEqual(len(validacao), 1)

    def test_undersampling_altera_somente_proporcao_do_treino(self) -> None:
        treino = pd.DataFrame(
            {
                "identificador": range(110),
                "alerta_violencia_grave_proximos_6m": [1] * 10 + [0] * 100,
            }
        )

        resultado = aplicar_estrategia_treino(
            treino, "undersampling_1_5", semente=42
        )

        distribuicao = resultado[
            "alerta_violencia_grave_proximos_6m"
        ].value_counts()
        self.assertEqual(distribuicao[1], 10)
        self.assertEqual(distribuicao[0], 50)
        self.assertEqual(len(treino), 110)

    def test_perfil_anual_nao_usa_meses_do_proprio_ano(self) -> None:
        datas = pd.date_range("2019-01-01", periods=36, freq="MS")
        base = pd.DataFrame(
            {
                "uf": ["PR"] * 36,
                "municipio": ["EXEMPLO"] * 36,
                "data_referencia": datas,
                "total_violencia_grave": list(range(24)) + [100] * 12,
            }
        )

        resumo_original = criar_resumos_anuais_municipios(base)
        base.loc[base["data_referencia"].dt.year == 2021, "total_violencia_grave"] = 999
        resumo_alterado = criar_resumos_anuais_municipios(base)
        perfil_original = resumo_original[resumo_original["ano"] == 2021].iloc[0]
        perfil_alterado = resumo_alterado[resumo_alterado["ano"] == 2021].iloc[0]

        self.assertEqual(
            perfil_original["municipio_media_24m_anteriores"],
            perfil_alterado["municipio_media_24m_anteriores"],
        )
        self.assertEqual(
            perfil_original["municipio_maximo_24m_anteriores"],
            perfil_alterado["municipio_maximo_24m_anteriores"],
        )

    def test_populacao_utiliza_ultimo_ano_anterior_disponivel(self) -> None:
        base = pd.DataFrame(
            {
                "uf": ["PR", "PR"],
                "municipio": ["EXEMPLO", "EXEMPLO"],
                "ano": [2020, 2023],
            }
        )
        populacao = pd.DataFrame(
            {
                "uf": ["PR", "PR"],
                "municipio_normalizado": ["EXEMPLO", "EXEMPLO"],
                "ano_populacao": [2019, 2021],
                "codigo_ibge": ["4100000", "4100000"],
                "populacao": [10_000, 11_000],
            }
        )

        resultado = adicionar_populacao_disponivel(base, populacao)

        self.assertEqual(resultado["ano_populacao"].tolist(), [2019, 2021])
        self.assertEqual(resultado["populacao"].tolist(), [10_000, 11_000])

    def test_historico_longo_nao_muda_com_ocorrencia_futura(self) -> None:
        base = pd.DataFrame(
            {
                "uf": ["SC"] * 40,
                "municipio": ["EXEMPLO"] * 40,
                "data_referencia": pd.date_range(
                    "2019-01-01", periods=40, freq="MS"
                ),
                "total_violencia_grave": [0] * 20 + [1] + [0] * 19,
                "populacao": [10_000] * 40,
            }
        )
        alterada = base.copy()
        alterada.loc[39, "total_violencia_grave"] = 100

        original, _ = criar_variaveis_historico_longo(base)
        resultado_alterado, _ = criar_variaveis_historico_longo(alterada)

        colunas = [
            "media_movel_24m",
            "media_movel_36m",
            "meses_com_ocorrencia_36m",
            "meses_desde_ultima_ocorrencia",
        ]
        self.assertEqual(
            original.loc[38, colunas].tolist(),
            resultado_alterado.loc[38, colunas].tolist(),
        )

    def test_taxa_futura_conta_casos_e_meses_dos_proximos_seis_meses(
        self,
    ) -> None:
        base = pd.DataFrame(
            {
                "uf": ["PR"] * 7,
                "municipio": ["EXEMPLO"] * 7,
                "total_violencia_grave": [0, 1, 0, 2, 0, 0, 0],
                "populacao": [10_000] * 7,
            }
        )

        resultado = adicionar_taxa_futura_6m(base)

        self.assertEqual(
            resultado.loc[0, "soma_violencia_grave_proximos_6m"], 3
        )
        self.assertEqual(
            resultado.loc[0, "meses_com_violencia_grave_proximos_6m"], 2
        )
        self.assertAlmostEqual(
            resultado.loc[0, "taxa_violencia_grave_proximos_6m_100k"], 30
        )

    def test_target_proporcional_ignora_um_caso_isolado(self) -> None:
        datas = pd.date_range("2020-01-01", periods=18, freq="MS")
        isolado = pd.DataFrame(
            {
                "uf": ["PR"] * 18,
                "municipio": ["ISOLADO"] * 18,
                "data_referencia": datas,
                "total_violencia_grave": [0] * 12 + [1] + [0] * 5,
                "populacao": [5_000] * 18,
            }
        )
        persistente = isolado.copy()
        persistente["municipio"] = "PERSISTENTE"
        persistente["total_violencia_grave"] = [0] * 12 + [1, 1] + [0] * 4
        base = pd.concat([isolado, persistente], ignore_index=True)

        resultado = criar_target_taxa_100k(base)
        consulta = resultado[resultado["data_referencia"] == datas[11]]
        targets = consulta.set_index("municipio")[COLUNA_TARGET_TAXA]

        self.assertEqual(targets["ISOLADO"], 0)
        self.assertEqual(targets["PERSISTENTE"], 1)


if __name__ == "__main__":
    unittest.main()
