import unittest

import pandas as pd

from scripts.criar_target_violencia_grave import (
    criar_target_alerta_proximos_meses,
    criar_target_por_maximo_historico,
    criar_target_por_maximo_movel,
)
from scripts.comparar_balanceamento import aplicar_estrategia_treino
from scripts.treinar_modelo_risco import (
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


if __name__ == "__main__":
    unittest.main()
