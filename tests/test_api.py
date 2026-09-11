import unittest
from datetime import date
from pathlib import Path
from time import sleep

import pandas as pd

from api.cache import CacheTTL
from api.schemas import ConsultaPredicao
from api.services.modelo import ServicoPredicao


class TestCacheAPI(unittest.TestCase):
    def test_frontend_usa_nome_do_contexto_definido_pela_api(self) -> None:
        javascript = Path("frontend/app.js").read_text(encoding="utf-8")

        self.assertIn("result.contexto", javascript)
        self.assertNotIn("result.context &&", javascript)

    def test_cache_retorna_resultado_guardado(self) -> None:
        cache = CacheTTL[str, dict[str, int]](tamanho_maximo=2, ttl_segundos=10)
        cache.guardar("PR|CURITIBA|2025-06-01", {"classe": 1})

        self.assertEqual(
            cache.obter("PR|CURITIBA|2025-06-01"), {"classe": 1}
        )
        self.assertEqual(cache.estatisticas()["acertos"], 1)

    def test_cache_remove_resultado_expirado(self) -> None:
        cache = CacheTTL[str, int](tamanho_maximo=2, ttl_segundos=0.01)
        cache.guardar("consulta", 1)
        sleep(0.02)

        self.assertIsNone(cache.obter("consulta"))

    def test_consulta_normaliza_uf_e_municipio(self) -> None:
        consulta = ConsultaPredicao(
            uf=" pr ", municipio="  Sao   Jose  ", data_referencia=date(2025, 6, 1)
        )

        self.assertEqual(consulta.uf, "PR")
        self.assertEqual(consulta.municipio, "Sao Jose")

    def test_contexto_usa_somente_historico_ate_consulta(self) -> None:
        servico = ServicoPredicao.__new__(ServicoPredicao)
        servico.historico = pd.DataFrame(
            {
                "uf": ["RS"] * 13,
                "municipio_normalizado": ["EXEMPLO"] * 13,
                "data_referencia": pd.date_range("2020-01-01", periods=13, freq="MS"),
                "total_violencia_grave": [1] * 6 + [2] * 6 + [999],
            }
        )
        registro = pd.Series(
            {
                "uf": "RS",
                "municipio_normalizado": "EXEMPLO",
                "data_referencia": pd.Timestamp("2020-12-01"),
                "populacao": 100_000,
                "total_violencia_grave": 2,
                "taxa_violencia_6m_100k": 12,
                "taxa_violencia_12m_100k": 18,
                "taxa_violencia_12m_uf_100k": 9,
                "homicidio_doloso_soma_12m": 5,
                "tentativa_homicidio_soma_12m": 10,
                "latrocinio_soma_12m": 1,
                "feminicidio_soma_12m": 1,
                "lesao_corporal_seguida_morte_soma_12m": 1,
            }
        )

        contexto = servico._criar_contexto(registro)

        self.assertEqual(len(contexto["historico_12m"]), 12)
        self.assertEqual(contexto["tendencia_6m"], "aumento")
        self.assertEqual(contexto["razao_municipio_estado"], 2)
        self.assertEqual(
            contexto["principal_categoria_12m"], "Tentativa de homicídio"
        )
        self.assertNotIn(999, [item["total"] for item in contexto["historico_12m"]])


if __name__ == "__main__":
    unittest.main()
