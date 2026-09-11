from __future__ import annotations

from datetime import date
from pathlib import Path

import joblib
import pandas as pd

from api.cache import CacheTTL
from scripts.dados.baixar_populacao_ibge import normalizar_nome
from scripts.experimentos.comparar_variaveis_populacionais import (
    criar_variaveis_populacionais,
)
from scripts.versoes.maximo_movel_6m.treinar_modelo_risco import (
    COLUNA_TARGET,
    criar_variaveis_historicas,
)


class ConsultaInvalidaError(ValueError):
    """A consulta não corresponde aos dados disponíveis."""


class ServicoPredicao:
    ROTULOS_EVENTOS = {
        "homicidio_doloso": "Homicídio doloso",
        "tentativa_homicidio": "Tentativa de homicídio",
        "latrocinio": "Latrocínio",
        "feminicidio": "Feminicídio",
        "lesao_corporal_seguida_morte": "Lesão corporal seguida de morte",
    }

    def __init__(
        self,
        arquivo_modelo: Path,
        arquivo_dados: Path,
        cache_tamanho: int,
        cache_ttl_segundos: int,
    ) -> None:
        if not arquivo_modelo.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {arquivo_modelo}")
        if not arquivo_dados.exists():
            raise FileNotFoundError(f"Dados não encontrados: {arquivo_dados}")

        pacote = joblib.load(arquivo_modelo)
        self.modelo = pacote["modelo"]
        self.limiar = float(pacote["limiar_alerta"])
        self.metadata = pacote["metadata"]
        self.colunas_modelo = list(self.metadata["colunas_modelo"])
        self.cache = CacheTTL[tuple[str, str, str], dict[str, object]](
            cache_tamanho, cache_ttl_segundos
        )
        self.base = self._preparar_base(arquivo_dados)

    def _preparar_base(self, arquivo_dados: Path) -> pd.DataFrame:
        base = pd.read_csv(arquivo_dados, dtype={"codigo_ibge": str})

        # A funcao compartilhada com o treinamento remove linhas sem target.
        # Na API queremos tambem os meses mais recentes, portanto usamos uma
        # coluna auxiliar neutra. Nenhuma resposta futura entra nas variaveis.
        base[COLUNA_TARGET] = 0
        base, _ = criar_variaveis_historicas(base)
        base, _ = criar_variaveis_populacionais(base)
        base["data_referencia"] = pd.to_datetime(base["data_referencia"])
        self.historico = base[
            [
                "uf",
                "municipio_normalizado",
                "data_referencia",
                "total_violencia_grave",
            ]
        ].sort_values(["uf", "municipio_normalizado", "data_referencia"])
        base = base[base["meses_historico"] >= 12].copy()

        faltantes = set(self.colunas_modelo) - set(base.columns)
        if faltantes:
            raise ValueError(
                f"A base não possui as colunas exigidas pelo modelo: {sorted(faltantes)}"
            )

        base = base.sort_values(["uf", "municipio", "data_referencia"])
        duplicadas = base.duplicated(
            ["uf", "municipio_normalizado", "data_referencia"]
        )
        if duplicadas.any():
            raise ValueError("Existem consultas duplicadas na base da API.")
        return base.set_index(
            ["uf", "municipio_normalizado", "data_referencia"], drop=False
        )

    @property
    def periodo_disponivel(self) -> dict[str, str]:
        return {
            "inicio": str(self.base["data_referencia"].min().date()),
            "fim": str(self.base["data_referencia"].max().date()),
        }

    def listar_municipios(self, uf: str) -> list[str]:
        uf = uf.strip().upper()
        if uf not in {"PR", "RS", "SC"}:
            raise ConsultaInvalidaError("UF deve ser PR, RS ou SC.")
        return sorted(
            self.base.loc[self.base["uf"] == uf, "municipio"].unique().tolist()
        )

    def listar_periodos(self, uf: str, municipio: str) -> list[str]:
        uf = uf.strip().upper()
        municipio_normalizado = normalizar_nome(municipio)
        mascara = (
            (self.base["uf"] == uf)
            & (self.base["municipio_normalizado"] == municipio_normalizado)
        )
        periodos = self.base.loc[mascara, "data_referencia"]
        if periodos.empty:
            raise ConsultaInvalidaError("Município não encontrado para a UF informada.")
        return [str(valor.date()) for valor in periodos.sort_values()]

    def prever(
        self, uf: str, municipio: str, data_referencia: date
    ) -> tuple[dict[str, object], bool]:
        chave = (
            uf.strip().upper(),
            normalizar_nome(municipio),
            data_referencia.isoformat(),
        )
        resultado_cache = self.cache.obter(chave)
        if resultado_cache is not None:
            return resultado_cache, True

        indice = (chave[0], chave[1], pd.Timestamp(data_referencia))
        try:
            linha = self.base.loc[[indice]]
        except KeyError as erro:
            raise ConsultaInvalidaError(
                "Não existem dados suficientes para esse município e período."
            ) from erro

        probabilidade = float(
            self.modelo.predict_proba(linha[self.colunas_modelo])[:, 1][0]
        )
        classe = int(probabilidade >= self.limiar)
        registro = linha.iloc[0]
        nome_municipio = str(registro["municipio"])
        resultado: dict[str, object] = {
            "uf": chave[0],
            "municipio": nome_municipio,
            "data_referencia": data_referencia,
            "horizonte_meses": 6,
            "classe": classe,
            "classificacao": "com_alerta" if classe else "sem_alerta",
            "alerta": bool(classe),
            "probabilidade": round(probabilidade, 6),
            "limiar": self.limiar,
            "mensagem": (
                "Há indicação de alerta para os próximos 6 meses; solicite uma "
                "análise complementar de segurança."
                if classe
                else "Não há indicação de alerta; continue a consulta preliminar "
                "com os demais indicadores."
            ),
            "contexto": self._criar_contexto(registro),
        }
        self.cache.guardar(chave, resultado)
        return resultado, False

    def _criar_contexto(self, registro: pd.Series) -> dict[str, object]:
        mascara_historico = (
            (self.historico["uf"] == registro["uf"])
            & (
                self.historico["municipio_normalizado"]
                == registro["municipio_normalizado"]
            )
            & (self.historico["data_referencia"] <= registro["data_referencia"])
        )
        ultimos_12m = self.historico.loc[mascara_historico].tail(12)
        totais = ultimos_12m["total_violencia_grave"].astype(int).tolist()
        total_6m_anterior = sum(totais[:6])
        total_6m_recente = sum(totais[6:])

        if total_6m_recente > total_6m_anterior:
            tendencia = "aumento"
        elif total_6m_recente < total_6m_anterior:
            tendencia = "queda"
        else:
            tendencia = "estabilidade"
        variacao = (
            round(
                (total_6m_recente - total_6m_anterior)
                / total_6m_anterior
                * 100,
                2,
            )
            if total_6m_anterior > 0
            else None
        )

        composicao = [
            {
                "categoria": rotulo,
                "total_12m": int(registro[f"{evento}_soma_12m"]),
            }
            for evento, rotulo in self.ROTULOS_EVENTOS.items()
        ]
        composicao.sort(key=lambda item: int(item["total_12m"]), reverse=True)

        taxa_municipal = float(registro["taxa_violencia_12m_100k"])
        taxa_estado = float(registro["taxa_violencia_12m_uf_100k"])
        return {
            "populacao": int(registro["populacao"]),
            "total_mes_referencia": int(registro["total_violencia_grave"]),
            "taxa_ultimos_6m_100k": round(
                float(registro["taxa_violencia_6m_100k"]), 2
            ),
            "taxa_ultimos_12m_100k": round(taxa_municipal, 2),
            "taxa_estado_ultimos_12m_100k": round(taxa_estado, 2),
            "razao_municipio_estado": (
                round(taxa_municipal / taxa_estado, 2) if taxa_estado > 0 else None
            ),
            "tendencia_6m": tendencia,
            "variacao_6m_percentual": variacao,
            "principal_categoria_12m": str(composicao[0]["categoria"]),
            "historico_12m": [
                {
                    "periodo": periodo.date(),
                    "total": int(total),
                }
                for periodo, total in zip(
                    ultimos_12m["data_referencia"], totais, strict=True
                )
            ],
            "composicao_12m": composicao,
        }
