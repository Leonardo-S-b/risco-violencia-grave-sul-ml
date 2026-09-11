from datetime import date

from pydantic import BaseModel, Field, field_validator


class ConsultaPredicao(BaseModel):
    uf: str = Field(min_length=2, max_length=2, examples=["PR"])
    municipio: str = Field(min_length=2, max_length=150, examples=["Curitiba"])
    data_referencia: date = Field(examples=["2025-06-01"])

    @field_validator("uf", mode="before")
    @classmethod
    def normalizar_uf(cls, valor: str) -> str:
        return valor.strip().upper()

    @field_validator("municipio", mode="before")
    @classmethod
    def limpar_municipio(cls, valor: str) -> str:
        return " ".join(valor.strip().split())

    @field_validator("data_referencia")
    @classmethod
    def validar_primeiro_dia(cls, valor: date) -> date:
        if valor.day != 1:
            raise ValueError("data_referencia deve ser o primeiro dia do mês")
        return valor


class PontoHistorico(BaseModel):
    periodo: date
    total: int


class ComposicaoViolencia(BaseModel):
    categoria: str
    total_12m: int


class ContextoMunicipio(BaseModel):
    populacao: int
    total_mes_referencia: int
    taxa_ultimos_6m_100k: float
    taxa_ultimos_12m_100k: float
    taxa_estado_ultimos_12m_100k: float
    razao_municipio_estado: float | None
    tendencia_6m: str
    variacao_6m_percentual: float | None
    principal_categoria_12m: str
    historico_12m: list[PontoHistorico]
    composicao_12m: list[ComposicaoViolencia]


class ResultadoPredicao(BaseModel):
    uf: str
    municipio: str
    data_referencia: date
    horizonte_meses: int = 6
    classe: int
    classificacao: str
    alerta: bool
    probabilidade: float
    limiar: float
    mensagem: str
    contexto: ContextoMunicipio
    observacao: str = (
        "Resultado destinado à consulta preliminar; não representa uma "
        "decisão automática de investimento."
    )
