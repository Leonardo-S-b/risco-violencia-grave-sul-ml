from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from api.config import (
    ARQUIVO_DADOS,
    ARQUIVO_MODELO,
    CACHE_TAMANHO_MAXIMO,
    CACHE_TTL_SEGUNDOS,
    PASTA_FRONTEND,
)
from api.controllers.consultas_controller import router as consultas_router
from api.controllers.predicoes_controller import router as predicoes_router
from api.controllers.sistema_controller import router as sistema_router
from api.services.modelo import ServicoPredicao
from api.static import StaticFilesRevalidado


@asynccontextmanager
async def lifespan(aplicacao: FastAPI) -> AsyncIterator[None]:
    aplicacao.state.servico = ServicoPredicao(
        ARQUIVO_MODELO,
        ARQUIVO_DADOS,
        CACHE_TAMANHO_MAXIMO,
        CACHE_TTL_SEGUNDOS,
    )
    yield


app = FastAPI(
    title="API de Consulta Preliminar de Violência Grave",
    description=(
        "Classifica o risco proporcional de violência grave nos próximos seis "
        "meses. Uso acadêmico e preliminar."
    ),
    version="1.1.3",
    lifespan=lifespan,
)

app.include_router(sistema_router)
app.include_router(consultas_router)
app.include_router(predicoes_router)
app.mount(
    "/",
    StaticFilesRevalidado(directory=PASTA_FRONTEND, html=True),
    name="frontend",
)
