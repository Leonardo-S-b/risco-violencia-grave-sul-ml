from fastapi import APIRouter, Depends

from api.dependencies import obter_servico
from api.services.modelo import ServicoPredicao


router = APIRouter(tags=["sistema"])


@router.get("/api/v1")
def inicio() -> dict[str, str]:
    return {
        "nome": "API de Consulta Preliminar de Violência Grave",
        "documentacao": "/docs",
        "status": "/api/v1/health",
    }


@router.get("/api/v1/health")
def health(
    servico: ServicoPredicao = Depends(obter_servico),
) -> dict[str, object]:
    return {
        "status": "ok",
        "modelo_carregado": True,
        "periodo_disponivel": servico.periodo_disponivel,
        "cache": servico.cache.estatisticas(),
    }


@router.get("/api/v1/metadata", tags=["modelo"])
def metadata(
    servico: ServicoPredicao = Depends(obter_servico),
) -> dict[str, object]:
    return servico.metadata
