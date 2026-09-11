from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import obter_servico
from api.services.modelo import ConsultaInvalidaError, ServicoPredicao


router = APIRouter(prefix="/api/v1", tags=["consultas"])


@router.get("/municipios")
def municipios(
    uf: str = Query(min_length=2, max_length=2, examples=["PR"]),
    servico: ServicoPredicao = Depends(obter_servico),
) -> dict[str, object]:
    try:
        itens = servico.listar_municipios(uf)
    except ConsultaInvalidaError as erro:
        raise HTTPException(status_code=422, detail=str(erro)) from erro
    return {"uf": uf.upper(), "quantidade": len(itens), "municipios": itens}


@router.get("/periodos")
def periodos(
    uf: str = Query(min_length=2, max_length=2, examples=["PR"]),
    municipio: str = Query(
        min_length=2, max_length=150, examples=["Curitiba"]
    ),
    servico: ServicoPredicao = Depends(obter_servico),
) -> dict[str, object]:
    try:
        itens = servico.listar_periodos(uf, municipio)
    except ConsultaInvalidaError as erro:
        raise HTTPException(status_code=404, detail=str(erro)) from erro
    return {
        "uf": uf.upper(),
        "municipio": municipio,
        "quantidade": len(itens),
        "periodos": itens,
    }
