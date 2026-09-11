from fastapi import APIRouter, Depends, HTTPException, Response

from api.dependencies import obter_servico
from api.schemas import ConsultaPredicao, ResultadoPredicao
from api.services.modelo import ConsultaInvalidaError, ServicoPredicao


router = APIRouter(prefix="/api/v1", tags=["predicoes"])


@router.post("/predicoes", response_model=ResultadoPredicao)
def predizer(
    consulta: ConsultaPredicao,
    response: Response,
    servico: ServicoPredicao = Depends(obter_servico),
) -> dict[str, object]:
    try:
        resultado, veio_do_cache = servico.prever(
            consulta.uf, consulta.municipio, consulta.data_referencia
        )
    except ConsultaInvalidaError as erro:
        raise HTTPException(status_code=404, detail=str(erro)) from erro
    response.headers["X-Cache"] = "HIT" if veio_do_cache else "MISS"
    return resultado
