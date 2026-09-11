from fastapi import Request

from api.services.modelo import ServicoPredicao


def obter_servico(request: Request) -> ServicoPredicao:
    """Entrega aos controllers o servico criado na inicializacao da API."""
    return request.app.state.servico
