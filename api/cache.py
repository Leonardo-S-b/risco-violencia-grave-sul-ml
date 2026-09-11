from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Generic, Hashable, TypeVar


Chave = TypeVar("Chave", bound=Hashable)
Valor = TypeVar("Valor")


@dataclass
class ItemCache(Generic[Valor]):
    valor: Valor
    expira_em: float


class CacheTTL(Generic[Chave, Valor]):
    """Cache LRU em memoria com expiracao, seguro para acesso concorrente."""

    def __init__(self, tamanho_maximo: int, ttl_segundos: int) -> None:
        if tamanho_maximo <= 0 or ttl_segundos <= 0:
            raise ValueError("Tamanho e TTL do cache precisam ser positivos.")
        self.tamanho_maximo = tamanho_maximo
        self.ttl_segundos = ttl_segundos
        self._itens: OrderedDict[Chave, ItemCache[Valor]] = OrderedDict()
        self._lock = RLock()
        self._acertos = 0
        self._falhas = 0

    def obter(self, chave: Chave) -> Valor | None:
        with self._lock:
            item = self._itens.get(chave)
            if item is None:
                self._falhas += 1
                return None
            if item.expira_em <= monotonic():
                del self._itens[chave]
                self._falhas += 1
                return None
            self._itens.move_to_end(chave)
            self._acertos += 1
            return deepcopy(item.valor)

    def guardar(self, chave: Chave, valor: Valor) -> None:
        with self._lock:
            self._itens[chave] = ItemCache(
                valor=deepcopy(valor),
                expira_em=monotonic() + self.ttl_segundos,
            )
            self._itens.move_to_end(chave)
            while len(self._itens) > self.tamanho_maximo:
                self._itens.popitem(last=False)

    def estatisticas(self) -> dict[str, int]:
        with self._lock:
            return {
                "itens": len(self._itens),
                "acertos": self._acertos,
                "falhas": self._falhas,
                "ttl_segundos": self.ttl_segundos,
                "tamanho_maximo": self.tamanho_maximo,
            }
