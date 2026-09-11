from pathlib import Path


RAIZ_PROJETO = Path(__file__).resolve().parents[1]
PASTA_FRONTEND = RAIZ_PROJETO / "frontend"
ARQUIVO_MODELO = (
    RAIZ_PROJETO
    / "models"
    / "versoes"
    / "taxa_100k"
    / "modelo_risco_taxa_100k.joblib"
)
ARQUIVO_DADOS = (
    RAIZ_PROJETO
    / "data"
    / "processed"
    / "versoes"
    / "taxa_100k"
    / "violencia_grave_sul_com_target_taxa_100k.csv"
)

# O cache e local ao processo da API. Cada resultado expira depois de uma hora.
CACHE_TAMANHO_MAXIMO = 4_096
CACHE_TTL_SEGUNDOS = 3_600
