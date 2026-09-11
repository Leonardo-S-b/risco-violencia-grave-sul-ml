from starlette.responses import Response
from starlette.staticfiles import StaticFiles


class StaticFilesRevalidado(StaticFiles):
    """Evita que HTML, CSS e JavaScript fiquem incompatíveis no navegador."""

    async def get_response(self, path: str, scope: dict) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
