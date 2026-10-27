"""带 CSRF 头的 TestClient 包装。

先 GET /api/auth/me 拿到 lm_csrf cookie（未登录也会下发），之后所有
不安全方法自动带 X-CSRF-Token 头；登录/注册会轮换 token，响应后自动同步。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


class CsrfClient(TestClient):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.csrf = ""
        self.get("/api/auth/me")
        self.csrf = self.cookies.get("lm_csrf", "")

    def request(self, method, url, **kwargs):
        if method.upper() in UNSAFE_METHODS and self.csrf:
            headers = dict(kwargs.get("headers") or {})
            headers.setdefault("X-CSRF-Token", self.csrf)
            kwargs["headers"] = headers
        response = super().request(method, url, **kwargs)
        token = self.cookies.get("lm_csrf")
        if token:
            self.csrf = token
        return response
