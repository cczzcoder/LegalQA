"""确定性领域别名扩展。

LLM 改写默认关闭；少数领域存在明显词表鸿沟（口语说法 vs 法条术语），
用纯规则把法条术语追加到检索串。规则必须窄：通用口语→术语词表实测会
稀释原句语义、拉低 R@6，所以只对命中领域标记的查询生效。
"""

from __future__ import annotations

__all__ = ["expand_query"]

# (触发标记, 追加的检索词)
_ALIAS_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        (
            "方言",
            "繁体字",
            "简体字",
            "普通话",
            "规范汉字",
            "国家通用语言文字",
            "国际中文教育",
            "中文教育",
            "海外中文",
            "学中文",
            "中文课",
            "汉语",
        ),
        "国际中文教育 普通话 规范汉字",
    ),
)


def expand_query(text: str) -> tuple[str, tuple[str, ...]]:
    """返回 (扩展后的查询, 实际追加的词)。无命中时原样返回。"""
    original = str(text or "").strip()
    if not original:
        return original, ()

    applied: list[str] = []
    for markers, expansion in _ALIAS_RULES:
        tokens = expansion.split()
        if all(token in original for token in tokens):
            continue
        if any(marker in original for marker in markers):
            applied.append(expansion)
    if not applied:
        return original, ()
    return original + " " + " ".join(applied), tuple(applied)
