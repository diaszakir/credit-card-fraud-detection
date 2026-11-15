"""Общий запуск команд модуля: понятное сообщение вместо traceback при ожидаемых ошибках."""

from __future__ import annotations

import sys
from collections.abc import Callable


def run_cli(main: Callable[[], int]) -> int:
    """Выполнить main(); отсутствующий файл или нарушенная проверка — сообщение и код 1."""
    try:
        return main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
