"""Unit tests for zerorl.compiler (fast_compile, _cxx_compiler_available)."""

import torch
from zerorl.compiler import fast_compile, _cxx_compiler_available


class TestCxxCompilerAvailable:
    def test_returns_bool(self) -> None:
        result = _cxx_compiler_available()
        assert isinstance(result, bool)

    def test_consistent_across_calls(self) -> None:
        a = _cxx_compiler_available()
        b = _cxx_compiler_available()
        assert a == b


class TestFastCompileNoCompiler:
    def test_noop_when_no_compiler(self, monkeypatch) -> None:
        monkeypatch.setattr("zerorl.compiler._cxx_compiler_available", lambda: False)

        @fast_compile
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    def test_noop_returns_same_identity(self, monkeypatch) -> None:
        monkeypatch.setattr("zerorl.compiler._cxx_compiler_available", lambda: False)

        @fast_compile
        def identity(x):
            return x

        inp = torch.randn(4)
        result = identity(inp)  # type: ignore[arg-type]
        assert result is inp


class TestFastCompileDebug:
    def test_debug_flag_always_noop(self, monkeypatch) -> None:
        @fast_compile(debug=True)
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    def test_debug_flag_noop_even_with_compiler(self, monkeypatch) -> None:
        monkeypatch.setattr("zerorl.compiler._cxx_compiler_available", lambda: True)

        @fast_compile(debug=True)
        def add(a, b):
            return a + b

        assert add(1, 2) == 3


class TestFastCompileAsDecorator:
    def test_called_with_parens(self) -> None:
        @fast_compile()
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    def test_called_with_debug_kwarg(self) -> None:
        @fast_compile(debug=True)
        def add(a, b):
            return a + b

        assert add(1, 2) == 3


class TestFastCompileOutput:
    def test_preserves_return_value(self, monkeypatch) -> None:
        monkeypatch.setattr("zerorl.compiler._cxx_compiler_available", lambda: False)

        @fast_compile
        def square(x):
            return x * x

        inp = torch.tensor([2.0, 3.0])
        result = square(inp)  # type: ignore[arg-type]
        torch.testing.assert_close(result, torch.tensor([4.0, 9.0]))

    def test_preserves_return_type(self, monkeypatch) -> None:
        monkeypatch.setattr("zerorl.compiler._cxx_compiler_available", lambda: False)

        @fast_compile
        def make_dict(x):
            return {"val": x}

        result = make_dict(42)  # type: ignore[arg-type]
        assert isinstance(result, dict)
        assert result["val"] == 42
