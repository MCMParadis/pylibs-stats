from pathlib import Path

from pylibs.interfaces.dsl.verbs import VERBS


def run_script(path: Path) -> None:
    source = Path(path).read_text()
    namespace = dict(VERBS)
    code = compile(source, str(path), "exec")
    exec(code, namespace)
