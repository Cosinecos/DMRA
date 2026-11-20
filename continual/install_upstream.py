"""Install the pinned baseline predictors into the local, untracked _upstream package."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
LOCK = json.loads((HERE / "upstream.lock.json").read_text(encoding="utf-8"))


def install(model, checkout):
    spec = LOCK[model]
    package = HERE / "_upstream" / spec["namespace"]
    verified = {}
    for path, expected in spec["files"].items():
        if checkout is None:
            url = f"https://raw.githubusercontent.com/{spec['repository']}/{spec['revision']}/{path}"
            with urlopen(url, timeout=60) as response:
                content = response.read()
        else:
            content = (checkout / path).read_bytes()
        content = content.replace(b"\r\n", b"\n")
        if hashlib.sha256(content).hexdigest() != expected["sha256_lf"]:
            raise ValueError(f"{model}: upstream content does not match the pinned revision for {path}")
        if expected["relative_gcn_import"]:
            content = content.replace(b"from model.gcn_conv import", b"from .gcn_conv import")
        verified[Path(path).name] = content

    package.mkdir(parents=True, exist_ok=True)
    for name, content in verified.items():
        with tempfile.NamedTemporaryFile(dir=package, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        try:
            os.replace(temporary, package / name)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Installed {model} from {spec['repository']} at {spec['revision']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["all", *LOCK])
    parser.add_argument("--checkout", type=Path, help="Read from the specified local checkout of one model")
    args = parser.parse_args()
    if args.checkout and args.model == "all":
        parser.error("--checkout requires an individual model")
    for model_name in (LOCK if args.model == "all" else [args.model]):
        install(model_name, args.checkout)

