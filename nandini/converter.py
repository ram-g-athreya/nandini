"""Makes conversions through a worker in Seed-VC's environment. Finished ones are kept and reused."""
import json
import subprocess
from pathlib import Path

from . import config as C


# Seed-VC needs different package versions from the rest of this code, so it runs as a separate process, in its own environment.
class Converter:
    def __init__(self, seed_vc_dir, seed_vc_python, voice_sample):
        self.dir, self.python, self.voice = seed_vc_dir, seed_vc_python, voice_sample
        self.worker = None

    def make(self, source, seed, out):
        """Convert `source` with `seed` into `out`, unless `out` already exists (a restart reuses finished work)."""
        out, source = Path(out).resolve(), Path(source).resolve()  # the worker runs from Seed-VC's folder
        if out.exists():
            return False
        if self.worker is None:
            script = Path(__file__).resolve().parent / "converter_worker.py"
            settings = {"repo": C.CONVERTER_REPO, "revision": C.CONVERTER_REV, "checkpoint": C.CONVERTER_CKPT,
                        "config": C.CONVERTER_CONFIG, "diffusion_steps": C.DIFFUSION_STEPS,
                        "inference_cfg_rate": C.INFERENCE_CFG_RATE}
            self.worker = subprocess.Popen([str(self.python), str(script), str(self.dir), str(self.voice),
                                            json.dumps(settings)],
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
            assert json.loads(self.worker.stdout.readline())["ready"]
        self.worker.stdin.write(json.dumps({"source": str(source), "seed": int(seed), "out": str(out)}) + "\n")
        if not self.worker.stdout.readline():
            raise RuntimeError(f"the converter stopped on {source}")
        if not out.exists():
            raise RuntimeError(f"no conversion written for {source}")
        return True

    def close(self):
        if self.worker is not None:
            self.worker.stdin.close()
            self.worker.wait()
