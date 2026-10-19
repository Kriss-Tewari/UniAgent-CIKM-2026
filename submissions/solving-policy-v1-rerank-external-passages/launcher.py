"""Configure the packaged Java runtime, then execute the unchanged agent."""
from pathlib import Path
import runpy

import pyterrier as pt

pt.terrier.set_version("5.11")
pt.terrier.set_helper_version("0.0.8")
pt.java.mavenresolver.offline()
pt.java.set_memory_limit(8192)

runpy.run_path(
    str(Path(__file__).with_name("predict.py")),
    run_name="__main__",
)
