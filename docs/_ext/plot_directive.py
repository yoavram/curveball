from __future__ import annotations

import os
import time

from docutils import nodes
from docutils.parsers.rst import Directive

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class PlotDirective(Directive):
    has_content = True

    def run(self):
        env = self.state.document.settings.env
        code = "\n".join(self.content)

        ns: dict[str, object] = {}
        plt.close("all")
        exec(code, ns)
        fig = plt.gcf()

        builder = env.app.builder
        source_images_dir = os.path.join(env.app.srcdir, "_images")
        os.makedirs(source_images_dir, exist_ok=True)
        basename = f"plot-{time.time_ns()}.png"
        filepath = os.path.join(source_images_dir, basename)
        fig.savefig(filepath, bbox_inches="tight")

        imgdir = builder.imgpath or "_images"
        uri = os.path.join(imgdir, basename)
        image_node = nodes.image(uri=uri)
        return [image_node]


def setup(app):
    app.add_directive("plot", PlotDirective)
    return {"parallel_read_safe": True}
