from pathlib import Path

import click
from flask import Flask
from flask import redirect
from flask import render_template
from flask import url_for
from forencics.controller import Controller
import yaml


app = Flask(__name__)
controller = Controller()


@app.route("/", methods=["GET", "POST"])
def start_page():
    return redirect(url_for("main_page"))


@app.route("/main", methods=["GET", "POST"])
def main_page():
    # TODO video load and analysis
    return render_template("main.html")


@app.route("/model", methods=["GET", "POST"])
def model_page():
    # TODO add 3d model and make it spin
    return render_template(
        "model.html",
        # stage_content=f"stages/stage{team_info.stage}.html",
        # team_info=team_info,
        # stage=stage,
        # stage_start_time=stage_start_time,
        # standings=standings_view,
        # hints=hints,
    )


@app.route("/pres")
def pres_page():
    # TODO make presentation there
    return render_template(
        "pres.html",
    )


@app.route("/info")
def info_page():
    # TODO add CVs and everything else
    return render_template(
        "info.html",
    )


@click.option(
    "-c",
    "--config_path",
    required=True,
    type=click.Path(dir_okay=False, exists=True, path_type=Path),
)
@click.option(
    "-p",
    "--port",
    type=int,
    default=5000,
)
def run_server(config_path: Path, port: int):
    with open(config_path) as f:
        config = yaml.full_load(f)
    app.run(host="0.0.0.0", port=port, debug=True, extra_files=[config_path])


if __name__ == "__main__":
    click.command(run_server)()
