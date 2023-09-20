from pathlib import Path

import click
import os
from flask import Flask
from flask import redirect
from flask import render_template
from flask import url_for
from flask import flash, request
from flask import send_from_directory
from flask import session
from forencics.controller import Controller
from werkzeug.utils import secure_filename
import yaml


# TODO add proper uploads folser if needed 
UPLOAD_FOLDER = 'uploads' 
# TODO add proper downloads folser if needed 
DOWNLOAD_FOLDER = 'uploads'
# TODO add all appripriate file formats
ALLOWED_EXTENSIONS = {'mp4'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
# TODO set limit there. 16 * 1000 * 1000 - 16mb 
controller = Controller()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def start_page():
    return redirect(url_for("main_page"))


@app.route("/main", methods=["GET", "POST"])
def main_page():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            session["uploaded_file"] = filename
            controller.user_uploaded_file(filename)
    
    return render_template(
        "main.html",
    )


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

@app.route('/remove_file') 
def remove_file(): 
    # TODO remove file from processing
    del session['uploaded_file']
    print("ebu")
    return {'result': 'success'} 


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

    app.config["SECRET_KEY"] = config["flask_secret_key"]
   
    app.run(host="0.0.0.0", port=port, debug=True, extra_files=[config_path])


if __name__ == "__main__":
    click.command(run_server)()
