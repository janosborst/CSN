from flask import Flask, render_template, send_from_directory, redirect, url_for

app = Flask(__name__, static_folder='build/static', template_folder='build/')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/<path:path>')
def send_report(path):
    # remove the replace in next to lines later later <-- important !!!!!!!!
    print("files_report:", path)
    if path.endswith("manifest.json"):
        path = "build/manifest.json"
    if path.endswith(".jpeg"):
        return send_from_directory(".", str(path))
    else:
        return send_from_directory('build/', str(path))


@app.route('/csn/static/<path:path>')
def send_report2(path):
    # remove the replace in next to lines later later <-- important !!!!!!!!
    print("files_report2:", path)
    return send_from_directory('build/static/', str(path))


@app.route('/csn/datasets/<path:path>')
def send_report3(path):
    # remove the replace in next to lines later later <-- important !!!!!!!!
    print("files_report3:", path)
    return send_from_directory('build/datasets/', str(path))

