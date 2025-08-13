import os, sys
sys.path.append(os.getcwd())

from sklearn.cluster import DBSCAN, HDBSCAN
from pathlib import Path
import pandas as pd

import json, math, os, io
import pandas as pd
import numpy as np
import logging

from CSN_utils import ImageSpriteGenerator, SimplePlot,PCAGenerator, UMAPGenerator,TSNEGenerator
from CSN_utils import HistogramGenerator, Utils
from itertools import combinations
import distinctipy
from flask import Flask, render_template, send_from_directory

import shutil

logging.basicConfig(level=logging.DEBUG)

class CSN:
    def __init__(self, example_data,
                 filenameColumn = "filename",
                 classColumns= ("filename",),
                 infoColumns = None,
                 sliderColumns = None,
                 filterColumns=None,
                 description= "Buchkindheiten Digital"
                 ):
        self.example_data = Path(example_data)
        self.datasetTitle = self.example_data.stem
        self.description = description

        self.imageLocation = self.example_data / "images"
        self.metadataLocation = self.example_data / "metadata.csv"
        self.metadata = pd.read_csv(self.metadataLocation, skipinitialspace=True)
        self.build_prefix = Path("build/datasets")

        self.embeddingLocations = list(x.name for x in (self.example_data).glob("*.csv") if x.name not in ["images", "metadata.csv"])

        self.imageWebLocation = str(self.example_data / "images/") + "/"
        self.logger = logging.getLogger(self.__class__.__name__)



        self.filenameColumn = filenameColumn
        assert pd.api.types.is_string_dtype(self.metadata[filenameColumn]) and self.metadata[filenameColumn].str.endswith((
                                                                        ".jpg", ".JPEG", "JPG", ".jpeg", ".png", ".PNG",)).all()

        self.classColumns  = classColumns
        self.infoColumns   = (infoColumns if infoColumns is not None else [mf for mf in self.metadata.columns if mf != "index"])
        self.sliderColumns = sliderColumns if sliderColumns is not None else [mf for mf in self.metadata.columns if pd.api.types.is_numeric_dtype(self.metadata[mf]) and mf != "index" and not mf.startswith("Unnamed")]
        self.filterColumns = filterColumns if filterColumns is not None else [mf for mf in self.metadata.columns if pd.api.types.is_string_dtype(self.metadata[mf]) and mf != 'URL' and not mf.startswith("Unnamed")]

        self.foldername = (str(self.example_data)).lower().replace(" ", "_")
        self.logger.info(f'Creating new dataset directory: build/datasets/{self.foldername}...')
        (Path("build/datasets/") / self.foldername).mkdir(exist_ok=True, parents=True)

        self.mappings = []

    def _load_data(self):
        imagNumb = len(os.listdir(self.imageLocation))
        print(f'found {imagNumb} files in {self.imageLocation}')

        metaNumb = len(self.metadata)
        print(f'found {metaNumb} entries in {self.metadataLocation}')

        self.embeddings = {}
        for name in self.embeddingLocations:
            self.logger.info(f"Loading {name}")
            e = pd.read_csv(self.example_data / name, skipinitialspace=True)
            e = e.loc[:, e.columns != 'id']
            e = e.loc[:, e.columns != 'ID']
            e = e.loc[:, e.columns != 'Game']
            e = e.loc[:, e.columns != 'game']
            self.embeddings[Path(name).stem] = e

            vecNumb = len(e)
            self.logger.info(f'found {vecNumb} entries in {self.example_data / name}')

            if metaNumb == vecNumb:
                if vecNumb <= imagNumb:
                    self.logger.info("Looks ok.")
                    self.logger.info("")
                    self.logger.info(f'Embedding file contains {vecNumb} vectors in {len(e.columns)} dimensions.')
                    self.logger.info("Metadata Head:")
                else:
                    self.logger.info("")
                    self.logger.info("ERROR: number of images is smaller than number of vectors")

        if metaNumb <= imagNumb:
            self.logger.info("Looks ok.")
        else:
            self.logger.error("ERROR: number of images and metadata elements don't match!")




    def _generate_data(self):
        # generate sprite sheets
        target = len(list((self.build_prefix / Path(self.foldername) ).glob(f"tile_*.png")))
        if target == 0:
            self.sprite_generator = ImageSpriteGenerator(self.foldername, spriteSize=2048, spriteRows=32,
                                                    imageFolder=self.imageLocation,
                                                    files=self.metadata[self.filenameColumn]).generate()

    def _generate_meta_plots(self):
        for a, b in combinations(self.sliderColumns,2):
            result = SimplePlot(self.foldername, A=a, B=b, metadata=self.metadata)
            filename = (a + "_" + b).replace(" ", "")
            self.mappings.append({"name": filename, "file": f"{filename}.json"})

    def _generate_projections_PCA(self, name, add_slider = False):
        target = self.build_prefix /  Path(self.foldername ) / f"PCA-{name}.json"
        PCAEembedding = PCAGenerator(self.foldername, scale=True, data=self.embeddings[name], components=2).generate()
        shutil.copyfile(self.build_prefix / Path(self.foldername ) / "PCA.json", target, )
        self.logger.info(f"Moved {self.build_prefix /  Path(self.foldername ) / 'PCA.json'} to  {self.build_prefix /  Path(self.foldername) / f'PCA-{name}.json'}")
        self.mappings.append({"name": f"PCA {name}", "file": f"PCA-{name}.json"})

        # add columns to metadata for each component
        self.metadata[f"{name}-PC1"] = PCAEembedding[:, 0]
        self.metadata[f"{name}-PC2"] = PCAEembedding[:, 1]

        # add slider for each component
        if add_slider:
            self.sliderColumns.append(f"{name}-PC1")
            self.sliderColumns.append(f"{name}-PC2")


    def _generate_projections_UMAP(self, name,n_neighbors=15, min_dist=0.18, metric="correlation", verbose=True, add_slider = False):
        target = self.build_prefix /  Path(self.foldername ) / f"UMAP-{name}-{metric}.json"
        if not target.exists():
            r = UMAPGenerator(self.foldername, data=self.embeddings[name],
                                           n_neighbors=n_neighbors, min_dist=min_dist, metric=metric,
                                           verbose=verbose,).generate( metric=metric)
            self.logger.info("Clustering the UMAP")
            db_labels = DBSCAN().fit_predict(r)
            self.metadata[f"DBSCAN-UMAP-{name}-{metric}"] = db_labels
            self.classColumns.append(f"DBSCAN-UMAP-{name}-{metric}")
            shutil.move(self.build_prefix / Path(self.foldername ) / "UMAP.json",target, )
            self.logger.info(f"Moved {self.build_prefix / Path(self.foldername ) / 'UMAP.json'} to  {target}")
        else:
            self.logger.info(f"{target} already exists")
        self.mappings.append({"name": f"UMAP-{name}-{metric}", "file": f"UMAP-{name}-{metric}.json"})

    def _generate_projects_TSNE(self, name, n_components = 2, verbose = 1, random_state = 123,metric="euclidean"):
        target = self.build_prefix /  Path(self.foldername ) / f"TSNE-{name}.json"
        if not target.exists():
            _ = TSNEGenerator(self.foldername, data=self.embeddings[name],
                                           n_components=n_components, verbose=verbose, random_state=random_state, ).generate(metric=metric)
            shutil.move(self.build_prefix / Path(self.foldername ) / "TSNE.json", target, )
            self.logger.info(f"Moved {self.build_prefix / Path(self.foldername ) / 'TSNE.json'} to  {target}")
        else:
            self.logger.info(f"{target} already exists")
        self.mappings.append({"name": f"t-SNE-{name}", "file": "TSNE.json"})

    def _generateSliders(self):
        # check if sliderCols exists
        self.logger.info("Generate Sliders")
        sliderCols = list(self.sliderColumns)
        self.sliderInfoDict = {x:x for x in self.sliderColumns}
        self.sliderNameDict = {x:x for x in self.sliderColumns}

        if len(sliderCols) > 0:
            self.sliderColorDict = {}
            colors = distinctipy.get_colors(len(sliderCols), pastel_factor=1)
            for i, sliderName in enumerate(sliderCols):
                self.sliderColorDict[sliderName] = distinctipy.get_hex(colors[i])
        else:
            print("No Cluster fields selected!")

    def _generateClassColors(self):
        self.logger.info("Generate Class Colors")

        if len(self.classColumns) > 0:
            classColorDict = {}
            amount = len(self.classColumns)
            allClasses = {}
            for className in self.classColumns:
                clusters = self.metadata[className].unique()
                allClasses[className] = len(clusters)
            length = max(allClasses.values())
            self.allColors = {}
            colors = distinctipy.get_colors(length)
            col = 5
            row = math.ceil(length / col)
            i = 0
            rows = []
            for r in (range(0, col)):
                newRow = []
                for c in range(0, row):
                    if i < len(colors):
                        self.allColors[i] = distinctipy.get_hex(colors[i])
                        i += 1
        else:
            self.allColors = False
            print("No cluster was selected.")

    def _create_metadata_json(self,):
        self.logger.info("Generate metadata JSON")

        sliderCols = self.sliderColumns

        metadataColumns = set(
            list(self.infoColumns) + sliderCols + list(self.filterColumns) + list(self.classColumns))

        metadataColumns.add(self.filenameColumn)
        metadata = self.metadata[list(metadataColumns)]
        Utils.write_metadata(self.foldername, metadata, self.filenameColumn)

    def _generate_histograms(self):
        self.logger.info("Generate Histograma")

        sliderCols = list(self.sliderColumns)
        BarChartData = HistogramGenerator(self.foldername, data=self.metadata, selection=sliderCols, bucketCount=50).generate()

    def save_datasetsJSON(self, datasetsJSON):
        self.logger.info("Generate datasets_config.json")

        with open(f'build/datasets/datasets_config.json', "w") as fd:
            json.dump(datasetsJSON, fd)
        print("saved datasets_config.json")

    def make_default(self, datasetsJSON, DEFAULT):
        datasetsJSON["default"] = DEFAULT
        print(f"changed default dataset to {datasetsJSON['data'][DEFAULT]['name']}")
        self.save_datasetsJSON(datasetsJSON)

    def _save_configuration(self):
        self.logger.info("Save configuration")

        sliderCols = list(self.sliderColumns)
        sliderSetting = []

        for k in sliderCols:
            dtype = 'float'
            if pd.api.types.is_integer_dtype(self.metadata[k]):
                dtype = 'int'
            slider = {"id": k, "title": self.sliderNameDict[k], "info": self.sliderInfoDict[k], "typeNumber": dtype,
                      "color": self.sliderColorDict[k], "min": self.metadata[k].min(), "max": self.metadata[k].max()}
            sliderSetting.append(slider)
        searchFields = []
        for k in self.filterColumns:
            filter = {"columnField": k, "type": "selection"}
            searchFields.append(filter)

        if self.allColors:
            clusters = {"clusterList": list(self.classColumns),
                            "clusterColors": [self.allColors[g] for g in self.allColors]}
        else:
            clusters = {"clusterList": [], "clusterColors": []}

        configData = Utils.write_config(directory=self.foldername, title=self.datasetTitle, description=self.description,
                                        mappings=self.mappings, clusters=clusters, total=len(self.metadata),
                                        sliderSetting=sliderSetting, infoColumns=self.infoColumns,
                                        searchFields=searchFields, imageWebLocation=self.imageWebLocation,
                                        spriteRows=32, squareSize=2048, spriteSize=64, spriteDir=None)
        newDataset = {'name': self.datasetTitle, 'directory': self.foldername}

        datasetsJSON = {"default": 0, "data": [newDataset]}
        self.save_datasetsJSON(datasetsJSON)

    def initialize(self):
        self._load_data()
        self._generate_data()
        self._generate_meta_plots()


        # self._generate_projections_UMAP("dHashEmbedding", metric="euclidean",)
        # self._generate_projections_UMAP("clip-vit-base-patch32", metric="euclidean")
        for name in self.embeddings.keys():
            self._generate_projections_UMAP(name, metric="euclidean",min_dist=0.01)
            self._generate_projections_UMAP(name, metric="cosine",min_dist=0.01)
            self._generate_projections_UMAP(name, metric="correlation",min_dist=0.01)


        self._generateSliders()
        self._generateClassColors()
        self._create_metadata_json()
        self._generate_histograms()
        self._save_configuration()


def run_app():
    from flask import Flask, render_template, send_from_directory, url_for

    app = Flask(__name__, static_folder='build/static', template_folder='build/')


    @app.route('/')
    def home():
        return render_template(('index.html'))


    @app.route('/<path:path>')
    def send_report(path):
        # remove the replace in next to lines later later <-- important !!!!!!!!
        print("files_report:", path)
        if path == "manifest.json":
            path = "manifest.json"
        if path.endswith(".jpeg"):
            return send_from_directory(".", str(path))
        else:
            return send_from_directory('build/', str(path))

    @app.route('/static/<path:path>')
    def send_report2(path):
        # remove the replace in next to lines later later <-- important !!!!!!!!
        print("files_report2:", path)
        return send_from_directory('build/static/', str(path))

    @app.route('/datasets/<path:path>')
    def send_report3(path):
        # remove the replace in next to lines later later <-- important !!!!!!!!
        print("files_report3:", path)
        return send_from_directory('build/datasets/', str(path))
    app.run(debug=False, port=8000,)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all-scenes", help="Title of the dataset")
    args = parser.parse_args()
    print(args)
    CSN(
        args.dataset,
        filenameColumn="filename",
        classColumns=["Jahr"],#, "spielen", "Spielzeug", "lesen", "unterrichten", "schreiben", "Interaktion mit Tieren"],
        infoColumns= ["filename", "ID", "Title", "Jahr", "Verlag", "Ort", "Verfasser", "Weitere Verfasser", "url"],
        sliderColumns=["Jahr"],
        filterColumns=["Jahr"],
        description="ChildBookIllu-Scenes"
    ).initialize()

    run_app()