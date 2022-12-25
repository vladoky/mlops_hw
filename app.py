from flask import Flask
from flask_restx import Resource, Api, reqparse, fields
from model import Model

MAX_MODEL_NUM = 10
DATA_PATH = "mlops_hw1/data/insurance.csv"
MODELS_DICT = dict()

app = Flask(__name__)
app.config["BUNDLE_ERRORS"] = True
api = Api(app)

model_add = api.model(
    "Model.add.input", {
        "name":
        fields.String(
            required=True,
            title="Model name",
            description="Used as a key in local models storage; Must be unique;"
        ),
        "type":
        fields.String(required=True,
                      title="Model type",
                      description="Must be 'linear' or 'gradboost';"),
        "params":
        fields.String(
            required=True,
            title="Model params",
            description="Params to use in model.fit(); Must be valid dict;",
            default="{}")
    })

model_predict = api.model(
    "Model.predict.input", {
        "name":
        fields.String(required=True,
                      title="Model name",
                      description="Name of your existing trained model;"),
        "age":
        fields.Float(required=True,
                     title="age",
                     description="how old a person is;",
                     default=0),
        "bmi":
        fields.Float(required=True,
                     title="bmi",
                     description="body mass index: (mass)/(height^2) kg/m^2;",
                     default=0),
        "children":
        fields.Float(required=True,
                     title="children",
                     description="how many children does a person have;",
                     default=0),
        "sex":
        fields.String(required=True,
                     title="sex",
                     description="sex of a person: m/f;",
                     default=0),
        "smoker":
        fields.String(required=True,
                     title="smoker",
                     description="is a person smoking: true/false;",
                     default=0),
        "region":
        fields.String(required=True,
                     title="region",
                     description="where a person lives: northeast/northwest/southeast/southwest;",
                     default=0)
    })

parserRemove = reqparse.RequestParser(bundle_errors=True)
parserRemove.add_argument("name",
                          type=str,
                          required=True,
                          help="Name of a model you want to remove",
                          location="args")

parserTrain = reqparse.RequestParser(bundle_errors=True)
parserTrain.add_argument("name",
                         type=str,
                         required=True,
                         help="Name of a model you want to train",
                         location="args")

parserTrain.add_argument("dataset_path",
                         type=str,
                         required=True,
                         help="path to train dataset",
                         location="args")

parserTest = reqparse.RequestParser(bundle_errors=True)
parserTest.add_argument("name",
                         type=str,
                         required=True,
                         help="Name of a model you want to train",
                         location="args")

parserTest.add_argument("dataset_path",
                         type=str,
                         required=True,
                         help="path to train dataset",
                         location="args")


@api.route("/models/list")
class ModelList(Resource):
    @api.doc(responses={201: "Success"})
    def get(self):
        return {
            "models": {
                i: {
                    "type":
                    MODELS_DICT[i].model_type,
                    "is fitted":
                    MODELS_DICT[i].fitted,
                    "train_accuracy":
                    None if not MODELS_DICT[i].fitted else
                    MODELS_DICT[i].train_score,
                    "test_accuracy":
                    None if not MODELS_DICT[i].test_score else
                    MODELS_DICT[i].test_score,
                }
                for i in MODELS_DICT.keys()
            }
        }, 201


@api.route("/models/add")
class ModelAdd(Resource):
    @api.expect(model_add)
    @api.doc(
        responses={
            201: "Success",
            401: "'params' error; Params must be a valid json or dict",
            402:
            "Error while initializing model; See description for more info",
            403: "Model with a given name already exists",
            408: "The max number of models has been reached"
        })
    def post(self):
        __name = api.payload["name"]
        __type = api.payload["type"]
        __params = api.payload["params"]

        try:
            __params = eval(__params)
        except Exception as e:
            return {
                "status": "Failed",
                "message":
                "'params' error; Params must be a valid json or dict"
            }, 401

        if len(MODELS_DICT) >= MAX_MODEL_NUM:
            return {
                "status":
                "Failed",
                "message":
                "The max number of models has been reached; You must delete one before creating another"
            }, 408

        if __name not in MODELS_DICT.keys():
            try:
                MODELS_DICT[__name] = Model(__type)
                return {"status": "OK", "message": "Model created!"}, 201
            except Exception as e:
                return {
                    "status": "Failed",
                    "message": getattr(e, "message", repr(e))
                }, 402
        else:
            return {
                "status": "Failed",
                "message": "Model with a given name already exists"
            }, 403


@api.route("/models/remove")
class ModelRemove(Resource):
    @api.expect(parserRemove)
    @api.doc(responses={
        201: "Success",
        404: "Model with a given name does not exist"
    })
    def delete(self):
        __name = parserRemove.parse_args()["name"]
        if __name not in MODELS_DICT.keys():
            return {
                "status": "Failed",
                "message": "Model with a given name does not exist"
            }, 404
        else:
            MODELS_DICT.pop(__name)
            return {"status": "OK", "message": "Model removed!"}, 201


@api.route("/models/train")
class ModelTrain(Resource):
    @api.expect(parserTrain)
    @api.doc(
        responses={
            201: "Success",
            404: "Model with a given name does not exist",
            406: "Error while training model; See description for more info"
        })
    def get(self):
        __name = parserTrain.parse_args()["name"]
        __dataset_path = parserTrain.parse_args()["dataset_path"]
        if __name not in MODELS_DICT.keys():
            return {
                "status": "Failed",
                "message": "Model with a given name does not exist!"
            }, 404
        else:
            try:
                MODELS_DICT[__name].fit(__dataset_path)
                return {"status": "OK", "message": f"Train score {MODELS_DICT[__name].train_score}"}, 201
            except Exception as e:
                return {
                    "status": "Failed",
                    "message": getattr(e, "message", repr(e))
                }, 406

@api.route("/models/test")
class ModelTest(Resource):
    @api.expect(parserTest)
    @api.doc(
        responses={
            201: "Success",
            404: "Model with a given name does not exist",
            406: "Error while testing model; See description for more info"
        })
    def get(self):
        __name = parserTrain.parse_args()["name"]
        __dataset_path = parserTrain.parse_args()["dataset_path"]
        if __name not in MODELS_DICT.keys():
            return {
                "status": "Failed",
                "message": "Model with a given name does not exist!"
            }, 404
        else:
            try:
                MODELS_DICT[__name].test(__dataset_path)
                return {"status": "OK", "message": f"Test score {MODELS_DICT[__name].test_score}"}, 201
            except Exception as e:
                return {
                    "status": "Failed",
                    "message": getattr(e, "message", repr(e))
                }, 406

@api.route("/models/predict")
class ModelPredict(Resource):
    @api.expect(model_predict)
    @api.doc(
        responses={
            201: "Success",
            404: "Model with a given name does not exist",
            407: "Error while predicting result; See description for more info"
        })
    def post(self):
        __name = api.payload["name"]
        __params = api.payload
        __params.pop("name")
        print("BLAAAH")
        if __name not in MODELS_DICT.keys():
            return {
                "status": "Failed",
                "message": "Model with a given name does not exist!"
            }, 404
        else:
            try:
                pred = MODELS_DICT[__name].predict(__params)
                return {"result": f"predicted value: {pred}"}, 201
            except Exception as e:
                return {
                    "status": "Failed",
                    "message": getattr(e, "message", repr(e))
                }, 407


if __name__ == "__main__":
    app.run(debug=True)
