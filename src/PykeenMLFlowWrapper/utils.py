from .wrapper import PykeenWrapper
import os
import mlflow
from sys import version_info
import torch
import pykeen

def save_model(pipeline_result, save_dir ,model_name, param_tracker=None):
    "Save Pykeen model and register it on MLFlow"


    save_path=os.path.join(save_dir,f"{model_name}_saved")
    pipeline_result.save_to_directory(save_path)


    
    
    model_path = os.path.join(save_path, "trained_model.pkl")
    triples_factory_path = os.path.join(save_path, "training_triples")

    artifacts = {"pykeen_triples_path": triples_factory_path,
                 "pykeen_model_path": model_path}

    print(artifacts)

    
    PYTHON_VERSION = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    conda_env = {
        'channels': ['defaults'],
        'dependencies': [
        'python={}'.format(PYTHON_VERSION),
        'pip',
        {
            'pip': [
            'mlflow',
            'torch',
            'pykeen',
            ],
        },
        ],
        'name': f'pykeen_{model_name}_env'
    }

    with mlflow.start_run() as run:

        if param_tracker is not None:
            mlflow.log_params(pykeen.utils.flatten_dictionary(param_tracker.configuration))

        mlflow.log_metrics(pipeline_result.metric_results.to_flat_dict())

        return mlflow.pyfunc.log_model(
            artifact_path=model_name,
            python_model=PykeenWrapper(),
            #code_path=["./your_code_path"],
            conda_env=conda_env,
            artifacts=artifacts,
            registered_model_name=model_name
        )


def load_model(model_name: str = "model_name", stage: str = "None"):

    loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")
    return loaded_model