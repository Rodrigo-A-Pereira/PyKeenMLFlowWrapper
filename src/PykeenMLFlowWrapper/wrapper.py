import pykeen
import mlflow.pyfunc

class PykeenWrapper(mlflow.pyfunc.PythonModel):
    """
    Class load and use Pykeen models
    """

    def load_context(self, context):
        """This method is called when loading an MLflow model with pyfunc.load_model(), as soon as the Python Model is constructed.
        Args:
            context: MLflow context where the model artifact is stored.
        """

        import torch

        #print(context.artifacts["pykeen_triples_path"])
        self.model = torch.load(context.artifacts["pykeen_model_path"])

        self.triple_factory = pykeen.triples.TriplesFactory.from_path_binary(context.artifacts["pykeen_triples_path"])

    def predict(self, context, model_input):
        """Make Predictions.
        Args:
            context ([type]): MLflow context where the model artifact is stored.
            model_input ([type]): the input data to fit into the model.
        Returns:
            [type]: the loaded model artifact.
        """

        predictions = [] 
        for tup in model_input:
            pred = self.model.get_tail_prediction_df(tup[0],tup[1], triples_factory = self.triple_factory)
            
            predictions.append(pred["tail_label"].values)

        return predictions