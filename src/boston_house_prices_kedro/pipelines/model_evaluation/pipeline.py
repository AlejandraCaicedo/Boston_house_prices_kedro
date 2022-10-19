"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
         node(
                func=evaluate_model,
                inputs=["predictions_train",
                        "y_train_model_input",
                        'params:train'],
                outputs="score_train",
                name="train_model_evaluation"
            ),
            node(
                func=evaluate_model,
                inputs=["predictions_test",
                        "y_test_validation",
                        'params:test'],
                outputs="score_test",
                name="test_model_evaluation"
            )
    ])
