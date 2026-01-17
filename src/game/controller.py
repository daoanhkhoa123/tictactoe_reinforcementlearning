from typing import Generic, Tuple, TypeVar, final

from numpy.typing import NDArray

ModelInputT = TypeVar("ModelInputT")
ModelOutputT = TypeVar("ModelOutputT")


class BaseController(Generic[ModelInputT, ModelOutputT]):
    """
    BaseController defines the canonical decision pipeline for a player/controller.

    The controller follows a fixed three-stage process:

        1. pre_processing(input_state)  -> ModelInputT
        2. model_call(model_input)       -> ModelOutputT
        3. post_processing(model_output) -> (row, col)

    Subclasses are expected to override one or more of the stage methods
    (`pre_processing`, `model_call`, `post_processing`) to implement different
    decision strategies (human input, random policy, rule-based AI, neural model, etc).

    The `decide` method is final and MUST NOT be overridden. This guarantees a
    consistent execution order across all controllers.
    """

    @final
    def decide(self, input_state: NDArray) -> Tuple[int, int]:
        """
        Execute the full decision pipeline and return a board coordinate.

        Parameters
        ----------
        input_state : NDArray
            The current board state. Conventionally, 0 represents an empty cell.

        Returns
        -------
        Tuple[int, int]
            A valid move as (row, col).

        Notes
        -----
        This method is marked as `final` and must not be overridden.
        Customization must be done by overriding the stage methods.
        """
        model_input = self.pre_processing(input_state)
        model_output = self.model_call(model_input)
        return self.post_processing(model_output)

    def pre_processing(self, input_state: NDArray) -> ModelInputT:
        """
        Transform the raw board state into a model-specific representation.
        """
        return input_state  # type: ignore

    def model_call(self, model_input: ModelInputT) -> ModelOutputT:
        """
        Core decision logic of the controller.
        """
        raise NotImplementedError("Model must be implemented")

    def post_processing(self, model_output: ModelOutputT) -> Tuple[int, int]:
        """
        Convert the model output into a concrete board coordinate.
        """
        raise model_output # type: ignore
