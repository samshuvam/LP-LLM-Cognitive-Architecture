import logging
import pandas as pd
import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.validator.validator import Validator
from ..config import GE_EXPECTATION_SUITE_NAME, GE_CONTEXT_ROOT_DIR

logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self, ge_suite_name: str = GE_EXPECTATION_SUITE_NAME, ge_context_root_dir: str = GE_CONTEXT_ROOT_DIR):
        self.suite_name = ge_suite_name
        # Initialize the Great Expectations Data Context
        self.context = gx.get_context(context_root_dir=ge_context_root_dir)
        logger.info(f"DataValidator initialized with suite: {self.suite_name}")

    def validate_interaction(self, interaction_dict: dict) -> bool:
        """
        Validates a single interaction record using Great Expectations.
        Args:
            interaction_dict (dict): The interaction data as a dictionary.
        Returns:
            bool: True if validation passes, False otherwise.
        """
        try:
            # Convert the dictionary to a Pandas DataFrame (single row)
            df = pd.DataFrame([interaction_dict])

            # Create a RuntimeBatchRequest for in-memory data
            batch_request = RuntimeBatchRequest(
                datasource_name="pandas_datasource", # This must be defined in your GE config
                data_connector_name="runtime_data_connector", # This must be defined in your GE config
                data_asset_name="temp_interactions",
                runtime_parameters={"batch_data": df},
                batch_identifiers={"default_identifier_name": "default_batch"},
            )

            # Get the validator associated with the expectation suite
            validator = self.context.get_validator(
                batch_request=batch_request,
                expectation_suite_name=self.suite_name,
            )

            # Run the validation
            result = validator.validate()

            # Log results for debugging
            if not result.success:
                logger.warning(f"Great Expectations validation failed for interaction: {interaction_dict}")
                logger.warning(f"Validation result details: {result.result}")
            else:
                logger.debug(f"Great Expectations validation passed for interaction: {interaction_dict}")

            return result.success

        except Exception as e:
            logger.error(f"Error during Great Expectations validation: {e}")
            # It's generally safer to fail validation on an error
            return False

# Global instance (or dependency injection can be used later)
# data_validator = DataValidator() # Commented out for now, initialize in main app