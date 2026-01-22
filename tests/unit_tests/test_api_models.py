
import unittest
from pydantic import ValidationError
import datus.api.models
from datus.api.models import RunWorkflowRequest, ChatResearchRequest

class TestApiModels(unittest.TestCase):
    def test_run_workflow_request_validation(self):
        print(f"DEBUG: datus.api.models file: {datus.api.models.__file__}")
        print(f"DEBUG: RunWorkflowRequest module: {RunWorkflowRequest.__module__}")
        print(f"DEBUG: RunWorkflowRequest fields: {RunWorkflowRequest.model_fields['workflow']}")
        print(f"DEBUG: Metadata: {RunWorkflowRequest.model_fields['workflow'].metadata}")
        """Test RunWorkflowRequest field validation."""
        # Valid request
        valid_data = {
            "workflow": "nl2sql",
            "namespace": "default",
            "task": "Show me sales",
            "task_id": "task_123"
        }
        req = RunWorkflowRequest(**valid_data)
        self.assertEqual(req.workflow, "nl2sql")

        # Invalid workflow (too long)
        invalid_data = valid_data.copy()
        invalid_data["workflow"] = "a" * 300
        try:
            req = RunWorkflowRequest(**invalid_data)
            print(f"Created request with workflow length: {len(req.workflow)}", flush=True)
        except ValidationError:
            print("Caught ValidationError in try block", flush=True)
            pass
        
        print(f"Invalid data workflow len: {len(invalid_data['workflow'])}", flush=True)
        try:
            RunWorkflowRequest(**invalid_data)
            self.fail("Did not raise ValidationError on second attempt")
        except ValidationError:
            pass # Success

        print(f"DEBUG: RunWorkflowRequest task field: {RunWorkflowRequest.model_fields['task']}")
        print(f"DEBUG: Task Metadata: {RunWorkflowRequest.model_fields['task'].metadata}")
        
        # Invalid task (too long)
        invalid_data = valid_data.copy()
        invalid_data["task"] = "a" * 10001
        with self.assertRaises(ValidationError):
            RunWorkflowRequest(**invalid_data)

    def test_chat_research_request_validation(self):
        """Test ChatResearchRequest field validation."""
        # Valid request
        valid_data = {
            "namespace": "default",
            "task": "Analyze growth"
        }
        req = ChatResearchRequest(**valid_data)
        self.assertEqual(req.namespace, "default")

        # Invalid prompt (too long)
        invalid_data = valid_data.copy()
        invalid_data["prompt"] = "a" * 50001
        with self.assertRaises(ValidationError):
            ChatResearchRequest(**invalid_data)

if __name__ == "__main__":
    unittest.main()
