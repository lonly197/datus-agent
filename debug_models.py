
import sys
import datus.api.models
from datus.api.models import RunWorkflowRequest

print(f"sys.path: {sys.path}")
print(f"datus.api.models file: {datus.api.models.__file__}")

with open(datus.api.models.__file__, 'r') as f:
    content = f.read()
    if 'max_length=256' in content:
        print("File content HAS max_length")
    else:
        print("File content DOES NOT HAVE max_length")

print(f"Metadata: {RunWorkflowRequest.model_fields['workflow'].metadata}")

from pydantic import BaseModel, Field
class LocalModel(BaseModel):
    f: str = Field(max_length=5)

print(f"LocalModel Metadata: {LocalModel.model_fields['f'].metadata}")

