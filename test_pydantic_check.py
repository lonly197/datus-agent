
from pydantic import BaseModel, Field, ValidationError
class M(BaseModel):
    s: str = Field(max_length=5)

try:
    M(s='aaaaaa')
    print('No Error')
except ValidationError:
    print('Error Raised')
