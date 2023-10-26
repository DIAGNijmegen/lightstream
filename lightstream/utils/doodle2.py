from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class MyDataClass:
    name: str
    age: int


# Serialize
obj = MyDataClass("John", 25)
print(obj)
serialized = obj.to_json()
print(serialized)

# Deserialize
restored_obj = MyDataClass.from_json(serialized)
print(restored_obj)
