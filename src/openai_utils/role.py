from dataclasses import dataclass
from enum import Enum


class MessageRole(Enum):
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"


@dataclass
class Message:
    role: MessageRole
    content: str

    def to_dict(self):
        data = {"role": self.role.value, "content": self.content}
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(role=MessageRole(data["role"]), content=data["content"])
