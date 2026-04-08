from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import uuid
import time

class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str
    content: str
    timestamp: float = Field(default_factory=time.time)
    ground_truth_priority: Priority
    keywords: List[str] = []

class Observation(BaseModel):
    current_message: Message
    inbox_position: int
    total_messages: int
    pinned_high_priority_ids: List[str]

class Action(BaseModel):
    action_type: str = Field(..., description="set_priority | pin_message | next")
    priority: Optional[Priority] = None
    message_id: Optional[str] = None

class SmartInboxEnv:
    def __init__(self, messages: List[Message], max_steps: int = 30):
        self.messages = messages
        self.max_steps = max_steps
        self.current_index = 0
        self.pinned_ids = []
        self.actions_taken = []
        self.done = False
        self.steps = 0

    def reset(self) -> Observation:
        self.current_index = 0
        self.pinned_ids = []
        self.actions_taken = []
        self.done = False
        self.steps = 0
        return self._get_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        if self.done:
            raise RuntimeError("Episode already done")
        reward = 0.0
        info = {"action": action.dict()}

        current_msg = self.messages[self.current_index]

        if action.action_type == "set_priority":
            if action.priority is None:
                reward -= 0.1
            else:
                if action.priority == current_msg.ground_truth_priority:
                    reward += 1.0
                else:
                    reward -= 0.3
                self.actions_taken.append(("set_priority", action.priority, current_msg.id))

        elif action.action_type == "pin_message":
            if action.message_id and action.message_id not in self.pinned_ids:
                self.pinned_ids.append(action.message_id)
                msg = next((m for m in self.messages if m.id == action.message_id), None)
                if msg and msg.ground_truth_priority == Priority.HIGH:
                    reward += 0.2
                else:
                    reward -= 0.1
            else:
                reward -= 0.05

        elif action.action_type == "next":
            self.current_index += 1
            reward += 0.05

        else:
            reward -= 0.2

        self.steps += 1
        if self.current_index >= len(self.messages) or self.steps >= self.max_steps:
            self.done = True
            all_high_ids = [m.id for m in self.messages if m.ground_truth_priority == Priority.HIGH]
            pinned_high = [pid for pid in self.pinned_ids if pid in all_high_ids]
            if all_high_ids:
                reward += 0.5 * (len(pinned_high) / len(all_high_ids))

        return self._get_observation(), reward, self.done, info

    def state(self) -> dict:
        return {
            "current_index": self.current_index,
            "pinned_ids": self.pinned_ids,
            "steps": self.steps,
            "done": self.done
        }

    def _get_observation(self) -> Observation:
        if self.current_index < len(self.messages):
            return Observation(
                current_message=self.messages[self.current_index],
                inbox_position=self.current_index + 1,
                total_messages=len(self.messages),
                pinned_high_priority_ids=self.pinned_ids
            )
        # fallback (should not happen when done)
        dummy = Message(sender="system", content="", ground_truth_priority=Priority.LOW)
        return Observation(current_message=dummy, inbox_position=0, total_messages=0, pinned_high_priority_ids=[])