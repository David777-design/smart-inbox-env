# environment.py
from typing import List, Dict, Optional, Any, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum
import time
import random
import uuid

# -------------------- Enums --------------------
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

    @validator('content')
    def content_not_empty(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Message content cannot be empty')
        return v

class Observation(BaseModel):
    current_message: Message
    inbox_position: int          # 1‑based index
    total_messages: int
    pinned_high_priority_ids: List[str]

class Action(BaseModel):
    action_type: str = Field(..., description="set_priority | pin_message | next")
    priority: Optional[Priority] = None
    message_id: Optional[str] = None

# -------------------- Environment --------------------
class SmartInboxEnv:
    """
    Simulates a message inbox where an AI agent must triage messages.
    The agent can set priority, pin important messages, and move to the next message.
    """
    def __init__(self, messages: List[Message], max_steps: int = 30):
        self.messages = messages
        self.max_steps = max_steps
        self.current_index = 0
        self.pinned_ids = []
        self.actions_taken = []   # list of (action_type, value, message_id)
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
            raise RuntimeError("Episode already finished. Call reset().")
        reward = 0.0
        info = {"action": action.dict()}

        # Validate action type
        if action.action_type not in ["set_priority", "pin_message", "next"]:
            reward -= 0.2
            info["error"] = f"Invalid action type: {action.action_type}"
            return self._get_observation(), reward, self.done, info

        current_msg = self.messages[self.current_index]

        # ---- set_priority ----
        if action.action_type == "set_priority":
            if action.priority is None:
                reward -= 0.1
                info["error"] = "Missing priority"
            else:
                if action.priority == current_msg.ground_truth_priority:
                    reward += 1.0
                    info["correct"] = True
                else:
                    # Partial penalty
                    reward -= 0.3
                    info["correct"] = False
                self.actions_taken.append(("set_priority", action.priority, current_msg.id))

        # ---- pin_message ----
        elif action.action_type == "pin_message":
            if action.message_id is None:
                reward -= 0.1
                info["error"] = "Missing message_id"
            else:
                if action.message_id not in self.pinned_ids:
                    self.pinned_ids.append(action.message_id)
                    # Find the message to check its true priority
                    msg = next((m for m in self.messages if m.id == action.message_id), None)
                    if msg and msg.ground_truth_priority == Priority.HIGH:
                        reward += 0.2
                        info["bonus"] = "pinned high priority"
                    else:
                        reward -= 0.1
                        info["penalty"] = "pinned non‑high priority"
                else:
                    reward -= 0.05  # already pinned
                self.actions_taken.append(("pin_message", action.message_id, action.message_id))

        # ---- next ----
        elif action.action_type == "next":
            self.current_index += 1
            reward += 0.05   # small reward for progressing
            self.actions_taken.append(("next", None, None))

        # Episode termination
        self.steps += 1
        if self.current_index >= len(self.messages) or self.steps >= self.max_steps:
            self.done = True
            # End‑of‑episode bonus: correctly pinned all high‑priority messages
            all_high_ids = [m.id for m in self.messages if m.ground_truth_priority == Priority.HIGH]
            pinned_high = [pid for pid in self.pinned_ids if pid in all_high_ids]
            if len(pinned_high) == len(all_high_ids) and len(all_high_ids) > 0:
                reward += 1.0
                info["end_bonus"] = "pinned all high priority"
            elif pinned_high:
                reward += 0.5 * (len(pinned_high) / len(all_high_ids))
                info["end_bonus"] = f"pinned {len(pinned_high)}/{len(all_high_ids)} high priority"

        return self._get_observation(), reward, self.done, info

    def state(self) -> dict:
        return {
            "current_index": self.current_index,
            "pinned_ids": self.pinned_ids,
            "actions_taken": self.actions_taken,
            "done": self.done,
            "steps": self.steps,
            "total_messages": len(self.messages)
        }

    def _get_observation(self) -> Observation:
        if self.current_index < len(self.messages):
            return Observation(
                current_message=self.messages[self.current_index],
                inbox_position=self.current_index + 1,
                total_messages=len(self.messages),
                pinned_high_priority_ids=self.pinned_ids
            )
        else:
            # Fallback – should not be called when done
            dummy = Message(
                sender="system",
                content="No more messages",
                ground_truth_priority=Priority.LOW
            )
            return Observation(
                current_message=dummy,
                inbox_position=len(self.messages),
                total_messages=len(self.messages),
                pinned_high_priority_ids=self.pinned_ids
            )