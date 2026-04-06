# tasks.py
from environment import Message, Priority
import time

def load_task(task_name: str):
    """
    Returns a tuple: (messages_list, grader_function, description, max_steps)
    """
    if task_name == "easy":
        messages = [
            Message(
                sender="boss",
                content="URGENT: Client deadline moved to today! Please respond immediately.",
                ground_truth_priority=Priority.HIGH,
                keywords=["urgent", "deadline", "immediately"]
            ),
            Message(
                sender="friend",
                content="Thanks for the help yesterday!",
                ground_truth_priority=Priority.LOW,
                keywords=["thanks"]
            ),
            Message(
                sender="team",
                content="Weekly standup meeting at 3pm in the conference room.",
                ground_truth_priority=Priority.MEDIUM,
                keywords=["meeting", "standup"]
            ),
            Message(
                sender="client",
                content="Please review the attached contract and get back to me by end of day.",
                ground_truth_priority=Priority.HIGH,
                keywords=["contract", "review", "end of day"]
            ),
            Message(
                sender="marketing",
                content="Newsletter #42: Our latest product updates!",
                ground_truth_priority=Priority.LOW,
                keywords=["newsletter"]
            ),
        ]
        def grader(actions, final_state):
            # Count correctly set priorities
            correct = 0
            total_priority_actions = 0
            for act in actions:
                if act[0] == "set_priority":
                    total_priority_actions += 1
                    msg_id = act[2]
                    chosen_priority = act[1]
                    msg = next(m for m in messages if m.id == msg_id)
                    if chosen_priority == msg.ground_truth_priority:
                        correct += 1
            if total_priority_actions == 0:
                return 0.0
            return correct / total_priority_actions
        return messages, grader, "Classify 5 obvious messages", 10

    elif task_name == "medium":
        # More messages with subtle clues
        messages = [
            Message(
                sender="manager",
                content="Following up on the quarterly report – please send me the draft.",
                ground_truth_priority=Priority.MEDIUM,
                keywords=["follow up", "report", "draft"]
            ),
            Message(
                sender="ceo",
                content="Call me ASAP – critical issue.",
                ground_truth_priority=Priority.HIGH,
                keywords=["asap", "critical", "call"]
            ),
            Message(
                sender="support",
                content="Your ticket #12345 has been updated.",
                ground_truth_priority=Priority.LOW,
                keywords=["ticket", "updated"]
            ),
            Message(
                sender="hr",
                content="Reminder: Performance reviews due Friday.",
                ground_truth_priority=Priority.MEDIUM,
                keywords=["reminder", "performance reviews", "due"]
            ),
            # Add more to make it challenging
        ]
        def grader(actions, final_state):
            # Partial credit: each correct priority gives 1 point, each one level off gives 0.5
            correct = 0.0
            total = 0
            priority_map = {"high": 3, "medium": 2, "low": 1}
            for act in actions:
                if act[0] == "set_priority":
                    total += 1
                    msg_id = act[2]
                    chosen = act[1]
                    msg = next(m for m in messages if m.id == msg_id)
                    expected = msg.ground_truth_priority
                    diff = abs(priority_map[chosen.value] - priority_map[expected.value])
                    if diff == 0:
                        correct += 1.0
                    elif diff == 1:
                        correct += 0.5
                    else:
                        correct += 0.0
            return correct / total if total > 0 else 0.0
        return messages, grader, "Infer priority from subtle language and context", 20

    else:  # hard
        # Dynamic stream – agent must pin high‑priority messages before moving on
        messages = [
            Message(sender="alerts", content="Server down! Immediate action required.", ground_truth_priority=Priority.HIGH, keywords=["down", "immediate"]),
            Message(sender="friend", content="What's up?", ground_truth_priority=Priority.LOW),
            Message(sender="boss", content="Get me the sales report by noon.", ground_truth_priority=Priority.HIGH, keywords=["report", "by noon"]),
            Message(sender="newsletter", content="Weekly digest", ground_truth_priority=Priority.LOW),
            Message(sender="security", content="Suspicious login detected from new device.", ground_truth_priority=Priority.HIGH, keywords=["suspicious", "login"]),
            # ... more messages
        ]
        def grader(actions, final_state):
            # Quality of pinned set (precision and recall)
            pinned_ids = final_state.get("pinned_ids", [])
            all_high_ids = [m.id for m in messages if m.ground_truth_priority == Priority.HIGH]
            if not all_high_ids:
                return 1.0
            true_positives = len([pid for pid in pinned_ids if pid in all_high_ids])
            precision = true_positives / len(pinned_ids) if pinned_ids else 0
            recall = true_positives / len(all_high_ids)
            if precision + recall == 0:
                return 0.0
            f1 = 2 * (precision * recall) / (precision + recall)
            return f1
        return messages, grader, "Dynamic triage: pin high‑priority messages in a stream", 30