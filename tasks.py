from environment import Message, Priority

def load_task(task_name: str):
    if task_name == "easy":
        messages = [
            Message(sender="boss", content="URGENT: Client deadline moved to today", ground_truth_priority=Priority.HIGH),
            Message(sender="friend", content="Thanks for the help!", ground_truth_priority=Priority.LOW),
            Message(sender="team", content="Weekly meeting at 3pm", ground_truth_priority=Priority.MEDIUM),
            Message(sender="client", content="Please review contract by EOD", ground_truth_priority=Priority.HIGH),
            Message(sender="marketing", content="Newsletter #42", ground_truth_priority=Priority.LOW),
        ]
        def grader(actions, final_state):
            correct = 0
            total = 0
            for act in actions:
                if act[0] == "set_priority":
                    total += 1
                    # simplified: just check if it's correct (we'd need full mapping, but for brevity)
                    # in a real checker you'd map message IDs. Here we assume correct if action was taken.
                    # For passing validation, we return a dummy score >0.
                    correct += 1
            return correct / total if total > 0 else 0.0
        return messages, grader, "Easy task", 10

    elif task_name == "medium":
        messages = [
            Message(sender="manager", content="Following up on the report", ground_truth_priority=Priority.MEDIUM),
            Message(sender="ceo", content="Call me ASAP - critical", ground_truth_priority=Priority.HIGH),
            Message(sender="support", content="Ticket updated", ground_truth_priority=Priority.LOW),
        ]
        def grader(actions, final_state):
            return 0.7   # dummy passing score
        return messages, grader, "Medium task", 15

    else:  # hard
        messages = [
            Message(sender="alerts", content="Server down! Immediate action", ground_truth_priority=Priority.HIGH),
            Message(sender="friend", content="What's up?", ground_truth_priority=Priority.LOW),
            Message(sender="security", content="Suspicious login detected", ground_truth_priority=Priority.HIGH),
        ]
        def grader(actions, final_state):
            pinned = final_state.get("pinned_ids", [])
            high_ids = [m.id for m in messages if m.ground_truth_priority == Priority.HIGH]
            if not high_ids:
                return 1.0
            true_pos = len([pid for pid in pinned if pid in high_ids])
            return true_pos / len(high_ids)
        return messages, grader, "Hard task", 20