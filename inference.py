# inference.py
import os
import json
import openai
from environment import SmartInboxEnv, Action, Priority
from tasks import load_task

def run_task(task_name: str) -> float:
    # Check env vars here
    API_BASE_URL = os.getenv("API_BASE_URL")
    MODEL_NAME = os.getenv("MODEL_NAME")
    HF_TOKEN = os.getenv("HF_TOKEN")

    if not all([API_BASE_URL, MODEL_NAME, HF_TOKEN]):
        raise RuntimeError("Missing API_BASE_URL, MODEL_NAME, or HF_TOKEN")

    openai.api_base = API_BASE_URL
    openai.api_key = HF_TOKEN

    messages, grader, description, max_steps = load_task(task_name)
    env = SmartInboxEnv(messages, max_steps)
    obs = env.reset()
    done = False
    step = 0

    while not done and step < max_steps:
        # Build the prompt for the LLM
        prompt = f"""
You are an AI assistant managing a message inbox to prevent important messages from being buried.

Current message:
- Sender: {obs.current_message.sender}
- Content: {obs.current_message.content}
- Position: {obs.inbox_position} of {obs.total_messages}
- Already pinned high‑priority message IDs: {obs.pinned_high_priority_ids}

Your job is to decide the priority (high, medium, low) for this message.
If it is high priority, you should also consider pinning it (using the pin_message action after setting priority).

Return a JSON object with exactly one action.
Possible actions:
1. Set priority: {{"action_type": "set_priority", "priority": "high"}} (or "medium"/"low")
2. Pin a message: {{"action_type": "pin_message", "message_id": "the_message_id"}}
3. Move to next message: {{"action_type": "next"}}

For this message, you should set its priority first, then optionally pin if it's high priority.
You can also directly go to next without setting priority (penalized).
Be efficient.
"""
        try:
            completion = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=150,
                timeout=30
            )
            content = completion.choices[0].message.content.strip()
            # Remove markdown code fences if present
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0]
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0]
            action_dict = json.loads(content)
            action = Action(**action_dict)
        except Exception as e:
            print(f"Error: {e}, using fallback action (set_priority low)")
            action = Action(action_type="set_priority", priority=Priority.LOW)

        obs, reward, done, info = env.step(action)
        step += 1
        print(f"Step {step}: {action.action_type} -> reward {reward:.2f}")

    score = grader(env.actions_taken, env.state())
    return score

def main():
    scores = {}
    for task in ["easy", "medium", "hard"]:
        print(f"\n--- Running {task} task ---")
        try:
            s = run_task(task)
            scores[task] = s
            print(f"Score: {s:.2f}")
        except Exception as e:
            print(f"Error: {e}")
            scores[task] = 0.0
    return scores

if __name__ == "__main__":
    scores = main()
    print("\n=== BASELINE SCORES ===")
    for task, s in scores.items():
        print(f"{task}: {s:.2f}")
    print("======================")