import os
import json
from openai import OpenAI
from environment import SmartInboxEnv, Action, Priority
from tasks import load_task

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

if not all([API_BASE_URL, MODEL_NAME, HF_TOKEN]):
    raise RuntimeError("Missing environment variables")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def run_task(task_name: str) -> float:
    messages, grader, _, max_steps = load_task(task_name)
    env = SmartInboxEnv(messages, max_steps)
    obs = env.reset()
    done = False
    step = 0
    while not done and step < max_steps:
        prompt = f"Message: {obs.current_message.content}\nSender: {obs.current_message.sender}\nChoose priority (high/medium/low). Return JSON: {{\"action_type\":\"set_priority\",\"priority\":\"high\"}}"
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=50
            )
            content = completion.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0]
            action_dict = json.loads(content)
            action = Action(**action_dict)
        except Exception:
            action = Action(action_type="set_priority", priority=Priority.LOW)
        obs, reward, done, _ = env.step(action)
        step += 1
    score = grader(env.actions_taken, env.state())
    return score

def main():
    scores = {}
    for task in ["easy", "medium", "hard"]:
        try:
            scores[task] = run_task(task)
        except Exception:
            scores[task] = 0.0
    return scores

if __name__ == "__main__":
    print(main())