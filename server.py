# server.py
from fastapi import FastAPI, HTTPException
from environment import SmartInboxEnv, Action
from tasks import load_task
import inference

app = FastAPI(title="Smart Message Prioritization Environment", version="1.0.0")

# Store active environment and grader per task
active = {}

@app.get("/reset")
def reset(task: str = "easy"):
    if task not in ["easy", "medium", "hard"]:
        raise HTTPException(status_code=400, detail="Invalid task. Choose easy, medium, or hard.")
    messages, grader, desc, steps = load_task(task)
    env = SmartInboxEnv(messages, steps)
    active[task] = (env, grader)
    obs = env.reset()
    return obs.dict()

@app.post("/step")
def step(task: str, action: Action):
    if task not in active:
        raise HTTPException(status_code=400, detail="Environment not reset. Call /reset first.")
    env, _ = active[task]
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def state(task: str):
    if task not in active:
        raise HTTPException(status_code=400, detail="Environment not reset.")
    env, _ = active[task]
    return env.state()

@app.get("/grader")
def grader(task: str):
    if task not in active:
        raise HTTPException(status_code=400, detail="Environment not reset.")
    env, grader_func = active[task]
    if not env.done:
        return {"score": 0.0, "message": "Episode not finished. Call step until done."}
    score = grader_func(env.actions_taken, env.state())
    return {"score": score}

@app.get("/tasks")
def tasks():
    return {
        "tasks": ["easy", "medium", "hard"],
        "descriptions": {
            "easy": "Classify 5 messages with obvious priority",
            "medium": "Infer priority from subtle language",
            "hard": "Dynamic triage: pin high‑priority messages in a stream"
        },
        "action_schema": Action.schema()
    }

@app.get("/baseline")
def baseline():
    try:
        scores = inference.main()
        return {"scores": scores, "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)