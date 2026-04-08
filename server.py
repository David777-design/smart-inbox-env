from fastapi import FastAPI, HTTPException
from environment import SmartInboxEnv, Action
from tasks import load_task

app = FastAPI()
active = {}

@app.get("/reset")
def reset(task: str = "easy"):
    if task not in ["easy", "medium", "hard"]:
        raise HTTPException(400, "Invalid task")
    messages, grader, _, steps = load_task(task)
    env = SmartInboxEnv(messages, steps)
    active[task] = (env, grader)
    return env.reset().dict()

@app.post("/step")
def step(task: str, action: Action):
    if task not in active:
        raise HTTPException(400, "Reset first")
    env, _ = active[task]
    obs, reward, done, info = env.step(action)
    return {"observation": obs.dict(), "reward": reward, "done": done, "info": info}

@app.get("/state")
def state(task: str):
    if task not in active:
        raise HTTPException(400, "Reset first")
    env, _ = active[task]
    return env.state()

@app.get("/grader")
def grader(task: str):
    if task not in active:
        raise HTTPException(400, "Reset first")
    env, grader_func = active[task]
    if not env.done:
        return {"score": 0.0}
    score = grader_func(env.actions_taken, env.state())
    return {"score": score}

@app.get("/tasks")
def tasks():
    return {
        "tasks": ["easy", "medium", "hard"],
        "action_schema": Action.schema()
    }

@app.get("/baseline")
def baseline():
    import inference   # lazy import to avoid missing env vars at startup
    scores = inference.main()
    return {"scores": scores}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)