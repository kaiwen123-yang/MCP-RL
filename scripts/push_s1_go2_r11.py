#!/usr/bin/env python3
"""Push R11 source/evidence with ephemeral authenticated gh/Git credentials."""
import base64,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPO="kaiwen123-yang/MCP-RL"
BRANCH="codex/r11-material-continuation"
def main():
    credential=json.loads(sys.stdin.read())
    env=dict(os.environ,GH_TOKEN=credential["token"],GIT_TERMINAL_PROMPT="0",
        GIT_CONFIG_COUNT="2",GIT_CONFIG_KEY_0="http.https://github.com/.extraheader",
        GIT_CONFIG_VALUE_0="AUTHORIZATION: basic "+base64.b64encode(("x-access-token:"+credential["token"]).encode()).decode(),
        GIT_CONFIG_KEY_1="http.proxy",GIT_CONFIG_VALUE_1=os.environ.get("https_proxy",""))
    def command(args):
        p=subprocess.run(args,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if p.returncode:
            error=p.stderr.lower()
            category="authentication" if "authentication" in error else "tls" if "tls" in error or "ssl" in error else "network" if any(x in error for x in ["connect","resolve","timeout","timed out"]) else "command"
            print(json.dumps(dict(operation=args[0],exit_code=p.returncode,error_class=category)),flush=True)
            raise SystemExit(p.returncode)
        return p.stdout.strip()
    command(["git","push","origin",f"HEAD:refs/heads/{BRANCH}"])
    head=command(["git","rev-parse","HEAD"])
    actual=json.loads(command(["gh","api",f"repos/{REPO}/git/ref/heads/{BRANCH}"]))["object"]["sha"]
    if actual!=head:raise RuntimeError("Remote source branch differs from submitted commit")
    print(json.dumps(dict(branch=BRANCH,commit=head,url=f"https://github.com/{REPO}/tree/{BRANCH}")),flush=True)
if __name__=="__main__":main()
