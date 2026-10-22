#!/usr/bin/env python3
"""Recover the dated 2024 Aliengo physics; adapt only foot-body naming to gym 1.1.8."""
import hashlib,json,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
repo=ROOT/"work/terrain-memory/upstreams/gym-quadruped-history"
commit="efb7217b9515613b238405846dcdbc92cf07f488"
dest=ROOT/"work/terrain-memory/aliengo-2024"
dest.mkdir(parents=True,exist_ok=True)
archive=subprocess.check_output(["git","archive",commit,"gym_quadruped/robot_model/aliengo"],cwd=repo)
subprocess.run(["tar","-xf","-","-C",str(dest)],input=archive,check=True)
source=dest/"gym_quadruped/robot_model/aliengo/aliengo.xml"
tree=ET.parse(source);root=tree.getroot()
root.find("compiler").set("meshdir",str(source.parent/"assets"))
for leg in ["FL","FR","RL","RR"]:
    calf=root.find(f".//body[@name='{leg}_calf']")
    visual=calf.find(f"geom[@name='{leg}']")
    collision=next(g for g in calf.findall("geom") if g.get("size")=="0.0265")
    # Same world position, collision radius, friction and inertia. The modern API
    # uses body origins for feet_pos/Jacobians, so give each original foot a massless body.
    foot=ET.SubElement(calf,"body",name=leg+"_foot",pos=visual.get("pos"))
    visual.set("name",leg+"_visual")
    collision.set("name",leg)
    for geom in [visual,collision]:
        calf.remove(geom);geom.set("pos","0 0 0");geom.set("mass","0");foot.append(geom)
target=dest/"aliengo-paper-2024.xml"
tree.write(target,encoding="unicode")
record=dict(repository="https://github.com/iit-DLSLab/gym-quadruped",commit=commit,
 source_path="gym_quadruped/robot_model/aliengo/aliengo.xml",
 source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 adapted_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
 adapted_path=str(target.relative_to(ROOT)),
 adaptation="Only absolute asset path and massless foot bodies/names for modern gym feet-position API. Joint damping, friction, armature, torque limits, geometry, mass, home pose and contact options are from the dated author XML.",
 paper_release_status="Dated author dependency snapshot, not certified by the authors as the exact Table I release.")
(ROOT/"experiments/tm01/aliengo-2024-source.json").write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record,indent=2))
