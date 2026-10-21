"""Remove gym-quadruped 1.1.8's shared XML-file race in parallel MuJoCo episodes."""
from pathlib import Path
import gym_quadruped
p=Path(gym_quadruped.__file__).parent/"quadruped_env.py"
s=p.read_text()
s=s.replace("            scene_env.write(scene_env_path)\n\n            self.mjModel: MjModel = mujoco.MjModel.from_xml_path(str(scene_env_path.absolute()))",
"""            # Each environment has its own XML tree; no shared aliengo-flat.xml write.
            self.mjModel: MjModel = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))""")
p.write_text(s)
