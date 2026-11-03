#!/usr/bin/env python3
"""Build the specified R12 revision from its retained executed source base."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BUILD = "S5-r26"

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--build',default='S5-r26')
    args=parser.parse_args()
    global BUILD
    BUILD=args.build
    specification=json.loads((ROOT/f'experiments/terrain_memory/s1-go2-r11-{BUILD}.json').read_text())
    prior = ROOT / "work/r11-frozen" / specification['based_on']
    frozen = ROOT / "work/r11-frozen" / BUILD
    source = ROOT / "work/terrain-memory/upstreams/mujoco_mpc-go2"
    specification = json.loads((ROOT / f"experiments/terrain_memory/s1-go2-r11-{BUILD}.json").read_text())
    if frozen.exists():
        raise ValueError("Preserve the already built controller; do not overwrite it.")
    # The pinned upstream supplies CMake/dependencies; all controller sources
    # come from the actual retained r24, including its earlier repairs.
    shutil.copytree(prior / "mjpc", source / "mjpc", dirs_exist_ok=True)
    shutil.copy2(prior / "SOURCE_COMMIT", source / "SOURCE_COMMIT")
    patch_path = ROOT / f"experiments/terrain_memory/s1-go2-r11-{BUILD}.patch"
    for line in patch_path.read_text().splitlines():
        if line.startswith("+++ b/"):
            target=source/line[6:]
            if target.exists():target.write_text(target.read_text())
    subprocess.run(["patch", "-p1", "--forward", "--batch", "-i", str(patch_path)],
                   cwd=source, check=True)
    changed_sources = [line[6:] for line in patch_path.read_text().splitlines() if line.startswith('+++ b/')]
    # Reuse actual frozen task assets instead of fetching unrelated model packs.
    root_cmake = subprocess.check_output(["git", "show", "HEAD:CMakeLists.txt"], cwd=source, text=True)
    first = root_cmake.index("FetchContent_Declare(\n  menagerie")
    last = root_cmake.index("if(NOT TARGET lodepng)", first)
    (source / "CMakeLists.txt").write_text(root_cmake[:first] +
        "# D57 reuses the frozen r24 task resources.\n" + root_cmake[last:])
    (source / "mjpc/tasks/CMakeLists.txt").write_text(
        'add_custom_target(copy_model_resources\n'
        '  COMMAND ${CMAKE_COMMAND} -E copy_directory\n'
        f'  "{prior / "runtime-tasks"}" "${{CMAKE_CURRENT_BINARY_DIR}}"\n'
        ')\nadd_dependencies(libmjpc copy_model_resources)\n')
    builddir = source / "build-autodl"
    # FindOrFetch uses UPDATE_DISCONNECTED, so changing its cached GIT_TAG
    # does not move an already populated dependency checkout.
    mujoco_source = builddir / "_deps/mujoco-src"
    if mujoco_source.is_dir():
        subprocess.run(["git", "checkout", "--detach",
            "088079eff0450e32b98ee743141780ed68307506"], cwd=mujoco_source, check=True)
    begin = time.time()
    command = ["cmake", "-S", str(source), "-B", str(builddir), "-DCMAKE_BUILD_TYPE=Release",
        "-DMJPC_BUILD_TESTS=OFF", "-DMJPC_GRPC_BUILD_TESTS=OFF", "-DPYMJPC_BUILD_TESTS=OFF",
        "-DBUILD_TESTING=OFF", "-DMJPC_BUILD_GRPC_SERVICE=ON", "-DMUJOCO_MPC_MUJOCO_GIT_TAG=088079eff0450e32b98ee743141780ed68307506"]
    log_path = source / f"build-{BUILD}.log"
    with log_path.open("w") as log:
        log.write(f"{BUILD} from executed {specification['based_on']}; reuse pinned MuJoCo dependency and runtime.\n")
        log.flush()
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(["cmake", "--build", str(builddir), "--target", "agent_server", "-j", "8"],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    frozen.mkdir()
    shutil.copytree(source / "mjpc", frozen / "mjpc")
    # Keep the original running MuJoCo library and runtime models unchanged.
    shutil.copytree(prior / "runtime-tasks", frozen / "runtime-tasks")
    shutil.copytree(prior / "lib", frozen / "lib")
    shutil.copy2(prior / "SOURCE_COMMIT", frozen / "SOURCE_COMMIT")
    shutil.copy2(builddir / "bin/agent_server", frozen / "agent_server")
    shutil.copy2(log_path, frozen / "build.log")
    identity = json.loads((prior / "identity.json").read_text())
    identity.update(build=BUILD, based_on=specification["based_on"], prior_binary_sha256=identity["binary_sha256"],
        binary_sha256=hashlib.file_digest((frozen / "agent_server").open("rb"), "sha256").hexdigest(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        build_started_unix_s=begin, build_wall_s=time.time()-begin,
        build_command=command, original_runtime_library_reused=True,
        mujoco_dependency_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=mujoco_source, text=True).strip())
    for relative in changed_sources:
        identity["source_files_sha256"][str(relative)] = hashlib.file_digest(
            (frozen / relative).open("rb"), "sha256").hexdigest()
    identity["configuration_overrides"].update(specification["configuration_overrides"])
    (frozen / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps(dict(build=BUILD, wall_s=identity["build_wall_s"], binary=str(frozen / "agent_server"))))

if __name__ == "__main__":
    main()
