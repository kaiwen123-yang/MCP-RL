#!/usr/bin/env python3
"""Build the W34 trunk-clearance correction from the retained S5-r21 source."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BUILD = "S5-r23"

def main():
    prior = ROOT / "work/r11-frozen/S5-r21"
    frozen = ROOT / "work/r11-frozen" / BUILD
    source = ROOT / "work/terrain-memory/upstreams/mujoco_mpc-go2"
    specification = json.loads((ROOT / f"experiments/terrain_memory/s1-go2-r11-{BUILD}.json").read_text())
    if frozen.exists():
        raise ValueError("Preserve the already built controller; do not overwrite it.")
    # The pinned upstream supplies CMake/dependencies; all controller sources
    # come from the actual retained r21, including its earlier repairs.
    shutil.copytree(prior / "mjpc", source / "mjpc", dirs_exist_ok=True)
    shutil.copy2(prior / "SOURCE_COMMIT", source / "SOURCE_COMMIT")
    cc = Path("mjpc/tasks/quadruped/quadruped.cc")
    header = Path("mjpc/tasks/quadruped/quadruped.h")
    old = (prior / cc).read_text()
    residual = """  // W34: the real task terminates at the first forbidden trunk-floor contact.
  // Penalize that same geometry/margin in each predicted state, before contact
  // can become an attractive support and recovery strategy.
  const double* clearance = GetCustomNumericData(model, "r12_trunk_clearance");
  if (clearance) {
    for (int geom : trunk_collision_geoms_) {
      const double margin = mju_max(model->geom_margin[floor_geom_id_],
                                    model->geom_margin[geom]);
      const double distance = mj_geomDistance(model, data, floor_geom_id_,
                                              geom, 1.0, nullptr);
      const double over = margin - distance;
      residual[counter++] = 0.5*(over + mju_sqrt(over*over + 1e-12))/clearance[0];
    }
  }

"""
    new = old.replace("  // sensor dim sanity check\n", residual + "  // sensor dim sanity check\n", 1)
    marker = '  if (residual_.torso_body_id_ < 0) mju_error("body \'trunk\' not found");'
    reset = """
  residual_.trunk_collision_geoms_.clear();
  if (GetCustomNumericData(model, "r12_trunk_clearance")) {
    residual_.floor_geom_id_ = mj_name2id(model, mjOBJ_GEOM, "floor");
    const int floor = residual_.floor_geom_id_;
    for (int geom = 0; geom < model->ngeom; ++geom) {
      if (model->geom_bodyid[geom] == residual_.torso_body_id_ &&
          ((model->geom_contype[geom] & model->geom_conaffinity[floor]) ||
           (model->geom_contype[floor] & model->geom_conaffinity[geom]))) {
        residual_.trunk_collision_geoms_.push_back(geom);
      }
    }
  }
"""
    if old.count(marker) != 1 or old.count("  // sensor dim sanity check\n") != 1:
        raise ValueError("The retained r21 source does not match the declared repair.")
    new = new.replace(marker, marker + "\n" + reset, 1)
    h_old = (prior / header).read_text()
    h_new = h_old.replace("    bool soft_landing_ = false;",
        "    int floor_geom_id_ = -1;\n"
        "    std::vector<int> trunk_collision_geoms_;\n"
        "    bool soft_landing_ = false;", 1)
    (source / cc).write_text(new)
    (source / header).write_text(h_new)
    patch = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                     fromfile="S5-r21/" + str(cc), tofile=BUILD + "/" + str(cc)))
    patch += "".join(difflib.unified_diff(h_old.splitlines(True), h_new.splitlines(True),
                      fromfile="S5-r21/" + str(header), tofile=BUILD + "/" + str(header)))
    (ROOT / f"experiments/terrain_memory/s1-go2-r11-{BUILD}.patch").write_text(patch)
    # Reuse actual frozen task assets instead of fetching unrelated model packs.
    root_cmake = subprocess.check_output(["git", "show", "HEAD:CMakeLists.txt"], cwd=source, text=True)
    first = root_cmake.index("FetchContent_Declare(\n  menagerie")
    last = root_cmake.index("if(NOT TARGET lodepng)", first)
    (source / "CMakeLists.txt").write_text(root_cmake[:first] +
        "# D57 reuses the frozen r21 task resources.\n" + root_cmake[last:])
    (source / "mjpc/tasks/CMakeLists.txt").write_text(
        'add_custom_target(copy_model_resources\n'
        '  COMMAND ${CMAKE_COMMAND} -E copy_directory\n'
        f'  "{prior / "runtime-tasks"}" "${{CMAKE_CURRENT_BINARY_DIR}}"\n'
        ')\nadd_dependencies(libmjpc copy_model_resources)\n')
    builddir = source / "build-autodl"
    begin = time.time()
    command = ["cmake", "-S", str(source), "-B", str(builddir), "-DCMAKE_BUILD_TYPE=Release",
        "-DMJPC_BUILD_TESTS=OFF", "-DMJPC_GRPC_BUILD_TESTS=OFF", "-DPYMJPC_BUILD_TESTS=OFF",
        "-DBUILD_TESTING=OFF", "-DMJPC_BUILD_GRPC_SERVICE=ON", "-DMUJOCO_MPC_MUJOCO_GIT_TAG=088079eff0450e32b98ee743141780ed68307506"]
    log_path = source / f"build-{BUILD}.log"
    with log_path.open("w") as log:
        log.write("The previous zero-step S5-r22 startup used release-tag headers against the retained pre-release library.\n")
        log.write("This build restores the original MuJoCo dependency commit 088079eff0450e32b98ee743141780ed68307506.\n")
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
    identity.update(build=BUILD, based_on="S5-r21", prior_binary_sha256=identity["binary_sha256"],
        binary_sha256=hashlib.file_digest((frozen / "agent_server").open("rb"), "sha256").hexdigest(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        build_started_unix_s=begin, build_wall_s=time.time()-begin,
        build_command=command, original_runtime_library_reused=True)
    for relative in (cc, header):
        identity["source_files_sha256"][str(relative)] = hashlib.file_digest(
            (frozen / relative).open("rb"), "sha256").hexdigest()
    identity["configuration_overrides"]["r12_trunk_clearance"] = specification["r12_trunk_clearance"]
    (frozen / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps(dict(build=BUILD, wall_s=identity["build_wall_s"], binary=str(frozen / "agent_server"))))

if __name__ == "__main__":
    main()
