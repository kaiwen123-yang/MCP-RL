"""Map the frozen project Go2 MJCF to the author's Quadruped task inputs.

This only assembles model source; it does not initialize or advance a robot.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _numbers(values):
    return " ".join(format(float(value), ".17g") for value in values)


def build_project_model(author_source, original_robot_dir, original_scene,
                        original_resolved, output_dir):
    """Return (XML path, provenance) for G1 without changing old source files.

    original_robot_dir is gym_quadruped/robot_model/go2; original_scene is
    gym_quadruped/utils/mujoco/assets/scene_flat.xml.  original_resolved is the
    frozen fourth-round development-v1-seed4/rigid-T1R-resolved.json.
    """
    author_source = Path(author_source).resolve()
    original_robot_dir = Path(original_robot_dir).resolve()
    original_scene = Path(original_scene).resolve()
    original_resolved = Path(original_resolved).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    robot_path = original_robot_dir / "go2.xml"
    task_path = author_source / "mjpc/tasks/quadruped/task_flat.xml"
    robot = ET.parse(robot_path).getroot()
    scene = ET.parse(original_scene).getroot()
    task = ET.parse(task_path).getroot()
    resolved = json.loads(original_resolved.read_text())

    # Keep the original robot's defaults, inertia, passive dynamics, geometry,
    # joint hierarchy and order.  Absolute mesh paths refer to retained assets.
    model = ET.Element("mujoco", model="G1 original Go2 with author task mapping")
    compiler = deepcopy(robot.find("compiler"))
    model.append(compiler)
    option = deepcopy(robot.find("option"))
    option.set("timestep", ".0002")
    model.append(option)
    model.append(deepcopy(robot.find("default")))
    model.append(deepcopy(task.find("custom")))
    for tag in ("statistic", "visual"):
        model.append(deepcopy(scene.find(tag)))
    assets = ET.SubElement(model, "asset")
    asset_sources = []
    for element in robot.find("asset"):
        copied = deepcopy(element)
        if "file" in copied.attrib:
            source = original_robot_dir / copied.get("file")
            copied.set("file", str(source))
            asset_sources.append(dict(path=str(source), sha256=_digest(source)))
        assets.append(copied)
    for element in scene.find("asset"):
        assets.append(deepcopy(element))

    world = ET.SubElement(model, "worldbody")
    for element in scene.find("worldbody"):
        world.append(deepcopy(element))
    for element in robot.find("worldbody"):
        world.append(deepcopy(element))
    base = world.find("body[@name='base']")
    base.set("name", "trunk")
    leg_roots = ("FL_hip", "FR_hip", "RL_hip", "RR_hip")
    allowed_ground_bodies = [body.get("name") for root in base.findall("body")
                            if root.get("name") in leg_roots for body in root.iter("body")]
    ET.SubElement(base, "site", name="head", pos="0.3 0 0", size=".02",
                  rgba="1 0 0 1", group="5")
    goal = ET.SubElement(world, "body", name="goal", mocap="true", pos=".3 0 .28")
    ET.SubElement(goal, "site", name="task_goal", size=".06",
                  rgba="0 1 0 .5", group="5")

    # These are the actual runtime overrides of the frozen rigid baseline.
    # Other collision shapes and all their original settings stay untouched.
    for name in ("floor", "FL", "FR", "RL", "RR"):
        geom = world.find(".//geom[@name='%s']" % name)
        geom.set("solref", _numbers(resolved["native_contact_solref"]))
        geom.set("solimp", _numbers(resolved["native_contact_solimp"]))
        geom.set("margin", "0")
        geom.set("friction", ".5 .005 0")

    ranges = dict(zip(resolved["joint_names"], resolved["joint_ranges_rad"]))
    actuators = ET.SubElement(model, "actuator")
    actuator_map = []
    for original, limits in zip(robot.find("actuator"), resolved["actuator_limits_Nm"]):
        joint = original.get("joint")
        attributes = dict(original.attrib)
        attributes.update(gainprm="60", biastype="affine", biasprm="0 -60 -5",
                          ctrllimited="true", ctrlrange=_numbers(ranges[joint]),
                          forcelimited="true", forcerange=_numbers(limits))
        # The original motors have default unit gear; retain any explicit gear.
        attributes["gear"] = original.get("gear", "1")
        ET.SubElement(actuators, "general", **attributes)
        actuator_map.append(dict(name=original.get("name"), joint=joint,
                                 gear=attributes["gear"], target_range_rad=ranges[joint],
                                 force_range_Nm=limits, kp=60, kd=5))

    # Author residual sensors must lead the sensor list.  The original measured
    # sensors are retained afterward and do not affect physical dynamics.
    sensors = ET.SubElement(model, "sensor")
    for collection in (task.find("sensor"), robot.find("sensor")):
        for element in collection:
            sensors.append(deepcopy(element))
    keyframes = ET.SubElement(model, "keyframe")
    for key in robot.find("keyframe"):
        keyframes.append(deepcopy(key))
    home = keyframes.find("key[@name='home']")
    home.set("qpos", _numbers(resolved["initial_qpos"]))
    home.set("qvel", _numbers([0.] * 18))
    home.set("ctrl", _numbers(resolved["initial_qpos"][7:]))

    ET.indent(model, space="  ")
    output = output_dir / "G1-project-go2.xml"
    ET.ElementTree(model).write(output, encoding="unicode", xml_declaration=False)
    provenance = dict(
        xml=str(output), xml_sha256=_digest(output),
        original_model_mass_kg=resolved["actual_mass_kg"],
        sources=[dict(path=str(path), sha256=_digest(path)) for path in
                 (robot_path, original_scene, original_resolved, task_path)],
        mesh_assets=asset_sources,
        body_name_mapping={"base": "trunk"},
        base_body_name="trunk", foot_geom_names=["FL", "FR", "RL", "RR"],
        allowed_ground_body_names=allowed_ground_bodies,
        added_nonphysical_sites={"head": [0.3, 0., 0.], "task_goal": "mocap goal"},
        actuator_mapping=actuator_map,
        initial_qpos=resolved["initial_qpos"], initial_qvel=[0.] * 18,
        initial_velocity_rule="The runner applies the declared physical initial-condition velocities after home initialization.",
        physical_timestep_s=.0002,
        native_rigid_contact=dict(solref=resolved["native_contact_solref"],
                                  solimp=resolved["native_contact_solimp"], margin=0.,
                                  ground_and_feet_friction=[.5, .005, 0.]),
        integration_scope="The runner writes the selected prediction numerics before Agent initialization; plant timestep remains 0.0002 s.",
        actuator_scope="The same compiled target/PD/force mapping is used by plant and Agent.",
    )
    return output, provenance


def write_flat_ground_height(model, xml_path):
    """Opt in only the actual world-fixed horizontal floor to analytic height."""
    import mujoco
    import numpy as np
    floor = model.geom('floor').id
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    mujoco.mj_kinematics(model, data)
    normal = data.geom_xmat[floor].reshape(3, 3)[:, 2]
    if (model.geom_type[floor] != mujoco.mjtGeom.mjGEOM_PLANE or
            model.geom_bodyid[floor] != 0 or
            not np.allclose(normal, [0., 0., 1.], atol=1e-12, rtol=0)):
        raise ValueError('s1_flat_ground_height requires a world-fixed upward horizontal plane')
    height = float(data.geom_xpos[floor, 2])
    tree = ET.parse(xml_path)
    ET.SubElement(tree.getroot().find('custom'), 'numeric',
                  name='s1_flat_ground_height', data=format(height, '.17g'))
    tree.write(xml_path, encoding='unicode')
    return dict(geom_name='floor', geom_id=int(floor), body_id=0,
                world_normal=normal.tolist(), world_height_m=height,
                numeric='s1_flat_ground_height', xml_sha256=_digest(xml_path))


def write_equilibrium_settings(xml_path, config, arm):
    """Wire declared controller settings into the XML actually loaded by Agent."""
    if config['stage'] in ('S1R8', 'S1R10'):
        prediction=config['prediction_settings']['AUTHOR_N' if arm=='AUTHOR_N' else 'C']
        values=dict(s1_stand_equilibrium=0,agent_timestep=prediction['integration_dt_s'],
            agent_integrator=3,agent_differentiable=prediction['differentiable'],
            agent_horizon=config['execution']['requested_prediction_horizon_s'],
            s1_control_hold_substeps=1 if arm=='AUTHOR_N' else prediction.get('substeps',2))
        tree=ET.parse(xml_path);custom=tree.getroot().find('custom')
        for name,value in values.items():
            element=custom.find("numeric[@name='%s']" % name)
            if element is None:element=ET.SubElement(custom,'numeric',name=name)
            element.set('data',format(float(value),'.17g'))
        if arm in ('H','S'):
            height=config['support_reference']['H_realized']['absolute_base_height_m']
            ET.SubElement(custom,'numeric',name='s1_stand_absolute_height',data=format(height,'.17g'))
            sensors=tree.getroot().find('sensor');index=next(i for i,s in enumerate(sensors) if s.tag!='user')
            sensors.insert(index,ET.Element('user',name='StandAbsoluteHeight',dim='1',user='6 1 0 3 0.04'))
            values['s1_stand_absolute_height']=height
        if arm=='S':
            ET.SubElement(custom,'numeric',name='s1_stand_support',data='1')
            values['s1_stand_support']=1
            scale=config['support_reference']['S_realized']['residuals']['Effort']['scale_rad']
            ET.SubElement(custom,'numeric',name='s1_stand_input_scale',data=format(scale,'.17g'))
            ET.SubElement(custom,'numeric',name='s1_stand_load_tracking',data='0')
            values.update(s1_stand_input_scale=scale,s1_stand_load_tracking=0)
            ET.SubElement(tree.getroot(),'size',nuserdata='62')
            index=next(i for i,s in enumerate(sensors) if s.tag!='user')
            for name,dim,weight in [('StandFootPosition',12,.1),('StandFootVelocity',12,.02),('StandSupportForce',4,0.)]:
                sensors.insert(index,ET.Element('user',name=name,dim=str(dim),user=f'0 {weight} 0 1'))
                index+=1
        if config.get('controller_formulation') in ('GN','GNC'):
            ET.SubElement(custom,'numeric',name='r10_normal_landing',data='1')
            values['r10_normal_landing']=1
            index=next(i for i,s in enumerate(sensors) if s.tag!='user')
            for name,weight in [('LandingVelocity',.02),('LandingIndentation',.1)]:
                sensors.insert(index,ET.Element('user',name=name,dim='4',user=f'0 {weight} 0 1'))
                index+=1
        if config.get('controller_formulation') == 'GNC':
            stiffness=config['local_contact_model']['instantaneous_stiffness_N_m']
            ET.SubElement(custom,'numeric',name='r10_local_contact',data=format(stiffness,'.17g'))
            values['r10_local_contact']=stiffness
        tree.write(xml_path,encoding='unicode')
        return dict(arm=arm,prediction_profile='AUTHOR_N' if arm=='AUTHOR_N' else 'C',
            numerics=values,cost_weights={'Orientation':.03 if arm=='AUTHOR_N' else 1.,'Angmom':0.},
            prediction_smoothing_width_m=None,refsafe='preserved',plant_options='unchanged',
            xml_sha256=_digest(xml_path))
    setting = next(x for x in config['development']['G0_named_arms'] + config['development'].get('G1_revisions', []) if x['id']==arm)
    prediction = config['prediction_settings'][setting['prediction']]
    values = dict(s1_stand_equilibrium=setting['stationary_objective'],
        agent_timestep=prediction['agent_timestep_s'],
        agent_integrator={'Euler':0,'implicitfast':3}[prediction['agent_integrator']],
        agent_differentiable=prediction['agent_differentiable'],
        agent_horizon=config['execution']['prediction_horizon_s'])
    tree=ET.parse(xml_path); custom=tree.getroot().find('custom')
    for name,value in values.items():
        element=custom.find("numeric[@name='%s']" % name)
        if element is None:
            element=ET.SubElement(custom,'numeric',name=name)
        element.set('data',format(float(value),'.17g'))
    tree.write(xml_path,encoding='unicode')
    return dict(arm=arm, prediction_profile=setting['prediction'], numerics=values, cost_weights=setting.get('cost_weights',{}), prediction_smoothing_width_m=setting.get('prediction_smoothing_width_m'),
        refsafe='preserved', plant_options='unchanged', xml_sha256=_digest(xml_path))


def build_prediction_model(plant_xml, prediction_xml, smoothing_width):
    """One declared prediction-only constraint approximation; plant stays intact."""
    tree=ET.parse(plant_xml)
    changes={}
    for name in ['floor','FL','FR','RL','RR']:
        geom=tree.getroot().find(".//geom[@name='%s']" % name)
        original=[float(x) for x in geom.get('solimp').split()]
        modified=list(original);modified[2]=smoothing_width
        geom.set('solimp',_numbers(modified))
        changes[name]=dict(plant_solimp=original,predictor_xml_solimp=modified)
    tree.write(prediction_xml,encoding='unicode')
    return dict(xml=str(prediction_xml),xml_sha256=_digest(prediction_xml),geometry_changes=changes,
        runtime_d0='Author MakeDifferentiable still sets joint, geom and explicit-pair d0 to zero during rollout and derivatives.',
        actual_plant='Unchanged native contact, physics timestep, refsafe, PD and original torque limits.')
