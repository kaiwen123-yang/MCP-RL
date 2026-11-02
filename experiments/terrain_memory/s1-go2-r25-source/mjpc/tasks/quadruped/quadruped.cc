// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/tasks/quadruped/quadruped.h"

#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include "mjpc/tasks/quadruped/r10_contact.h"

namespace mjpc {
std::string QuadrupedHill::XmlPath() const {
  return GetModelPath("quadruped/task_hill.xml");
}
std::string QuadrupedFlat::XmlPath() const {
  return GetModelPath("quadruped/task_flat.xml");
}
std::string QuadrupedHill::Name() const { return "Quadruped Hill"; }
std::string QuadrupedFlat::Name() const { return "Quadruped Flat"; }

void QuadrupedFlat::ResidualFn::Residual(const mjModel* model,
                                         const mjData* data,
                                         double* residual) const {
  // start counter
  int counter = 0;

  // get foot positions
  double* foot_pos[kNumFoot];
  for (A1Foot foot : kFootAll)
    foot_pos[foot] = data->geom_xpos + 3 * foot_geom_id_[foot];

  // average foot position
  double avg_foot_pos[3];
  AverageFootPos(avg_foot_pos, foot_pos);

  double* torso_xmat = data->xmat + 9*torso_body_id_;
  double* goal_pos = data->mocap_pos + 3*goal_mocap_id_;
  double* compos = SensorByName(model, data, "torso_subtreecom");
  const double* s1_stand_equilibrium =
      GetCustomNumericData(model, "s1_stand_equilibrium");
  const bool stand_equilibrium =
      s1_stand_equilibrium && s1_stand_equilibrium[0] == 1.0 &&
      current_mode_ == kModeQuadruped && GetGait() == kGaitStand;


  const double* support_opt = GetCustomNumericData(model, "s1_stand_support");
  const bool support = support_opt && current_mode_ == kModeQuadruped &&
                       GetGait() == kGaitStand && data->userdata[0] == 1;
  const double* ref = data->userdata;

  // S5-r3: first-order roll/pitch error on the original material footprint.
  const bool r11_moving_body_support = current_mode_ == kModeQuadruped &&
      GetGait() != kGaitStand &&
      GetNumberOrDefault(0, model, "r11_moving_body_support") == 1;
  bool r11_material_support = r11_moving_body_support;
  if (soft_landing_ && GetCustomNumericData(model, "r10_local_contact")) {
    constexpr double half = 0.015811388300841896;
    for (A1Foot foot : kFootAll) {
      const double* p = foot_pos[foot];
      r11_material_support |= p[0]>=.8+half && p[0]<=2.2-half && mju_abs(p[1])<=.6-half;
    }
  }
  // ---------- Upright ----------
  if (stand_equilibrium || r11_material_support) {
    // Body z-axis minus world z-axis: linear in small roll/pitch errors.
    residual[counter++] = torso_xmat[2];
    residual[counter++] = torso_xmat[5];
    residual[counter++] = torso_xmat[8] - 1;
  } else if (current_mode_ != kModeFlip) {
    if (current_mode_ == kModeBiped) {
      double biped_type = parameters_[biped_type_param_id_];
      int handstand = ReinterpretAsInt(biped_type) ? -1 : 1;
      residual[counter++] = torso_xmat[6] - handstand;
    } else {
      double reference_matrix[9];
      if (support) mju_quat2Mat(reference_matrix, ref + 3);
      residual[counter++] = torso_xmat[8] - (support ? reference_matrix[8] : 1);
    }
    residual[counter++] = 0;
    residual[counter++] = 0;
  } else {
    // special handling of flip orientation
    double flip_time = data->time - mode_start_time_;
    double quat[4];
    FlipQuat(quat, flip_time);
    double* torso_xquat = data->xquat + 4*torso_body_id_;
    mju_subQuat(residual + counter, torso_xquat, quat);
    counter += 3;
  }


  // ---------- Height ----------
  // quadrupedal or bipedal height of torso over feet
  double* torso_pos = data->xipos + 3*torso_body_id_;
  bool is_biped = current_mode_ == kModeBiped;
  // double height_goal = is_biped ? kHeightBiped : kHeightQuadruped;
  double height_goal;
  if (current_mode_ == kModeBiped) {
    double biped_type = parameters_[biped_type_param_id_];
    int handstand = ReinterpretAsInt(biped_type) ? -1 : 1;
    height_goal = (handstand == -1) ? kHeightBipedHand : kHeightBiped;
  } else {
    height_goal = kHeightQuadruped;
  }

  if (support) height_goal = ref[1];

  if (current_mode_ == kModeScramble) {
    // disable height term in Scramble
    residual[counter++] = 0;
  } else if (current_mode_ == kModeFlip) {
    // height target for Backflip
    double flip_time = data->time - mode_start_time_;
    residual[counter++] = torso_pos[2] - FlipHeight(flip_time);
  } else {
    residual[counter++] = (torso_pos[2] - avg_foot_pos[2]) - height_goal;
  }


  // ---------- Position ----------
  double* head = data->site_xpos + 3*head_site_id_;
  double target[3];
  if (current_mode_ == kModeWalk) {
    // follow prescribed Walk trajectory
    double mode_time = data->time - mode_start_time_;
    Walk(target, mode_time);
  } else {
    // go to the goal mocap body
    target[0] = goal_pos[0];
    target[1] = goal_pos[1];
    target[2] = goal_pos[2];
  }
  if (support) mju_copy3(target, ref + 7);
  residual[counter++] = head[0] - target[0];
  residual[counter++] = head[1] - target[1];
  residual[counter++] =
      current_mode_ == kModeScramble ? 2 * (head[2] - target[2]) : 0;

  // ---------- Gait ----------
  A1Gait gait = GetGait();
  double step[kNumFoot], step_velocity[kNumFoot];
  FootStep(step, GetPhase(data->time), gait, step_velocity);
  // Explicit opt-in for a known, static horizontal ground plane.
  const double* flat_ground_height =
      GetCustomNumericData(model, "s1_flat_ground_height");
  for (A1Foot foot : kFootAll) {
    if (is_biped) {
      // ignore "hands" in biped mode
      bool handstand = ReinterpretAsInt(parameters_[biped_type_param_id_]);
      bool front_hand = !handstand && (foot == kFootFL || foot == kFootFR);
      bool back_hand = handstand && (foot == kFootHL || foot == kFootHR);
      if (front_hand || back_hand) {
        residual[counter++] = 0;
        continue;
      }
    }
    double query[3] = {foot_pos[foot][0], foot_pos[foot][1], foot_pos[foot][2]};

    if (current_mode_ == kModeScramble) {
      double torso_to_goal[3];
      double* goal = data->mocap_pos + 3*goal_mocap_id_;
      mju_sub3(torso_to_goal, goal, torso_pos);
      mju_normalize3(torso_to_goal);
      mju_sub3(torso_to_goal, goal, foot_pos[foot]);
      torso_to_goal[2] = 0;
      mju_normalize3(torso_to_goal);
      mju_addToScl3(query, torso_to_goal, 0.15);
    }

    double ground_height = flat_ground_height ? flat_ground_height[0]
                                             : Ground(model, data, query);
    double height_target = ground_height + model->geom_size[3*foot_geom_id_[foot]] + step[foot];
    double height_difference = foot_pos[foot][2] - height_target;
    if (current_mode_ == kModeScramble) {
      // in Scramble, foot higher than target is not penalized
      height_difference = mju_min(0, height_difference);
    }
    const bool moving_stance_height = gait != kGaitStand &&
        GetNumberOrDefault(0, model, "r11_moving_stance_height") == 1;
    residual[counter++] = (step[foot] || moving_stance_height) ? height_difference : 0;
  }


  // ---------- Balance ----------
  double* comvel = SensorByName(model, data, "torso_subtreelinvel");
  double capture_point[3];
  double fall_time = mju_sqrt(2*height_goal / 9.81);
  mju_addScl3(capture_point, compos, comvel, fall_time);
  residual[counter++] = capture_point[0] - avg_foot_pos[0];
  residual[counter++] = capture_point[1] - avg_foot_pos[1];


  // ---------- Effort ----------
  if (support) {
    mju_sub(residual + counter, data->ctrl, ref + 34, model->nu);
    mju_scl(residual + counter, residual + counter,
            1.0 / GetNumberOrDefault(0.1, model, "s1_stand_input_scale"), model->nu);
  } else {
    mju_scl(residual + counter, data->actuator_force, 2e-2, model->nu);
  }
  counter += model->nu;


  // ---------- Posture ----------
  double* home = KeyQPosByName(model, data, "home");
  mju_sub(residual + counter, data->qpos + 7, support ? ref + 22 : home + 7, model->nu);
  if (current_mode_ == kModeFlip) {
    double flip_time = data->time - mode_start_time_;
    if (flip_time < crouch_time_) {
      double* crouch = KeyQPosByName(model, data, "crouch");
      mju_sub(residual + counter, data->qpos + 7, crouch + 7, model->nu);
    } else if (flip_time >= crouch_time_ &&
               flip_time < jump_time_ + flight_time_) {
      // free legs during flight phase
      mju_zero(residual + counter, model->nu);
    }
  }
  for (A1Foot foot : kFootAll) {
    for (int joint = 0; joint < 3; joint++) {
      residual[counter + 3*foot + joint] *= kJointPostureGain[joint];
    }
  }
  if (current_mode_ == kModeBiped) {
    // loosen the "hands" in Biped mode
    bool handstand = ReinterpretAsInt(parameters_[biped_type_param_id_]);
    double arm_posture = parameters_[arm_posture_param_id_];
    if (handstand) {
      residual[counter + 6] *= arm_posture;
      residual[counter + 7] *= arm_posture;
      residual[counter + 8] *= arm_posture;
      residual[counter + 9] *= arm_posture;
      residual[counter + 10] *= arm_posture;
      residual[counter + 11] *= arm_posture;
    } else {
      residual[counter + 0] *= arm_posture;
      residual[counter + 1] *= arm_posture;
      residual[counter + 2] *= arm_posture;
      residual[counter + 3] *= arm_posture;
      residual[counter + 4] *= arm_posture;
      residual[counter + 5] *= arm_posture;
    }
  }
  counter += model->nu;


  // ---------- Yaw ----------
  double torso_heading[2] = {torso_xmat[0], torso_xmat[3]};
  if (current_mode_ == kModeBiped) {
    int handstand =
        ReinterpretAsInt(parameters_[biped_type_param_id_]) ? 1 : -1;
    torso_heading[0] = handstand * torso_xmat[2];
    torso_heading[1] = handstand * torso_xmat[5];
  }
  mju_normalize(torso_heading, 2);
  double heading_goal = parameters_[ParameterIndex(model, "Heading")];
  residual[counter++] = torso_heading[0] - mju_cos(heading_goal);
  residual[counter++] = torso_heading[1] - mju_sin(heading_goal);


  // ---------- Angular momentum ----------
  mju_copy3(residual + counter, SensorByName(model, data, "torso_angmom"));
  counter +=3;


  // ---------- Stand linear and angular velocity ----------
  // Go2's leading free joint matches the runner's base velocity quantities.
  // The scales are the original stopping thresholds: 0.05 m/s and 0.1 rad/s.
  // These residuals are evaluated at every stage, including the final state.
  if (stand_equilibrium) {
    mju_scl(residual + counter, data->qvel, 1.0 / 0.05, 3);
    mju_scl(residual + counter + 3, data->qvel + 3, 1.0 / 0.1, 3);
  } else {
    mju_zero(residual + counter, 6);
  }
  counter += 6;


  // R8 H: opt-in absolute base height during Stand, on the actual plane.
  const double* stand_height = GetCustomNumericData(model, "s1_stand_absolute_height");
  if (stand_height) {
    const bool stand = current_mode_ == kModeQuadruped && GetGait() == kGaitStand;
    residual[counter++] = stand ? data->xpos[3 * torso_body_id_ + 2] - (support ? ref[2] : stand_height[0]) : 0;
  }

  if (support_opt) {
    // Python reference order FL, FR, RL, RR; enum order FL, HL, FR, HR.
    constexpr A1Foot order[4] = {kFootFL, kFootFR, kFootHL, kFootHR};
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 3; ++j)
        residual[counter++] = support ? (foot_pos[order[i]][j] - ref[10 + 3*i + j]) / 0.02 : 0;
    }
    for (A1Foot foot : order) {
      double velocity[6];
      if (support) mj_objectVelocity(model, data, mjOBJ_GEOM, foot_geom_id_[foot], velocity, 0);
      for (int j = 0; j < 3; ++j)
        residual[counter++] = support ? velocity[3+j] / 0.05 : 0;
    }
    // S2: preserve the solved load distribution as a soft MPC objective.
    // Micrometre foot lifting is too cheap under position tracking alone.
    const bool track_load = support && GetNumberOrDefault(0, model, "s1_stand_load_tracking") == 1;
    double vertical_load[4] = {0, 0, 0, 0};
    if (track_load) {
      for (int k = 0; k < data->ncon; ++k) {
        const mjContact& contact = data->contact[k];
        for (int i = 0; i < 4; ++i) {
          const int geom = foot_geom_id_[order[i]];
          const bool first = contact.geom1 == geom && model->geom_bodyid[contact.geom2] == 0;
          const bool second = contact.geom2 == geom && model->geom_bodyid[contact.geom1] == 0;
          if (first || second) {
            double local[6], world[3];
            mj_contactForce(model, data, k, local);
            mju_mulMatTVec(world, contact.frame, local, 3, 3);
            vertical_load[i] += (second ? 1 : -1) * world[2];
          }
        }
      }
    }
    for (int i = 0; i < 4; ++i)
      residual[counter++] = track_load ? (vertical_load[i] - ref[58+i]) / 40.0 : 0;
  }

  // W36: phase describes requested support, force describes actual support.
  const double* stable_support=GetCustomNumericData(model,"r12_stable_support");
  const bool moving_support=stable_support && current_mode_==kModeQuadruped &&
                            gait!=kGaitStand;
  double support_share[4]={0,0,0,0}, support_target[4]={0,0,0,0};
  double requested_load=0;
  if (moving_support) {
    const double gravity=mju_abs(model->opt.gravity[2]);
    const double mass=model->body_subtreemass[torso_body_id_];
    const double cadence=parameters_[cadence_param_id_];
    const double transfer_time=.5/cadence;
    const double* height=GetCustomNumericData(model,"r10_material_support_height");
    const double dz=height[0]-data->xpos[3*torso_body_id_+2];
    const double az=gravity*mju_tanh(
        dz/(gravity*transfer_time*transfer_time)-comvel[2]/(gravity*transfer_time));
    double sum=0;
    for (A1Foot foot:kFootAll) {
      // Smooth transfer over the existing 1mm domain scale. These weights
      // request landing; they never declare an airborne foot to be loaded.
      support_share[foot]=mju_exp(-step[foot]/stable_support[0]);
      sum+=support_share[foot];
    }
    // Nominal flight is allowed; no hard count of supporting feet is imposed.
    requested_load=mass*(gravity+az)*mju_min(sum/2.,1.);
    for (A1Foot foot:kFootAll)
      support_target[foot]=requested_load*support_share[foot]/sum;
  }

  // S4 N: candidate normal motion, before and during actual predicted contact.
  // The 5 mm approach band precedes the unchanged 1 mm material domain.
  if (soft_landing_) {
    double velocity_residual[4] = {0,0,0,0};
    double indentation_residual[4] = {0,0,0,0};
    constexpr double half_sole = 0.015811388300841896;
    for (A1Foot foot : kFootAll) {
      const double* p = foot_pos[foot];
      const bool material = p[0] >= 0.8+half_sole && p[0] <= 2.2-half_sole &&
                            mju_abs(p[1]) <= 0.6-half_sole;
      const double approach_width = GetNumberOrDefault(0., model, "r11_landing_approach_band");
      const bool approaching = p[0] >= .8+half_sole-approach_width &&
          p[0] <= 2.2-half_sole+approach_width &&
          mju_abs(p[1]) <= .6-half_sole+approach_width;
      if (material || approaching) {
        double z = p[2] - model->geom_size[3*foot_geom_id_[foot]] - flat_ground_height[0];
        double velocity[6];
        mj_objectVelocity(model,data,mjOBJ_GEOM,foot_geom_id_[foot],velocity,0);
        const double* normal_stiffness = GetCustomNumericData(model,"r10_local_contact");
        const bool complete_contact = material && normal_stiffness &&
            GetNumberOrDefault(0,model,"r11_contact_completion")==1;
        double rest=0, loaded_target=0;
        if (complete_contact) {
          const double point[3]={p[0],p[1],z+flat_ground_height[0]};
          rest=R10ContactRest(model,data,point);
          const double load=moving_support ? support_target[foot] :
              model->body_subtreemass[torso_body_id_]*mju_abs(model->opt.gravity[2])/2;
          loaded_target=rest-load/normal_stiffness[0];
          if (moving_support)
            loaded_target=mju_max(loaded_target,-stable_support[1]);
        }
        double gate = 0.5*(1-mju_tanh((z-rest-0.005)/0.002));
        // 1 mm / 20 ms logical replanning interval; approaches zero at the plane.
        double desired_vz = -0.05*mju_tanh(mju_max(z-loaded_target,0)/0.001);
        if (slow_approach_ && (!complete_contact || step[foot]>0))
          desired_vz=mju_max(mju_min(step_velocity[foot],0.),desired_vz);
        velocity_residual[foot] = gate*(mju_min(velocity[5],0)-desired_vz)/0.05;
        // 15.206408*9.81 / (2*375000) m: two-foot equilibrium scale.
        double over = moving_support && complete_contact ? loaded_target-z :
                      -z-0.00019889981664;
        indentation_residual[foot] = material ?
            0.5*(over+mju_sqrt(over*over+1e-12))/0.001 : 0;
      }
    }
    mju_copy(residual+counter,velocity_residual,4); counter+=4;
    mju_copy(residual+counter,indentation_residual,4); counter+=4;
    const double* support_height=GetCustomNumericData(model,"r10_material_support_height");
    if (support_height) {
      bool material_support=false;
      for (A1Foot foot : kFootAll) {
        const double* p=foot_pos[foot];
        material_support |= p[0]>=.8+half_sole && p[0]<=2.2-half_sole &&
                            mju_abs(p[1])<=.6-half_sole;
      }
      residual[counter++]=(material_support || r11_moving_body_support) ?
          (data->xpos[3*torso_body_id_+2]-support_height[0])/.04 : 0;
    }
  }

  // S5-r6: forecast cost near the unchanged true material-domain boundary.
  const double* domain_margin = GetCustomNumericData(model, "r11_domain_margin");
  if (domain_margin) {
    constexpr double half = 0.015811388300841896;
    for (A1Foot foot : kFootAll) {
      const double* p = foot_pos[foot];
      const bool material = p[0]>=.8+half && p[0]<=2.2-half && mju_abs(p[1])<=.6-half;
      double value = 0;
      if (material) {
        const double depth = flat_ground_height[0] + model->geom_size[3*foot_geom_id_[foot]] - p[2];
        const double over = depth-domain_margin[0];
        value = .5*(over+mju_sqrt(over*over+1e-12))/domain_margin[1];
      }
      residual[counter++] = value;
    }
  }

  // W34: the real task terminates at the first forbidden trunk-floor contact.
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

  if (stable_support) {
    double balance[3]={0,0,0}, placement[4]={0,0,0,0};
    double impact[4]={0,0,0,0};
    const double mass=model->body_subtreemass[torso_body_id_];
    const double gravity=mju_abs(model->opt.gravity[2]), mg=mass*gravity;
    double force[4]={0,0,0,0};
    if (moving_support) {
      for (A1Foot foot:kFootAll) {
        const int geom=foot_geom_id_[foot];
        if (R10MaterialFoot(model,data,geom)) {
          force[foot]=R10ContactNormal(model,data,geom);
          const double* p=foot_pos[foot];
          double point[3]={p[0],p[1],p[2]-model->geom_size[3*geom]};
          const double rest=R10ContactRest(model,data,point);
          const double stiffness=GetCustomNumericData(model,"r10_local_contact")[0];
          const double target=mju_max(rest-support_target[foot]/stiffness,
                                      -stable_support[1]);
          const double gap=point[2]-flat_ground_height[0]-target;
          placement[foot]=support_share[foot]*
              .5*(gap+mju_sqrt(gap*gap+1e-12))/.005;
        } else {
          for (int k=0;k<data->ncon;++k) {
            const mjContact& contact=data->contact[k];
            const bool first=contact.geom1==geom &&
                             model->geom_bodyid[contact.geom2]==0;
            const bool second=contact.geom2==geom &&
                              model->geom_bodyid[contact.geom1]==0;
            if (first || second) {
              double local[6],world[3]; mj_contactForce(model,data,k,local);
              mju_mulMatTVec(world,contact.frame,local,3,3);
              force[foot]+=(second ? 1 : -1)*world[2];
            }
          }
        }
        balance[0]+=force[foot];
        balance[1]+=(foot_pos[foot][1]-compos[1])*force[foot];
        balance[2]-=(foot_pos[foot][0]-compos[0])*force[foot];
      }
      balance[0]=(balance[0]-requested_load)/mg;
      // Original trunk length/width scales; load moments coordinate transfer.
      balance[1]/=mg*.1;
      balance[2]/=mg*.2;
    }
    for (A1Foot foot:kFootAll) {
      const double* p=foot_pos[foot];
      constexpr double half=.015811388300841896;
      const double band=GetNumberOrDefault(0.,model,"r11_landing_approach_band");
      const bool approaching=p[0]>=.8+half-band && p[0]<=2.2-half+band &&
                            mju_abs(p[1])<=.6-half+band;
      if (!approaching) continue;
      const double z=p[2]-model->geom_size[3*foot_geom_id_[foot]]-flat_ground_height[0];
      double velocity[6];
      mj_objectVelocity(model,data,mjOBJ_GEOM,foot_geom_id_[foot],velocity,0);
      const double down=mju_max(-velocity[5],0.);
      const double hold=model->opt.timestep*
          GetNumberOrDefault(1,model,"s1_control_hold_substeps");
      // One held-input interval plus stopping distance at one g. The remaining
      // margin is to the original .8mm prediction onset, not a relaxed domain.
      // This is a soft viability objective, not a guarantee of foot deceleration.
      const double over=hold*down+down*down/(2*gravity)-(z+stable_support[1]);
      impact[foot]=.5*(over+mju_sqrt(over*over+1e-12))/stable_support[2];
    }
    mju_copy(residual+counter,balance,3); counter+=3;
    mju_copy(residual+counter,placement,4); counter+=4;
    mju_copy(residual+counter,impact,4); counter+=4;
  }

  // sensor dim sanity check
  CheckSensorDim(model, counter);
}

//  ============  transition  ============
void QuadrupedFlat::TransitionLocked(mjModel* model, mjData* data) {
  // ---------- handle mjData reset ----------
  if (data->time < residual_.last_transition_time_ ||
      residual_.last_transition_time_ == -1) {
    if (mode != ResidualFn::kModeQuadruped && mode != ResidualFn::kModeBiped) {
      mode = ResidualFn::kModeQuadruped;  // mode stateful, switch to Quadruped
    }
    residual_.last_transition_time_ = residual_.phase_start_time_ =
        residual_.phase_start_ = data->time;
  }

  // ---------- prevent forbidden mode transitions ----------
  // switching mode, not from quadruped
  if (mode != residual_.current_mode_ &&
      residual_.current_mode_ != ResidualFn::kModeQuadruped) {
    // switch into stateful mode only allowed from Quadruped
    if (mode == ResidualFn::kModeWalk || mode == ResidualFn::kModeFlip) {
      mode = ResidualFn::kModeQuadruped;
    }
  }

  // ---------- handle phase velocity change ----------
  double phase_velocity = 2 * mjPI * parameters[residual_.cadence_param_id_];
  if (phase_velocity != residual_.phase_velocity_) {
    residual_.phase_start_ = residual_.GetPhase(data->time);
    residual_.phase_start_time_ = data->time;
    residual_.phase_velocity_ = phase_velocity;
  }


  // ---------- automatic gait switching ----------
  double* comvel = SensorByName(model, data, "torso_subtreelinvel");
  double beta = mju_exp(-(data->time - residual_.last_transition_time_) /
                        ResidualFn::kAutoGaitFilter);
  residual_.com_vel_[0] = beta * residual_.com_vel_[0] + (1 - beta) * comvel[0];
  residual_.com_vel_[1] = beta * residual_.com_vel_[1] + (1 - beta) * comvel[1];
  // TODO(b/268398978): remove reinterpret, int64_t business
  int auto_switch =
      ReinterpretAsInt(parameters[residual_.gait_switch_param_id_]);
  if (mode == ResidualFn::kModeBiped) {
    // biped always trots
    parameters[residual_.gait_param_id_] =
        ReinterpretAsDouble(ResidualFn::kGaitTrot);
  } else if (auto_switch) {
    double com_speed = mju_norm(residual_.com_vel_, 2);
    for (int64_t gait : ResidualFn::kGaitAll) {
      // scramble requires a non-static gait
      if (mode == ResidualFn::kModeScramble && gait == ResidualFn::kGaitStand)
        continue;
      bool lower = com_speed > ResidualFn::kGaitAuto[gait];
      bool upper = gait == ResidualFn::kGaitGallop ||
                   com_speed <= ResidualFn::kGaitAuto[gait + 1];
      bool wait = mju_abs(residual_.gait_switch_time_ - data->time) >
                  ResidualFn::kAutoGaitMinTime;
      if (lower && upper && wait) {
        parameters[residual_.gait_param_id_] = ReinterpretAsDouble(gait);
        residual_.gait_switch_time_ = data->time;
      }
    }
  }


  // ---------- handle gait switch, manual or auto ----------
  double gait_selection = parameters[residual_.gait_param_id_];
  if (gait_selection != residual_.current_gait_) {
    residual_.current_gait_ = gait_selection;
    ResidualFn::A1Gait gait = residual_.GetGait();
    parameters[residual_.duty_param_id_] = ResidualFn::kGaitParam[gait][0];
    parameters[residual_.cadence_param_id_] = ResidualFn::kGaitParam[gait][1];
    parameters[residual_.amplitude_param_id_] = ResidualFn::kGaitParam[gait][2];
    weight[residual_.balance_cost_id_] = ResidualFn::kGaitParam[gait][3];
    weight[residual_.upright_cost_id_] = ResidualFn::kGaitParam[gait][4];
    weight[residual_.height_cost_id_] = ResidualFn::kGaitParam[gait][5];
  }


  // ---------- Walk ----------
  double* goal_pos = data->mocap_pos + 3*residual_.goal_mocap_id_;
  if (mode == ResidualFn::kModeWalk) {
    double angvel = parameters[ParameterIndex(model, "Walk turn")];
    double speed = parameters[ParameterIndex(model, "Walk speed")];

    // current torso direction
    double* torso_xmat = data->xmat + 9*residual_.torso_body_id_;
    double forward[2] = {torso_xmat[0], torso_xmat[3]};
    mju_normalize(forward, 2);
    double leftward[2] = {-forward[1], forward[0]};

    // switching into Walk or parameters changed, reset task state
    if (mode != residual_.current_mode_ || residual_.angvel_ != angvel ||
        residual_.speed_ != speed) {
      // save time
      residual_.mode_start_time_ = data->time;

      // save current speed and angvel
      residual_.speed_ = speed;
      residual_.angvel_ = angvel;

      // compute and save rotation axis / walk origin
      double axis[2] = {data->xpos[3*residual_.torso_body_id_],
                        data->xpos[3*residual_.torso_body_id_+1]};
      if (mju_abs(angvel) > ResidualFn::kMinAngvel) {
        // don't allow turning with very small angvel
        double d = speed / angvel;
        axis[0] += d * leftward[0];
        axis[1] += d * leftward[1];
      }
      residual_.position_[0] = axis[0];
      residual_.position_[1] = axis[1];

      // save vector from axis to initial goal position
      residual_.heading_[0] = goal_pos[0] - axis[0];
      residual_.heading_[1] = goal_pos[1] - axis[1];
    }

    // move goal
    double time = data->time - residual_.mode_start_time_;
    residual_.Walk(goal_pos, time);
  }


  // ---------- Flip ----------
  double* compos = SensorByName(model, data, "torso_subtreecom");
  if (mode == ResidualFn::kModeFlip) {
    // switching into Flip, reset task state
    if (mode != residual_.current_mode_) {
      // save time
      residual_.mode_start_time_ = data->time;

      // save body orientation, ground height
      mju_copy4(residual_.orientation_,
                data->xquat + 4 * residual_.torso_body_id_);
      residual_.ground_ = Ground(model, data, compos);

      // save parameters
      residual_.save_weight_ = weight;
      residual_.save_gait_switch_ = parameters[residual_.gait_switch_param_id_];

      // set parameters
      weight[CostTermByName(model, "Upright")] = 0.2;
      weight[CostTermByName(model, "Height")] = 5;
      weight[CostTermByName(model, "Position")] = 0;
      weight[CostTermByName(model, "Gait")] = 0;
      weight[CostTermByName(model, "Balance")] = 0;
      weight[CostTermByName(model, "Effort")] = 0.005;
      weight[CostTermByName(model, "Posture")] = 0.1;
      parameters[residual_.gait_switch_param_id_] = ReinterpretAsDouble(0);
    }

    // time from start of Flip
    double flip_time = data->time - residual_.mode_start_time_;

    if (flip_time >=
        residual_.jump_time_ + residual_.flight_time_ + residual_.land_time_) {
      // Flip ended, back to Quadruped, restore values
      mode = ResidualFn::kModeQuadruped;
      weight = residual_.save_weight_;
      parameters[residual_.gait_switch_param_id_] = residual_.save_gait_switch_;
      goal_pos[0] = data->site_xpos[3*residual_.head_site_id_ + 0];
      goal_pos[1] = data->site_xpos[3*residual_.head_site_id_ + 1];
    }
  }

  // save mode
  residual_.current_mode_ = static_cast<ResidualFn::A1Mode>(mode);
  residual_.last_transition_time_ = data->time;
}

// colors of visualisation elements drawn in ModifyScene()
constexpr float kStepRgba[4] = {0.6, 0.8, 0.2, 1};  // step-height cylinders
constexpr float kHullRgba[4] = {0.4, 0.2, 0.8, 1};  // convex hull
constexpr float kAvgRgba[4] = {0.4, 0.2, 0.8, 1};   // average foot position
constexpr float kCapRgba[4] = {0.3, 0.3, 0.8, 1};   // capture point
constexpr float kPcpRgba[4] = {0.5, 0.5, 0.2, 1};   // projected capture point

// draw task-related geometry in the scene
void QuadrupedFlat::ModifyScene(const mjModel* model, const mjData* data,
                           mjvScene* scene) const {
  // flip target pose
  if (residual_.current_mode_ == ResidualFn::kModeFlip) {
    double flip_time = data->time - residual_.mode_start_time_;
    double* torso_pos = data->xpos + 3*residual_.torso_body_id_;
    double pos[3] = {torso_pos[0], torso_pos[1],
                     residual_.FlipHeight(flip_time)};
    double quat[4];
    residual_.FlipQuat(quat, flip_time);
    double mat[9];
    mju_quat2Mat(mat, quat);
    double size[3] = {0.25, 0.15, 0.05};
    float rgba[4] = {0, 1, 0, 0.5};
    AddGeom(scene, mjGEOM_BOX, size, pos, mat, rgba);

    // don't draw anything else during flip
    return;
  }

  // current foot positions
  double* foot_pos[ResidualFn::kNumFoot];
  for (ResidualFn::A1Foot foot : ResidualFn::kFootAll)
    foot_pos[foot] = data->geom_xpos + 3 * residual_.foot_geom_id_[foot];

  // stance and flight positions
  double flight_pos[ResidualFn::kNumFoot][3];
  double stance_pos[ResidualFn::kNumFoot][3];
  // set to foot horizontal position:
  for (ResidualFn::A1Foot foot : ResidualFn::kFootAll) {
    flight_pos[foot][0] = stance_pos[foot][0] = foot_pos[foot][0];
    flight_pos[foot][1] = stance_pos[foot][1] = foot_pos[foot][1];
  }

  // ground height below feet
  double ground[ResidualFn::kNumFoot];
  for (ResidualFn::A1Foot foot : ResidualFn::kFootAll) {
    ground[foot] = Ground(model, data, foot_pos[foot]);
  }

  // step heights
  ResidualFn::A1Gait gait = residual_.GetGait();
  double step[ResidualFn::kNumFoot];
  residual_.FootStep(step, residual_.GetPhase(data->time), gait);

  // draw step height
  for (ResidualFn::A1Foot foot : ResidualFn::kFootAll) {
    stance_pos[foot][2] = model->geom_size[3*residual_.foot_geom_id_[foot]] + ground[foot];
    if (residual_.current_mode_ == ResidualFn::kModeBiped) {
      // skip "hands" in biped mode
      bool handstand =
          ReinterpretAsInt(parameters[residual_.biped_type_param_id_]);
      bool front_hand = !handstand && (foot == ResidualFn::kFootFL ||
                                       foot == ResidualFn::kFootFR);
      bool back_hand = handstand && (foot == ResidualFn::kFootHL ||
                                     foot == ResidualFn::kFootHR);
      if (front_hand || back_hand) continue;
    }
    if (step[foot]) {
      flight_pos[foot][2] = model->geom_size[3*residual_.foot_geom_id_[foot]] + step[foot] + ground[foot];
      AddConnector(scene, mjGEOM_CYLINDER, ResidualFn::kFootRadius,
                   stance_pos[foot], flight_pos[foot], kStepRgba);
    }
  }

  // support polygon (currently unused for cost)
  double polygon[2*ResidualFn::kNumFoot];
  for (ResidualFn::A1Foot foot : ResidualFn::kFootAll) {
    polygon[2*foot] = foot_pos[foot][0];
    polygon[2*foot + 1] = foot_pos[foot][1];
  }
  int hull[ResidualFn::kNumFoot];
  int num_hull = Hull2D(hull, ResidualFn::kNumFoot, polygon);
  for (int i=0; i < num_hull; i++) {
    int j = (i + 1) % num_hull;
    AddConnector(scene, mjGEOM_CAPSULE, ResidualFn::kFootRadius/2,
                 stance_pos[hull[i]], stance_pos[hull[j]], kHullRgba);
  }

  // capture point
  bool is_biped = residual_.current_mode_ == ResidualFn::kModeBiped;
  double height_goal =
      is_biped ? ResidualFn::kHeightBiped : ResidualFn::kHeightQuadruped;
  double fall_time = mju_sqrt(2*height_goal / residual_.gravity_);
  double capture[3];
  double* compos = SensorByName(model, data, "torso_subtreecom");
  double* comvel = SensorByName(model, data, "torso_subtreelinvel");
  mju_addScl3(capture, compos, comvel, fall_time);

  // ground under CoM
  double com_ground = Ground(model, data, compos);

  // average foot position
  double feet_pos[3];
  residual_.AverageFootPos(feet_pos, foot_pos);
  feet_pos[2] = com_ground;

  double foot_size[3] = {ResidualFn::kFootRadius, 0, 0};

  // average foot position
  AddGeom(scene, mjGEOM_SPHERE, foot_size, feet_pos, /*mat=*/nullptr, kAvgRgba);

  // capture point
  capture[2] = com_ground;
  AddGeom(scene, mjGEOM_SPHERE, foot_size, capture, /*mat=*/nullptr, kCapRgba);

  // capture point, projected onto hull
  double pcp2[2];
  NearestInHull(pcp2, capture, polygon, hull, num_hull);
  double pcp[3] = {pcp2[0], pcp2[1], com_ground};
  AddGeom(scene, mjGEOM_SPHERE, foot_size, pcp, /*mat=*/nullptr, kPcpRgba);
}

//  ============  task-state utilities  ============
// save task-related ids
void QuadrupedFlat::ResetLocked(const mjModel* model) {
  // ----------  task identifiers  ----------
  residual_.gait_param_id_ = ParameterIndex(model, "select_Gait");
  residual_.gait_switch_param_id_ = ParameterIndex(model, "select_Gait switch");
  residual_.flip_dir_param_id_ = ParameterIndex(model, "select_Flip dir");
  residual_.biped_type_param_id_ = ParameterIndex(model, "select_Biped type");
  residual_.cadence_param_id_ = ParameterIndex(model, "Cadence");
  residual_.amplitude_param_id_ = ParameterIndex(model, "Amplitude");
  residual_.duty_param_id_ = ParameterIndex(model, "Duty ratio");
  residual_.arm_posture_param_id_ = ParameterIndex(model, "Arm posture");
  residual_.balance_cost_id_ = CostTermByName(model, "Balance");
  residual_.upright_cost_id_ = CostTermByName(model, "Upright");
  residual_.height_cost_id_ = CostTermByName(model, "Height");

  residual_.soft_landing_ = GetNumberOrDefault(0, model, "r10_normal_landing") == 1;
  residual_.slow_approach_ = GetNumberOrDefault(0, model, "r11_slow_approach") == 1;

  // ----------  model identifiers  ----------
  residual_.torso_body_id_ = mj_name2id(model, mjOBJ_XBODY, "trunk");
  if (residual_.torso_body_id_ < 0) mju_error("body 'trunk' not found");

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


  residual_.head_site_id_ = mj_name2id(model, mjOBJ_SITE, "head");
  if (residual_.head_site_id_ < 0) mju_error("site 'head' not found");

  int goal_id = mj_name2id(model, mjOBJ_XBODY, "goal");
  if (goal_id < 0) mju_error("body 'goal' not found");

  residual_.goal_mocap_id_ = model->body_mocapid[goal_id];
  if (residual_.goal_mocap_id_ < 0) mju_error("body 'goal' is not mocap");

  // foot geom ids
  int foot_index = 0;
  for (const char* footname : {"FL", "RL", "FR", "RR"}) {
    int foot_id = mj_name2id(model, mjOBJ_GEOM, footname);
    if (foot_id < 0) mju_error_s("geom '%s' not found", footname);
    residual_.foot_geom_id_[foot_index] = foot_id;
    foot_index++;
  }

  // shoulder body ids
  int shoulder_index = 0;
  for (const char* shouldername : {"FL_hip", "RL_hip", "FR_hip", "RR_hip"}) {
    int foot_id = mj_name2id(model, mjOBJ_BODY, shouldername);
    if (foot_id < 0) mju_error_s("body '%s' not found", shouldername);
    residual_.shoulder_body_id_[shoulder_index] = foot_id;
    shoulder_index++;
  }

  // ----------  derived kinematic quantities for Flip  ----------
  residual_.gravity_ = mju_norm3(model->opt.gravity);
  // velocity at takeoff
  residual_.jump_vel_ =
      mju_sqrt(2 * residual_.gravity_ *
               (ResidualFn::kMaxHeight - ResidualFn::kLeapHeight));
  // time in flight phase
  residual_.flight_time_ = 2 * residual_.jump_vel_ / residual_.gravity_;
  // acceleration during jump phase
  residual_.jump_acc_ =
      residual_.jump_vel_ * residual_.jump_vel_ /
      (2 * (ResidualFn::kLeapHeight - ResidualFn::kCrouchHeight));
  // time in crouch sub-phase of jump
  residual_.crouch_time_ =
      mju_sqrt(2 * (ResidualFn::kHeightQuadruped - ResidualFn::kCrouchHeight) /
               residual_.jump_acc_);
  // time in leap sub-phase of jump
  residual_.leap_time_ = residual_.jump_vel_ / residual_.jump_acc_;
  // jump total time
  residual_.jump_time_ = residual_.crouch_time_ + residual_.leap_time_;
  // velocity at beginning of crouch
  residual_.crouch_vel_ = -residual_.jump_acc_ * residual_.crouch_time_;
  // time of landing phase
  residual_.land_time_ =
      2 * (ResidualFn::kLeapHeight - ResidualFn::kHeightQuadruped) /
      residual_.jump_vel_;
  // acceleration during landing
  residual_.land_acc_ = residual_.jump_vel_ / residual_.land_time_;
  // rotational velocity during flight phase (rotates 1.25 pi)
  residual_.flight_rot_vel_ = 1.25 * mjPI / residual_.flight_time_;
  // rotational velocity at start of leap (rotates 0.5 pi)
  residual_.jump_rot_vel_ =
      mjPI / residual_.leap_time_ - residual_.flight_rot_vel_;
  // rotational acceleration during leap (rotates 0.5 pi)
  residual_.jump_rot_acc_ =
      (residual_.flight_rot_vel_ - residual_.jump_rot_vel_) /
      residual_.leap_time_;
  // rotational deceleration during land (rotates 0.25 pi)
  residual_.land_rot_acc_ =
      2 * (residual_.flight_rot_vel_ * residual_.land_time_ - mjPI / 4) /
      (residual_.land_time_ * residual_.land_time_);
}

// compute average foot position, depending on mode
void QuadrupedFlat::ResidualFn::AverageFootPos(
    double avg_foot_pos[3], double* foot_pos[kNumFoot]) const {
  if (current_mode_ == kModeBiped) {
    int handstand = ReinterpretAsInt(parameters_[biped_type_param_id_]);
    if (handstand) {
      mju_add3(avg_foot_pos, foot_pos[kFootFL], foot_pos[kFootFR]);
    } else {
      mju_add3(avg_foot_pos, foot_pos[kFootHL], foot_pos[kFootHR]);
    }
    mju_scl3(avg_foot_pos, avg_foot_pos, 0.5);
  } else {
    mju_add3(avg_foot_pos, foot_pos[kFootHL], foot_pos[kFootHR]);
    mju_addTo3(avg_foot_pos, foot_pos[kFootFL]);
    mju_addTo3(avg_foot_pos, foot_pos[kFootFR]);
    mju_scl3(avg_foot_pos, avg_foot_pos, 0.25);
  }
}

// return phase as a function of time
double QuadrupedFlat::ResidualFn::GetPhase(double time) const {
  return phase_start_ + (time - phase_start_time_) * phase_velocity_;
}

// horizontal Walk trajectory
void QuadrupedFlat::ResidualFn::Walk(double pos[2], double time) const {
  if (mju_abs(angvel_) < kMinAngvel) {
    // no rotation, go in straight line
    double forward[2] = {heading_[0], heading_[1]};
    mju_normalize(forward, 2);
    pos[0] = position_[0] + heading_[0] + time*speed_*forward[0];
    pos[1] = position_[1] + heading_[1] + time*speed_*forward[1];
  } else {
    // walk on a circle
    double angle = time * angvel_;
    double mat[4] = {mju_cos(angle), -mju_sin(angle),
                     mju_sin(angle),  mju_cos(angle)};
    mju_mulMatVec(pos, mat, heading_, 2, 2);
    pos[0] += position_[0];
    pos[1] += position_[1];
  }
}

// get gait
QuadrupedFlat::ResidualFn::A1Gait QuadrupedFlat::ResidualFn::GetGait() const {
  if (current_mode_ == kModeBiped)
    return kGaitTrot;
  return static_cast<A1Gait>(ReinterpretAsInt(current_gait_));
}

// return normalized target step height
double QuadrupedFlat::ResidualFn::StepHeight(double time, double footphase,
                                             double duty_ratio) const {
  double angle = fmod(time + mjPI - footphase, 2*mjPI) - mjPI;
  double value = 0;
  if (duty_ratio < 1) {
    angle *= 0.5 / (1 - duty_ratio);
    value = mju_cos(mju_clip(angle, -mjPI/2, mjPI/2));
    if (soft_landing_) {
      double u = (mju_clip(angle, -mjPI/2, mjPI/2) + mjPI/2) / mjPI;
      double s = u < 0.5 ? 2*u : 2*(1-u);
      value = s*s*s*(10 + s*(-15 + 6*s));
    }
  }
  return mju_abs(value) < 1e-6 ? 0.0 : value;
}

// compute target step height for all feet
// A C2 swing with a separately timed final 5 mm approach.
// S5-r20: original2Hz cadence with a low-clearance, velocity-compatible swing.
void QuadrupedFlat::ResidualFn::FootStep(double step[kNumFoot], double time,
                                        A1Gait gait, double* velocity) const {
  const double amplitude=parameters_[amplitude_param_id_];
  const double duty_ratio=parameters_[duty_param_id_];
  if (velocity) mju_zero(velocity,kNumFoot);
  for (A1Foot foot : kFootAll) {
    const double footphase=2*mjPI*kGaitPhase[gait][foot];
    if (!slow_approach_ || duty_ratio>=1 || amplitude==0) {
      step[foot]=amplitude*StepHeight(time,footphase,duty_ratio);
      continue;
    }
    const double angle=(fmod(time+mjPI-footphase,2*mjPI)-mjPI)*.5/(1-duty_ratio);
    if (angle<=-mjPI/2 || angle>=mjPI/2) {step[foot]=0;continue;}
    const double swing_duration=(1-duty_ratio)*2*mjPI/phase_velocity_;
    constexpr double descent_duration=.20;
    const double rise_duration=swing_duration-descent_duration;
    const double t=(angle+mjPI/2)/mjPI*swing_duration;
    const bool rising=t<rise_duration;
    const double duration=rising ? rise_duration : descent_duration;
    const double u=rising ? t/duration : (t-rise_duration)/duration;
    const double start=rising ? 0 : amplitude;
    const double change=rising ? amplitude : -amplitude;
    const double smooth=u*u*u*(10+u*(-15+6*u));
    step[foot]=start+change*smooth;
    if (velocity) velocity[foot]=change/duration*30*u*u*(1-u)*(1-u);
  }
}

// height during flip
double QuadrupedFlat::ResidualFn::FlipHeight(double time) const {
  if (time >= jump_time_ + flight_time_ + land_time_) {
    return kHeightQuadruped + ground_;
  }
  double h = 0;
  if (time < jump_time_) {
    h = kHeightQuadruped + time * crouch_vel_ + 0.5 * time * time * jump_acc_;
  } else if (time >= jump_time_ && time < jump_time_ + flight_time_) {
    time -= jump_time_;
    h = kLeapHeight + jump_vel_*time - 0.5*9.81*time*time;
  } else if (time >= jump_time_ + flight_time_) {
    time -= jump_time_ + flight_time_;
    h = kLeapHeight - jump_vel_*time + 0.5*land_acc_*time*time;
  }
  return h + ground_;
}

// orientation during flip
//  total rotation = leap + flight + land
//            2*pi = pi/2 + 5*pi/4 + pi/4
void QuadrupedFlat::ResidualFn::FlipQuat(double quat[4], double time) const {
  double angle = 0;
  if (time >= jump_time_ + flight_time_ + land_time_) {
    angle = 2*mjPI;
  } else if (time >= crouch_time_ && time < jump_time_) {
    time -= crouch_time_;
    angle = 0.5 * jump_rot_acc_ * time * time + jump_rot_vel_ * time;
  } else if (time >= jump_time_ && time < jump_time_ + flight_time_) {
    time -= jump_time_;
    angle = mjPI/2 + flight_rot_vel_ * time;
  } else if (time >= jump_time_ + flight_time_) {
    time -= jump_time_ + flight_time_;
    angle = 1.75*mjPI + flight_rot_vel_*time - 0.5*land_rot_acc_ * time * time;
  }
  int flip_dir = ReinterpretAsInt(parameters_[flip_dir_param_id_]);
  double axis[3] = {0, flip_dir ? 1.0 : -1.0, 0};
  mju_axisAngle2Quat(quat, axis, angle);
  mju_mulQuat(quat, orientation_, quat);
}


// --------------------- Residuals for quadruped task --------------------
//   Number of residuals: 4
//     Residual (0): position_z - average(foot position)_z - height_goal
//     Residual (1): position - goal_position
//     Residual (2): orientation - goal_orientation
//     Residual (3): control
//   Number of parameters: 1
//     Parameter (1): height_goal
// -----------------------------------------------------------------------
void QuadrupedHill::ResidualFn::Residual(const mjModel* model,
                                         const mjData* data,
                                         double* residual) const {
  // ---------- Residual (0) ----------
  // standing height goal
  double height_goal = parameters_[0];

  // system's standing height
  double standing_height = SensorByName(model, data, "position")[2];

  // average foot height
  double FRz = SensorByName(model, data, "FR")[2];
  double FLz = SensorByName(model, data, "FL")[2];
  double RRz = SensorByName(model, data, "RR")[2];
  double RLz = SensorByName(model, data, "RL")[2];
  double avg_foot_height = 0.25 * (FRz + FLz + RRz + RLz);

  residual[0] = (standing_height - avg_foot_height) - height_goal;

  // ---------- Residual (1) ----------
  // goal position
  const double* goal_position = data->mocap_pos;

  // system's position
  double* position = SensorByName(model, data, "position");

  // position error
  mju_sub3(residual + 1, position, goal_position);

  // ---------- Residual (2) ----------
  // goal orientation
  double goal_rotmat[9];
  const double* goal_orientation = data->mocap_quat;
  mju_quat2Mat(goal_rotmat, goal_orientation);

  // system's orientation
  double body_rotmat[9];
  double* orientation = SensorByName(model, data, "orientation");
  mju_quat2Mat(body_rotmat, orientation);

  mju_sub(residual + 4, body_rotmat, goal_rotmat, 9);

  // ---------- Residual (3) ----------
  mju_copy(residual + 13, data->ctrl, model->nu);
}

// -------- Transition for quadruped task --------
//   If quadruped is within tolerance of goal ->
//   set goal to next from keyframes.
// -----------------------------------------------
void QuadrupedHill::TransitionLocked(mjModel* model, mjData* data) {
  // set mode to GUI selection
  if (mode > 0) {
    residual_.current_mode_ = mode - 1;
  } else {
    // ---------- Compute tolerance ----------
    // goal position
    const double* goal_position = data->mocap_pos;

    // goal orientation
    const double* goal_orientation = data->mocap_quat;

    // system's position
    double* position = SensorByName(model, data, "position");

    // system's orientation
    double* orientation = SensorByName(model, data, "orientation");

    // position error
    double position_error[3];
    mju_sub3(position_error, position, goal_position);
    double position_error_norm = mju_norm3(position_error);

    // orientation error
    double geodesic_distance =
        1.0 - mju_abs(mju_dot(goal_orientation, orientation, 4));

    // ---------- Check tolerance ----------
    double tolerance = 1.5e-1;
    if (position_error_norm <= tolerance && geodesic_distance <= tolerance) {
      // update task state
      residual_.current_mode_ += 1;
      if (residual_.current_mode_ == model->nkey) {
        residual_.current_mode_ = 0;
      }
    }
  }

  // ---------- Set goal ----------
  mju_copy3(data->mocap_pos, model->key_mpos + 3 * residual_.current_mode_);
  mju_copy4(data->mocap_quat, model->key_mquat + 4 * residual_.current_mode_);
}

}  // namespace mjpc
