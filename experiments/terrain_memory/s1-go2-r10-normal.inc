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
      if (material) {
        double z = p[2] - model->geom_size[3*foot_geom_id_[foot]] - flat_ground_height[0];
        double velocity[6];
        mj_objectVelocity(model,data,mjOBJ_GEOM,foot_geom_id_[foot],velocity,0);
        double gate = 0.5*(1-mju_tanh((z-0.005)/0.002));
        // 1 mm / 20 ms logical replanning interval; approaches zero at the plane.
        double desired_vz = -0.05*mju_tanh(mju_max(z,0)/0.001);
        velocity_residual[foot] = gate*(mju_min(velocity[5],0)-desired_vz)/0.05;
        // 15.206408*9.81 / (2*375000) m: two-foot equilibrium scale.
        double over = -z - 0.00019889981664;
        indentation_residual[foot] = 0.5*(over+mju_sqrt(over*over+1e-12))/0.001;
      }
    }
    mju_copy(residual+counter,velocity_residual,4); counter+=4;
    mju_copy(residual+counter,indentation_residual,4); counter+=4;
  }
