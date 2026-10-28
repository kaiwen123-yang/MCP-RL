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
