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
