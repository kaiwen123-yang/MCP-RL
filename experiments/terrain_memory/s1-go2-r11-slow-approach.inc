// A C2 swing with a separately timed final 5 mm approach.
void QuadrupedFlat::ResidualFn::FootStep(double step[kNumFoot], double time,
                                        A1Gait gait, double* velocity) const {
  double amplitude = parameters_[amplitude_param_id_];
  double duty_ratio = parameters_[duty_param_id_];
  if (velocity) mju_zero(velocity, kNumFoot);
  for (A1Foot foot : kFootAll) {
    double footphase = 2*mjPI*kGaitPhase[gait][foot];
    if (!slow_approach_ || duty_ratio >= 1 || amplitude == 0) {
      step[foot] = amplitude * StepHeight(time, footphase, duty_ratio);
      continue;
    }
    double angle = (fmod(time+mjPI-footphase,2*mjPI)-mjPI)*.5/(1-duty_ratio);
    if (angle<=-mjPI/2 || angle>=mjPI/2) {
      step[foot]=0;
      continue;
    }
    const double swing_duration=(1-duty_ratio)*2*mjPI/phase_velocity_;
    constexpr double approach_duration=.20;
    constexpr double approach_height=.005;
    const double fast_duration=swing_duration-approach_duration;
    const double rise_duration=fast_duration*amplitude/(2*amplitude-approach_height);
    const double high_descent_duration=fast_duration-rise_duration;
    const double t=(angle+mjPI/2)/mjPI*swing_duration;
    double start_height,change,duration,u;
    if (t<rise_duration) {
      start_height=0;change=amplitude;duration=rise_duration;u=t/duration;
    } else if (t<fast_duration) {
      start_height=amplitude;change=approach_height-amplitude;
      duration=high_descent_duration;u=(t-rise_duration)/duration;
    } else {
      start_height=approach_height;change=-approach_height;
      duration=approach_duration;u=(t-fast_duration)/duration;
    }
    const double smooth=u*u*u*(10+u*(-15+6*u));
    step[foot]=start_height+change*smooth;
    if (velocity) velocity[foot]=change/duration*30*u*u*(1-u)*(1-u);
  }
}
