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

#include "mjpc/planners/model_derivatives.h"

#include <algorithm>

#include <mujoco/mujoco.h>
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"
#include "mjpc/r12_path.h"

namespace mjpc {

// allocate memory
void ModelDerivatives::Allocate(int dim_state_derivative, int dim_action,
                                int dim_sensor, int T) {
  A.resize(dim_state_derivative * dim_state_derivative * T);
  B.resize(dim_state_derivative * dim_action * T);
  C.resize(dim_sensor * dim_state_derivative * T);
  D.resize(dim_sensor * dim_action * T);
}

// reset memory to zeros
void ModelDerivatives::Reset(int dim_state_derivative, int dim_action,
                             int dim_sensor, int T) {
  std::fill(A.begin(),
            A.begin() + T * dim_state_derivative * dim_state_derivative, 0.0);
  std::fill(B.begin(), B.begin() + T * dim_state_derivative * dim_action, 0.0);
  std::fill(C.begin(), C.begin() + T * dim_sensor * dim_state_derivative, 0.0);
  std::fill(D.begin(), D.begin() + T * dim_sensor * dim_action, 0.0);
}

// compute derivatives at all time steps
void ModelDerivatives::Compute(const mjModel* m,
                               const std::vector<UniqueMjData>& data,
                               const double* x, const double* u,
                               const double* h, int dim_state,
                               int dim_state_derivative, int dim_action,
                               int dim_sensor, int T, double tol, int mode,
                               ThreadPool& pool, int skip) {
  // reset indices
  evaluate_.clear();
  interpolate_.clear();

  // evaluate indices
  int s = skip + 1;
  evaluate_.push_back(0);
  for (int t = s; t < T - s; t += s) {
    evaluate_.push_back(t);
  }
  // The last interior node may already be present when skip==0.
  // One job owns each output block during held-input A/B composition.
  if (evaluate_.back() != T - 2) evaluate_.push_back(T - 2);
  evaluate_.push_back(T - 1);

  // interpolate indices
  for (int t = 0, e = 0; t < T; t++) {
    if (e == evaluate_.size() || evaluate_[e] > t) {
      interpolate_.push_back(t);
    } else {
      e++;
    }
  }

  // evaluate derivatives
  int count_before = pool.GetCount();
  for (int t : evaluate_) {
    pool.Schedule([&m, &data, &A = A, &B = B, &C = C, &D = D, &x, &u, &h,
                   dim_state, dim_state_derivative, dim_action, dim_sensor, tol,
                   mode, t, T]() {
      mjData* d = data[ThreadPool::WorkerId()].get();
      // set state
      SetState(m, d, x + t * dim_state);
      d->time = h[t];

      // set action
      mju_copy(d->ctrl, u + t * dim_action, dim_action);

      const auto path_rows=R12PathRows(m);
      if (!path_rows.empty() && t<T-1) {
        const int nx=dim_state_derivative,nu=dim_action,ns=dim_sensor;
        double* total_a=DataAt(A,t*nx*nx);
        double* total_b=DataAt(B,t*nx*nu);
        double* total_c=DataAt(C,t*ns*nx);
        double* total_d=DataAt(D,t*ns*nu);
        std::vector<double> prefix_a(nx*nx,0.),prefix_b(nx*nu,0.);
        std::vector<double> next_a(nx*nx),next_b(nx*nu),next_c(ns*nx),next_d(ns*nu);
        std::vector<double> peak(ns);
        std::vector<bool> path(ns,false);
        for (int row:path_rows) path[row]=true;
        for (int j=0;j<nx;++j) prefix_a[j*nx+j]=1.;
        const int substeps=GetNumberOrDefault(1,m,"s1_control_hold_substeps");
        for (int sub=0;sub<substeps;++sub) {
          // FD returns this substep's A/B/C/D and restores its input state.
          mjd_transitionFD(m,d,tol,mode,next_a.data(),next_b.data(),
                           next_c.data(),next_d.data());
          // Recompute the nominal sensor value, exactly as rollout does.
          mj_step(m,d);
          for (int row=0;row<ns;++row) {
            if (sub && (!path[row] || d->sensordata[row]<=peak[row])) continue;
            peak[row]=d->sensordata[row];
            mju_mulMatMat(total_c+row*nx,next_c.data()+row*nx,
                          prefix_a.data(),1,nx,nx);
            mju_mulMatMat(total_d+row*nu,next_c.data()+row*nx,
                          prefix_b.data(),1,nx,nu);
            mju_addTo(total_d+row*nu,next_d.data()+row*nu,nu);
          }
          mju_mulMatMat(total_a,next_a.data(),prefix_a.data(),nx,nx,nx);
          mju_mulMatMat(total_b,next_a.data(),prefix_b.data(),nx,nx,nu);
          mju_addTo(total_b,next_b.data(),nx*nu);
          mju_copy(prefix_a.data(),total_a,nx*nx);
          mju_copy(prefix_b.data(),total_b,nx*nu);
        }
      } else if (t == T - 1) {
        // Jacobians
        mjd_transitionFD(m, d, tol, mode, nullptr, nullptr,
                         DataAt(C, t * (dim_sensor * dim_state_derivative)),
                         nullptr);
      } else {
        // Node-start C/D and the first 1ms state transition.
        mjd_transitionFD(
            m, d, tol, mode,
            DataAt(A, t * (dim_state_derivative * dim_state_derivative)),
            DataAt(B, t * (dim_state_derivative * dim_action)),
            DataAt(C, t * (dim_sensor * dim_state_derivative)),
            DataAt(D, t * (dim_sensor * dim_action)));
        const int control_substeps = GetNumberOrDefault(1,m,"s1_control_hold_substeps");
        if (control_substeps > 1) {
          const int nx=dim_state_derivative, nu=dim_action;
          double* total_a=DataAt(A,t*nx*nx);
          double* total_b=DataAt(B,t*nx*nu);
          std::vector<double> old_a(nx*nx),old_b(nx*nu),next_a(nx*nx),next_b(nx*nu);
          for (int substep=1;substep<control_substeps;++substep) {
            mju_copy(old_a.data(),total_a,nx*nx);mju_copy(old_b.data(),total_b,nx*nu);
            mj_step(m,d);
            mjd_transitionFD(m,d,tol,mode,next_a.data(),next_b.data(),nullptr,nullptr);
            mju_mulMatMat(total_a,next_a.data(),old_a.data(),nx,nx,nx);
            mju_mulMatMat(total_b,next_a.data(),old_b.data(),nx,nx,nu);
            mju_addTo(total_b,next_b.data(),nx*nu);
          }
        }
      }
    });
  }
  pool.WaitCount(count_before + evaluate_.size());
  pool.ResetCount();

  // interpolate derivatives
  count_before = pool.GetCount();
  for (int t : interpolate_) {
    pool.Schedule([&A = A, &B = B, &C = C, &D = D, &evaluate_ = this->evaluate_,
                   dim_state_derivative, dim_action, dim_sensor, t]() {
      // find interval
      int bounds[2];
      FindInterval(bounds, evaluate_, t, evaluate_.size());
      int e0 = evaluate_[bounds[0]];
      int e1 = evaluate_[bounds[1]];

      // normalized input
      double tt = double(t - e0) / double(e1 - e0);
      if (bounds[0] == bounds[1]) {
        tt = 0.0;
      }

      // A
      int nA = dim_state_derivative * dim_state_derivative;
      double* Ai = DataAt(A, t * nA);
      const double* AL = DataAt(A, e0 * nA);
      const double* AU = DataAt(A, e1 * nA);

      mju_scl(Ai, AL, 1.0 - tt, nA);
      mju_addToScl(Ai, AU, tt, nA);

      // B
      int nB = dim_state_derivative * dim_action;
      double* Bi = DataAt(B, t * nB);
      const double* BL = DataAt(B, e0 * nB);
      const double* BU = DataAt(B, e1 * nB);

      mju_scl(Bi, BL, 1.0 - tt, nB);
      mju_addToScl(Bi, BU, tt, nB);

      // C
      int nC = dim_sensor * dim_state_derivative;
      double* Ci = DataAt(C, t * nC);
      const double* CL = DataAt(C, e0 * nC);
      const double* CU = DataAt(C, e1 * nC);

      mju_scl(Ci, CL, 1.0 - tt, nC);
      mju_addToScl(Ci, CU, tt, nC);

      // D
      int nD = dim_sensor * dim_action;
      double* Di = DataAt(D, t * nD);
      const double* DL = DataAt(D, e0 * nD);
      const double* DU = DataAt(D, e1 * nD);

      mju_scl(Di, DL, 1.0 - tt, nD);
      mju_addToScl(Di, DU, tt, nD);
    });
  }

  pool.WaitCount(count_before + interpolate_.size());
  pool.ResetCount();
}

}  // namespace mjpc
