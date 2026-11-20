#pragma once
#include <mujoco/mujoco.h>

// userdata[0:62] is the frozen R8 stand reference. The remaining read-only
// packet is captured once per 20 ms policy epoch and cloned with each mjData.
constexpr int W55Epoch=62, W55FootObservation=63, W55Surface=79;
constexpr int W55Nx=70, W55Ny=60, W55Data=W55Surface+W55Nx*W55Ny;
constexpr int W55Diag=24;
extern "C" {
int w55_project(const mjModel*,mjData*,const double*,double*,double*);
void w55_recovery(const mjModel*,mjData*,double*);
int w55_checks(const mjModel*,mjData*,double*,double*);
void w55_native_contact_filter();
void w55_native_step(const mjModel*,mjData*,bool integrate);
}
