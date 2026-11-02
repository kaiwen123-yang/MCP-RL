// W36: the same active substep defines rollout cost and its Jacobian row.
#ifndef MJPC_R12_PATH_H_
#define MJPC_R12_PATH_H_
#include <vector>
#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"
namespace mjpc {
inline std::vector<int> R12PathRows(const mjModel* m) {
  std::vector<int> rows;
  if (!GetCustomNumericData(m,"r12_substep_path")) return rows;
  for (const char* name:{"MaterialDomainMargin","TrunkClearance","ImpactMargin"}) {
    const int sensor=mj_name2id(m,mjOBJ_SENSOR,name);
    for (int j=0;j<m->sensor_dim[sensor];++j)
      rows.push_back(m->sensor_adr[sensor]+j);
  }
  return rows;
}
inline void R12AccumulatePath(double* peak,const double* value,
                              const std::vector<int>& rows) {
  // Earliest active substep wins ties in both rollout and differentiation.
  for (int row:rows) if (value[row]>peak[row]) peak[row]=value[row];
}
}
#endif
