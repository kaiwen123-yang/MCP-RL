      if (!path_rows.empty() && t<T-1) {
        const int nx=dim_state_derivative,nu=dim_action,ns=dim_sensor;
        const int nr=path_rows.size();
        const int substeps=GetNumberOrDefault(1,m,"s1_control_hold_substeps");
        std::vector<int> active(nr,0);
        std::vector<double> peak(nr);
        // Locate the same earliest maximizing substep used by the rollout.
        for(int sub=0;sub<substeps;++sub) {
          mj_step(m,d);
          for(int row=0;row<nr;++row)
            if(!sub || d->sensordata[path_rows[row]]>peak[row]) {
              peak[row]=d->sensordata[path_rows[row]];active[row]=sub;
            }
        }
        SetState(m,d,x+t*dim_state);d->time=h[t];
        mju_copy(d->ctrl,u+t*nu,nu);
        double* total_a=DataAt(A,t*nx*nx);
        double* total_b=DataAt(B,t*nx*nu);
        double* total_c=DataAt(C,t*ns*nx);
        double* total_d=DataAt(D,t*ns*nu);
        std::vector<double> prefix_a(nx*nx,0.),prefix_b(nx*nu,0.);
        std::vector<double> next_a(nx*nx),next_b(nx*nu),risk_c(nr*nx);
        for(int j=0;j<nx;++j)prefix_a[j*nx+j]=1.;
        for(int sub=0;sub<substeps;++sub) {
          // Full C/D only at the node start. A/B retain all ten transitions.
          mjd_transitionFD(m,d,tol,mode,next_a.data(),next_b.data(),
                           sub==0 ? total_c : nullptr,sub==0 ? total_d : nullptr);
          if(sub>0 && std::find(active.begin(),active.end(),sub)!=active.end()) {
            R26PathJacobian(m,d,tol,mode,nr,nx,risk_c.data());
            for(int row=0;row<nr;++row)if(active[row]==sub) {
              const int sensor=path_rows[row];
              mju_mulMatMat(total_c+sensor*nx,risk_c.data()+row*nx,
                            prefix_a.data(),1,nx,nx);
              mju_mulMatMat(total_d+sensor*nu,risk_c.data()+row*nx,
                            prefix_b.data(),1,nx,nu);
            }
          }
          mj_step(m,d);
          mju_mulMatMat(total_a,next_a.data(),prefix_a.data(),nx,nx,nx);
          mju_mulMatMat(total_b,next_a.data(),prefix_b.data(),nx,nx,nu);
          mju_addTo(total_b,next_b.data(),nx*nu);
          mju_copy(prefix_a.data(),total_a,nx*nx);
          mju_copy(prefix_b.data(),total_b,nx*nu);
        }
