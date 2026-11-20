// Deterministic dual active-set projection for min ||tau-target||^2/2,
// a*tau >= b. Every projection starts from an empty working set. QR of
// its at most 12 independent normals is rebuilt after a blocking deletion.
bool project(const Vec& target,Vec& x) const {
  std::vector<size_t> active;
  std::vector<double> lambda;
  x=target;
  for(int iteration=0;iteration<1024;++iteration){
    size_t index=rows.size();double worst=0.;
    for(size_t i=0;i<rows.size();++i){
      const Row& r=rows[i];double slack=dot(r.a,x)-r.b;
      if(slack<worst-1.e-10){worst=slack;index=i;}
    }
    if(index==rows.size()) return true;
    Vec normal=rows[index].a;
    double entering=0.;
    for(;;){
      int n=active.size();
      Vec basis[U]{};
      double upper[U][U]{};
      for(int col=0;col<n;++col){
        Vec v=rows[active[col]].a;
        for(int pass=0;pass<2;++pass)for(int j=0;j<col;++j){
          double h=dot(v,basis[j]);upper[j][col]+=h;
          for(int k=0;k<U;++k)v[k]-=h*basis[j][k];
        }
        upper[col][col]=std::sqrt(dot(v,v));
        if(upper[col][col]<1.e-13)return false;
        for(int k=0;k<U;++k)basis[col][k]=v[k]/upper[col][col];
      }
      Vec z=normal,r{};
      for(int pass=0;pass<2;++pass)for(int j=0;j<n;++j){
        double v=dot(z,basis[j]);r[j]+=v;
        for(int k=0;k<U;++k)z[k]-=v*basis[j][k];
      }
      for(int j=n-1;j>=0;--j){
        for(int k=j+1;k<n;++k)r[j]-=upper[j][k]*r[k];
        r[j]/=upper[j][j];
      }
      double dual_step=std::numeric_limits<double>::infinity();
      int blocker=-1;
      for(int j=0;j<n;++j)if(r[j]>1.e-13){
        double t=std::max(0.,lambda[j])/r[j];
        if(t<dual_step){dual_step=t;blocker=j;}
      }
      double den=dot(z,z);
      double primal_step=den>1.e-24 && n<U ? std::max(0.,rows[index].b-dot(normal,x))/den :
          std::numeric_limits<double>::infinity();
      double step=std::min(primal_step,dual_step);
      if(!std::isfinite(step))return false;
      for(int k=0;k<U;++k)x[k]+=step*z[k];
      for(int j=0;j<n;++j)lambda[j]-=step*r[j];
      entering+=step;
      if(primal_step<=dual_step){active.push_back(index);lambda.push_back(entering);break;}
      active.erase(active.begin()+blocker);lambda.erase(lambda.begin()+blocker);
    }
  }
  return false;
}
