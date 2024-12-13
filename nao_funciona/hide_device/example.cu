#include <iostream>
#include <cstdio>

#pragma hd_warning_disable
template<class Function>
__host__ __device__
void invoke(Function f)
{
  f();
}

struct host_only
{
  __host__
  void operator()()
  {
    std::cout << "host_only()" << std::endl;
  }
};

struct device_only
{
  __device__
  void operator()()
  {
    printf("device_only(): thread %d\n", threadIdx.x);
  }
};

__global__
void kernel()
{
  // use from device with device functor
  invoke(device_only());

  // XXX error
  // invoke(host_only());
}

int main()
{
  // use from host with host functor
  

  kernel<<<1,1>>>();
  cudaDeviceSynchronize();

  // XXX error
  // invoke(device_only());

  return 0;
}