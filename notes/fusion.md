old_impl -> old implementation
expr -> experimental implementation using tupples 
gpu_vec -> new vector implementation using composition

all of the implementations have the same occupancy (66.6%) , but the number of registers is higher in the new implementations
without limiting the number of registers and having O3 flag, the old implementation is faster
with the flags, the gpu_vec implementation is faster, in all cases expr is the slowest (which is a shame since it is the most flexible implementation)


antes de usar -O3 --maxrregcount=64
| Kernel Type       | Registers | Stack Frame | Cumulative Stack | cmem\[0] | Notes                             |
| ----------------- | --------- | ----------- | ---------------- | -------- | --------------------------------- |
| `mapKernel`       | 35        | 8 B         | \~96 B           | \~368 B  | Flat computation, simple memory   |
| `kernelUnary`     | 35        | 16 B        | \~376 B          | \~392 B  | Composition of functions          |
| `pipeline_kernel` | 104       | 240 B       | **\~9208 B**     | 456 B    | Heavy inlining, highest resources |
mapKernel = old_impl
kernelUnary = gpu_vec
pipeline_kernel = expr

depois o gpu_vec fica o mais rapido