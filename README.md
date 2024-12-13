# tese

## To do List
- [x] retirar cópia
  - [x] Meter If statement em compile time
  - [x] assumir que é sempre um vetor a usar template specialization
- [ ] Função por referência -> todos os argumentos
- [ ] Coleção por referência
  - [ ] adicionar por referência no kernel
  - [X] aceitar vetor
  - [ ] aceitar array of structs
- [ ] Separar o código em ficheiros
- [X] Fazer testes automaticos -> mais ou menos
- [X] Pesquisar template specialization
- [x] Fazer to do list
- [x] Ver Explicit template instantiation
- [x] ver maneiras de retirar o __device__ e __host__ de funções: NÃO DA
- [ ] suportar compilacao independente
- [x] testar se [] é random iterator -> pelo que percebi é
- [x] ver se o map da com +threadId em vez de usar como um array normal
- [x] ver definição de std::reverse
- [x] Estudar melhor iteradores
- [ ] unzip de iteradores (ver)-> memory coalescing



## Notas
https://stackoverflow.com/questions/61984390/see-reference-to-function-template-instantiation-message-when-catch-stdexcepti

https://stackoverflow.com/questions/2351148/explicit-template-instantiation-when-is-it-used

https://stackoverflow.com/questions/31705764/cuda-c-using-a-template-function-which-calls-a-template-kernel

https://stackoverflow.com/questions/11578381/cuda-kernel-call-in-a-simple-sample

https://stackoverflow.com/questions/67563443/how-can-i-put-function-with-different-parameters-into-map-in-c

https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=function%2520transform#constexpr-functions-and-function-templates

https://github.com/kokkos

ver kokkos, thrust e cub

https://github.com/nvidia/cccl

https://nvidia.github.io/cccl/thrust/api/function_group__transformations_1gabbda6380c902223d777cc72d3b1b9d1a.html


transform.h:

https://nvidia.github.io/cccl/thrust/api/program_listing_file_thrust_transform.h.html

transform.inl (implementação a serio):
https://github.com/ROCm/Thrust/blob/master/thrust/system/detail/generic/transform.inl