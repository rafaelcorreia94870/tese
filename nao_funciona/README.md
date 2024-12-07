Nos seguintes ficheiros encontram-se tentativas de mudar um código host para __device__ em compile time.

Os exemplos vão estar o mais simples possível para facilitar a compreensão.

Todos estes casos dariam se fizessemos a logica da funcao dentro do wrapper, lambda expression ou simplesmente usasse mos __device__ na funçãp.

O que consegui fazer é uma de 2 hipoteses, ou compila e o cuda não altera os valores, ou não compila porque se queixa que o device nao acede à host function.

argument.cu:


argument.cu(58): error: calling a __host__ function("square(float)") from a __device__ function("main::[lambda(float) (instance 1)]::operator () const") is not allowed

argument.cu(58): error: identifier "square" is undefined in device code



lambda.cu:

error: an automatic "__device__" variable declaration is not allowed inside a host function body



wrapper.cu:


error: calling a __host__ function("increment(int)") from a __device__ function("FunctionWrapper<int, & ::increment> ::operator () const") is not allowed

C:\5ano\tese_main\tese\nao_funciona\wrapper.cu(9): error: identifier "increment" is undefined in device code


macro_wrapper.cu:

Compila, mas não altera valores.