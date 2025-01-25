types of what i could consider a map skeleton, the inputs and outputs could be by reference or with iterators and, as of now I am not considering higher dimension containers, but it could be something to add

---------- Map ---------------

apply functions to each element

- [x] Map (one input, one function) writes in place
- [x] Map (one input, one function, one output)

- [ ] Map (two inputs, one function) writes in place of the first input
- [ ] Map (two inputs, one function, one output)

generic of the above
- [ ] ⏳ In progress ... ⏳ Map (many inputs, one function) writes in place of the first input
- [ ] Map (many inputs, one function, one output)

- [ ] ⏳ In progress ... ⏳ Map (one input, many functions) writes in place (fog)
- [ ] Map (one input, many functions, one output) (fog)

- [ ] Map (two inputs, many functions) writes in place of the first input
- [ ] Map (two inputs, many functions, one output)
generic of the above
- [ ] Map (many inputs, many functions) writes in place of the first input
- [ ] Map (many inputs, many functions, one output)
-------------------------------------

---------- Map Reduce ---------------

apply functions to each element and then reduce the results

- [ ] MapReduce( one input, one function, one Reduce function ) writes in place
- [ ] MapReduce( one input, one function, one Reduce function, one output)

- [ ] MapReduce( two inputs, one function, one Reduce function) writes in place of the first input
- [ ] MapReduce( two inputs, one function, one Reduce function, one output)

genericof the above
- [ ] MapReduce( many inputs, one function, one Reduce function) writes in place of the first input
- [ ] MapReduce( many inputs, one function, one Reduce function, one output)

- [ ] MapReduce( one input, many functions, one Reduce function) writes in place (fog)
- [ ] MapReduce( one input, many functions, one Reduce function, one output) (fog)

- [ ] MapReduce (two inputs, many functions, one Reduce function) writes in place of the first input
- [ ] MapReduce (two inputs, many functions, one Reduce function, one output)

generic of the above
- [ ] MapReduce (many inputs, many functions, one Reduce function) writes in place of the first input
- [ ] MapReduce (many inputs, many functions, one Reduce function, one output)
-------------------------------------

---------- Map Pairs ---------------

cartesian product

- [ ] MapPairs (two inputs, one function, one output)
- [ ] MapPairs (two inputs, many functions, one output) (fog)
-------------------------------------

---------- Map Pairs Reduce---------------

cartesian product and then Reduce

- [ ] MapPairs (two inputs, one - [ ] map function, one Reduce function, one output)
- [ ] MapPairs (two inputs, many functions, one Reduce function,one output) (fog)
-------------------------------------

---------- Map Filter---------------

apply functions to each element and then filter the results with a condition

- [ ] MapFilter (one input, one function, one condition) writes in place
- [ ] MapFilter (one input, one function, one condition, one output)

- [ ] MapFilter (two inputs, one function, one condition) writes in place of the first input
- [ ] MapFilter (two inputs, one function, one condition, one output)
generic of the above
- [ ] MapFilter (many inputs, one function, one condition) writes in place of the first input
- [ ] MapFilter (many inputs, one function, one condition, one output)

- [ ] MapFilter (one input, many functions, one condition) writes in place (fog)
- [ ] MapFilter (one input, many functions, one condition, one output) (fog)

- [ ] MapFilter (two inputs, many functions, one condition) writes in place of the first input
- [ ] MapFilter (two inputs, many functions, one condition, one output)
generic of the above
- [ ] MapFilter (many inputs, many functions, one condition) writes in place of the first input
- [ ] MapFilter (many inputs, many functions, one condition, one output)
-------------------------------------

---------- Map Filter with stencil container---------------

same as Map Filter but checking the condition in another container

- [ ] MapFilter (one input, one function, one stencil, one condition) writes in place
- [ ] MapFilter (one input, one function, one stencil, one condition, one output)

- [ ] MapFilter (two inputs, one function, one stencil, one condition) writes in place of the first input
- [ ] MapFilter (two inputs, one function, one stencil, one condition, one output)
generic of the above
- [ ] MapFilter (many inputs, one function, one stencil, one condition) writes in place of the first input
- [ ] MapFilter (many inputs, one function, one stencil, one condition, one output)

- [ ] MapFilter (one input, many functions, one stencil, one condition) writes in place (fog)
- [ ] MapFilter (one input, many functions, one stencil, one condition, one output) (fog)

- [ ] MapFilter (two inputs, many functions, one stencil, one condition) writes in place of the first input
- [ ] MapFilter (two inputs, many functions, one stencil, one condition, one output)
generic of the above
- [ ] MapFilter (many inputs, many functions, one stencil, one condition) writes in place of the first input
- [ ] MapFilter (many inputs, many functions, one stencil, one condition, one output)

-------------------------------------

---------- Map Filter Reduce---------------

apply functions to each element and then filter the results with a condition and finally reduce the results

all the combinations of the above Map Filters but with a Reduce function

-------------------------------------

---------- Things that could be added ---------------

Stencil

Scan

-------------------------------------



