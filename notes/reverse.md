```
template<class BidirIt>
constexpr // since C++20
void reverse(BidirIt first, BidirIt last)
{
    using iter_cat = typename std::iterator_traits<BidirIt>::iterator_category;
 
    // Tag dispatch, e.g. calling reverse_impl(first, last, iter_cat()),
    // can be used in C++14 and earlier modes.
    if constexpr (std::is_base_of_v<std::random_access_iterator_tag, iter_cat>)
    {
        if (first == last)
            return;
 
        for (--last; first < last; (void)++first, --last)
            std::iter_swap(first, last);
    }
    else
        while (first != last && first != --last)
            std::iter_swap(first++, last);
}
```

BidirIt -> Iterador bidirecional

std::iterator_traits -> Extrair informaçoes do iterador ( difference_type, value_type, pointer, reference, iterator_category)

std::iterator_category -> Categoria do iterador (Random Acess, Bidirecional, forward e input)

o if constexpr checka se o iterador é de random acess

    * se for random acess:
        * se a lista estiver vazia ou só tiver um elemento não faz nada
        * --last sets last to the last valid element
        * faz o swap 
        * move o first para a frente e o last para trás
    * se foor biderecional:
        * enquanto o first não for igual ao last ou o last-1 faz swap
