#pragma once
namespace rafa {

    template <typename T>
    class iterator {
    public:
        using value_type = T;
        using pointer = T*;
        using reference = T&;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::random_access_iterator_tag;
    
        iterator(T* p = nullptr) : ptr(p) {}
        T& operator*() const { return *ptr; }
        iterator& operator++() { ++ptr; return *this; }
        iterator operator++(int) { iterator tmp = *this; ++(*this); return tmp; }
    
        iterator& operator--() { --ptr; return *this; }
        iterator operator--(int) { iterator tmp = *this; --(*this); return tmp; }
    
        iterator operator+(difference_type n) const { return iterator(ptr + n); }
        iterator& operator+=(difference_type n) { ptr += n; return *this; }
    
        iterator operator-(difference_type n) const { return iterator(ptr - n); }
        iterator& operator-=(difference_type n) { ptr -= n; return *this; }
    
        difference_type operator-(const iterator& other) const { return ptr - other.ptr; }
    
        bool operator==(const iterator& other) const { return ptr == other.ptr; }
        bool operator!=(const iterator& other) const { return ptr != other.ptr; }
        bool operator<(const iterator& other) const { return ptr < other.ptr; }
        bool operator<=(const iterator& other) const { return ptr <= other.ptr; }
        bool operator>(const iterator& other) const { return ptr > other.ptr; }
        bool operator>=(const iterator& other) const { return ptr >= other.ptr; }
    
        T* operator->() const { return ptr; }
    
    private:
        T* ptr;
    };
    
    } // namespace rafa
    